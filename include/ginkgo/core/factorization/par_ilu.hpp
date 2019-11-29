/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_FACTORIZATION_PAR_ILU_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILU_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * ParILU is an incomplete LU factorization which is computed in parallel.
 *
 * $L$ is a lower unitriangular, while $U$ is an upper triangular matrix, which
 * approximate a given matrix $A$ with $A \approx LU$. Here, $L$ and $U$ have
 * the same sparsity pattern as $A$, which is also called ILU(0).
 *
 * The ParILU algorithm generates the incomplete factors iteratively, using a
 * fixed-point iteration of the form
 *
 * $
 * F(L, U) =
 * \begin{cases}
 *     \frac{1}{u_{jj}}
 *         \left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}\right), \quad & i>j \\
 *     a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj}, \quad & i\leq j
 * \end{cases}
 * $
 *
 * In general, the entries of $L$ and $U$ can be iterated in parallel and in
 * asynchronous fashion, the algorithm asymptotically converges to the
 * incomplete factors $L$ and $U$ fulfilling $\left(R = A - L \cdot
 * U\right)\vert_\mathcal{S} = 0\vert_\mathcal{S}$ where $\mathcal{S}$ is the
 * pre-defined sparsity pattern (in case of ILU(0) the sparsity pattern of the
 * system matrix $A$). The number of ParILU sweeps needed for convergence
 * depends on the parallelism level: For sequential execution, a single sweep
 * is sufficient, for fine-grained parallelism, 3 sweeps are typically
 * generating a good approximation.
 *
 * The ParILU algorithm in Ginkgo follows the design of E. Chow and A. Patel,
 * Fine-grained Parallel Incomplete LU Factorization, SIAM Journal on Scientific
 * Computing, 37, C169-C193 (2015).
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIlu : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using l_matrix_type = matrix::Csr<ValueType, IndexType>;
    using u_matrix_type = matrix::Csr<ValueType, IndexType>;

    std::shared_ptr<const l_matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const l_matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const u_matrix_type> get_u_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const u_matrix_type>(
            this->get_operators()[1]);
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args &&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The number of iterations the `compute` kernel will use when doing
         * the factorization. The default value `0` means `Auto`, so the
         * implementation decides on the actual value depending on the
         * ressources that are available.
         */
        size_type GKO_FACTORY_PARAMETER(iterations, 0);

        /**
         * @brief `true` means it is known that the matrix given to this
         *        factory will be sorted first by row, then by column index,
         *        `false` means it is unknown or not sorted, so an additional
         *        sorting step will be performed during the factorization
         *        (it will not change the matrix given).
         *        The matrix must be sorted for this factorization to work.
         *
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, the factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ParIlu(const Factory *factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        generate_l_u(system_matrix, parameters_.skip_sorting)->move_to(this);
    }

    /**
     * Generates the incomplete LU factors, which will be returned as a
     * composition of the lower (first element of the composition) and the
     * upper factor (second element). The dynamic type of L is l_matrix_type,
     * while the dynamic type of U is u_matrix_type.
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertable to a Csr
     *                              Matrix, otherwise, an exception is thrown.
     * @param skip_sorting  if set to `true`, the sorting will be skipped.
     *                      @note: If the matrix is not sorted, the
     *                             factorization fails.
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        const std::shared_ptr<const LinOp> &system_matrix,
        bool skip_sorting) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILU_HPP_
