/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_PAR_BILU_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_PAR_BILU_HPP_


#include <memory>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * ParBILU is an approximate incomplete block LU factorization computed
 * in parallel.
 *
 * $L$ is a block lower unit triangular, while $U$ is a block upper triangular
 * matrix, which approximate a given matrix $A$ with $A \approx LU$. Here, $L$
 * and $U$ have the same sparsity pattern as $A$, which is also called BILU(0).
 *
 * The ParBILU algorithm generates the incomplete factors iteratively, using a
 * fixed-point iteration of the form
 *
 * $
 * F(L, U) =
 * \begin{cases}
 *     \left(A_{ij}-\sum_{k=1}^{j-1}L_{ik}U_{kj}\right)U_{jj}^{-1}, \quad & i>j
 * \\ A_{ij}-\sum_{k=1}^{i-1}L_{ik}U_{kj}, \quad & i\leq j \end{cases}
 * $
 *
 * In general, the entries of $L$ and $U$ can be iterated in parallel and in
 * asynchronous fashion, the algorithm asymptotically converges to the
 * incomplete factors $L$ and $U$ fulfilling $\left(R = A - L \cdot
 * U\right)\vert_\mathcal{S} = 0\vert_\mathcal{S}$ where $\mathcal{S}$ is the
 * pre-defined sparsity pattern (in case of BILU(0) the sparsity pattern of the
 * system matrix $A$). The number of ParBILU sweeps needed for convergence
 * depends on the parallelism level: For sequential execution, a single sweep
 * is sufficient, for fine-grained parallelism, the number of sweeps necessary
 * to get a good approximation of the incomplete factors depends heavily on the
 * problem. On the OpenMP executor, 3 sweeps usually give a decent approximation
 * in our experiments, while GPU executors can take 10 or more iterations.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParBilu : public EnableLinOp<ParBilu<ValueType, IndexType>>,
                public Transposable {
    friend class EnablePolymorphicObject<ParBilu, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Fbcsr<ValueType, IndexType>;
    using l_matrix_type = matrix_type;
    using u_matrix_type = matrix_type;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        return factors_.l_factor;
    }

    std::shared_ptr<const matrix_type> get_u_factor() const
    {
        return factors_.u_factor;
    }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The number of iterations the `compute` kernel will use when doing
         * the factorization. The default value `-1` means `Auto`, so the
         * implementation decides on the actual value depending on the
         * ressources that are available.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(iterations, -1);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, the factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParBilu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    struct factors {
        std::shared_ptr<matrix_type> l_factor;
        std::shared_ptr<matrix_type> u_factor;
    };

    explicit ParBilu(const std::shared_ptr<const Executor> exec)
        : EnableLinOp<ParBilu>(exec)
    {}

    explicit ParBilu(const Factory *const factory,
                     const std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<ParBilu>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        // generate_block_lu(system_matrix, parameters_.skip_sorting)
        //     ->move_to(factors_);
        factors_ = generate_block_lu(system_matrix, parameters_.skip_sorting);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * @brief Generates the incomplete block LU factors
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertible to a Fbcsr
     *                              Matrix, otherwise, an exception is thrown.
     * @param skip_sorting  if set to `true`, the sorting will be skipped.
     *                      @note: If the matrix is not sorted, the
     *                             factorization fails.
     * @return  The incomplete block LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    factors generate_block_lu(const std::shared_ptr<const LinOp> system_matrix,
                              bool skip_sorting) const;

    factors factors_;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_PAR_ILU_HPP_
