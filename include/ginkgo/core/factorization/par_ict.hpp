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

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_PAR_ICT_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_PAR_ICT_HPP_


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
 * ParICT is an incomplete threshold-based Cholesky factorization which is
 * computed in parallel.
 *
 * $L$ is a lower triangular matrix which approximates a given symmetric
 * positive definite matrix $A$ with $A \approx LL^T$. Here, $L$ has a sparsity
 * pattern that is improved iteratively based on its element-wise magnitude.
 * The initial sparsity pattern is chosen based on the lower triangle of $A$.
 *
 * One iteration of the ParICT algorithm consists of the following steps:
 *
 * 1. Calculating the residual $R = A - LL^T$
 * 2. Adding new non-zero locations from $R$ to $L$.
 *    The new non-zero locations are initialized based on the corresponding
 *    residual value.
 * 3. Executing a fixed-point iteration on $L$ according to
 * $
 * F(L) =
 * \begin{cases}
 *     \frac{1}{l_{jj}}
 *         \left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}l_{jk}\right), \quad & i \neq j \\
 *     \sqrt{a_{ij}-\sum_{k=1}^{j-1}l_{ik}l_{jk}}, \quad & i = j \\
 * \end{cases}
 * $
 * 4. Removing the smallest entries (by magnitude) from $L$
 * 5. Executing a fixed-point iteration on the (now sparser) $L$
 *
 * This ParICT algorithm thus improves the sparsity pattern and the
 * approximation of $L$ simultaneously.
 *
 * The implementation follows the design of H. Anzt et al.,
 * ParILUT - A Parallel Threshold ILU for GPUs, 2019 IEEE International
 * Parallel and Distributed Processing Symposium (IPDPS), pp. 231–241.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIct : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const matrix_type> get_lt_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
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
         * The number of total iterations of ParICT that will be executed.
         * The default value is 5.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(iterations, 5);

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
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /**
         * @brief `true` means the candidate selection will use an inexact
         * selection algorithm. `false` means an exact selection algorithm will
         * be used.
         *
         * Using the approximate selection algorithm can give a significant
         * speed-up, but may in the worst case cause the algorithm to vastly
         * exceed its `fill_in_limit`.
         * The exact selection needs more time, but more closely fulfills the
         * `fill_in_limit` except for pathological cases (many candidates with
         * equal magnitude).
         *
         * The default behavior is to use approximate selection.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(approximate_select, true);

        /**
         * @brief `true` means the sample used for the selection algorithm will
         *        be chosen deterministically. This is only relevant when using
         *        `approximate_select`. It is mostly used for testing.
         *
         * The selection algorithm used for `approximate_select` uses a small
         * sample of the input data to determine an approximate threshold.
         * The choice of elements can either be randomized, i.e., we may use
         * different elements during each execution, or deterministic, i.e., the
         * element choices are always the same.
         *
         * Note that even though the threshold selection step may be made
         * deterministic this way, the calculation of the IC factors can still
         * be non-deterministic due to its asynchronous iterations.
         *
         * The default behavior is to use a random sample.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(deterministic_sample, false);

        /**
         * @brief the amount of fill-in that is allowed in L compared to
         *        the lower triangle of A.
         *
         * The threshold for removing candidates from the intermediate L
         * is set such that the resulting sparsity pattern has at most
         * `fill_in_limit` times the number of non-zeros of the lower triangle
         * of A factorization..
         *
         * The default value `2.0` allows twice the number of non-zeros in
         * L compared to the lower triangle of A.
         */
        double GKO_FACTORY_PARAMETER_SCALAR(fill_in_limit, 2.0);

        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * Strategy which will be used by the L^T matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(lt_strategy, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParIct, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ParIct(const Factory *factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        if (parameters_.lt_strategy == nullptr) {
            parameters_.lt_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate_l_lt(std::move(system_matrix))->move_to(this);
    }

    /**
     * Generates the incomplete LL^T factors, which will be returned as a
     * composition of the lower (first element of the composition) and the
     * upper factor (second element). The dynamic type of L and L^T is
     * matrix_type
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertable to a Csr
     *                              Matrix, otherwise, an exception is thrown.
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then L^T)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_lt(
        const std::shared_ptr<const LinOp> &system_matrix) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_PAR_ICT_HPP_
