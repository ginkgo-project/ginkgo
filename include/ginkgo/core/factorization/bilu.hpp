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

#ifndef GKO_CORE_FACTORIZATION_BLOCK_ILU_HPP_
#define GKO_CORE_FACTORIZATION_BLOCK_ILU_HPP_


#include <memory>


#include <ginkgo/core/base/lin_op.hpp>
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
 * Represents an incomplete block LU factorization -- BILU(0) -- of a
 * sparse matrix with a block structure with small dense blocks
 *
 * More specifically, it consists of a block lower unitriangular factor $L$ and
 * a block upper triangular factor $U$ with sparsity pattern
 * $\mathcal S(L + U)$ = $\mathcal S(A)$
 * fulfilling $LU = A$ at every non-zero location of $A$.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Bilu : public EnableLinOp<Bilu<ValueType, IndexType>>,
             public Transposable {
    friend class EnablePolymorphicObject<Bilu, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Fbcsr<ValueType, IndexType>;
    using u_matrix_type = matrix::Fbcsr<ValueType, IndexType>;
    using l_matrix_type = matrix::Fbcsr<ValueType, IndexType>;

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
         * it must remain `false`, otherwise, this factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Bilu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    struct factors {
        std::shared_ptr<matrix_type> l_factor;
        std::shared_ptr<matrix_type> u_factor;
    };

    explicit Bilu(const std::shared_ptr<const Executor> exec)
        : EnableLinOp<Bilu>(exec)
    {}

    Bilu(const Factory *const factory,
         const std::shared_ptr<const gko::LinOp> system_matrix)
        : EnableLinOp<Bilu>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        factors_ = generate_block_LU(system_matrix, parameters_.skip_sorting);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * @brief Generates the incomplete block LU factors
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertible to
     *                              a Fbcsr matrix, otherwise,
     *                              an exception is thrown.
     * @return The incomplete block LU factors for the
     *         given system_matrix (first element is L, then U)
     */
    factors generate_block_LU(std::shared_ptr<const LinOp> system_matrix,
                              bool skip_sorting) const;

    factors factors_;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_BLOCK_ILU_HPP_
