// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_ILU_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_ILU_HPP_


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
 * Represents an incomplete LU factorization -- ILU(0) -- of a sparse matrix.
 *
 * More specifically, it consists of a lower unitriangular factor $L$ and
 * an upper triangular factor $U$ with sparsity pattern
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
class Ilu : public Composition<ValueType> {
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

    std::shared_ptr<const matrix_type> get_u_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[1]);
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args&&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * Strategy which will be used by the U matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(u_strategy, nullptr);

        /**
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
    GKO_ENABLE_LIN_OP_FACTORY(Ilu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    Ilu(const Factory* factory, std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        if (parameters_.u_strategy == nullptr) {
            parameters_.u_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate_l_u(system_matrix, parameters_.skip_sorting)->move_to(this);
    }

    /**
     * Generates the incomplete LU factors, which will be returned as a
     * composition of the lower (first element of the composition) and the
     * upper factor (second element). The dynamic type of L is l_matrix_type,
     * while the dynamic type of U is u_matrix_type.
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertible to a Csr
     *                              Matrix, otherwise, an exception is thrown.
     * @param skip_sorting  determines if the sorting of system_matrix can be
     *                      skipped (therefore, marking that it is already
     *                      sorted)
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        const std::shared_ptr<const LinOp>& system_matrix,
        bool skip_sorting) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_ILU_HPP_
