// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_IC_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_IC_HPP_


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
 * Represents an incomplete Cholesky factorization (IC(0)) of a sparse matrix.
 *
 * More specifically, it consists of a lower triangular factor $L$ and
 * its conjugate transpose $L^H$ with sparsity pattern
 * $\mathcal S(L + L^H)$ = $\mathcal S(A)$
 * fulfilling $LL^H = A$ at every non-zero location of $A$.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Ic : public Composition<ValueType> {
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
        if (this->get_operators().size() == 2) {
            // Can be `static_cast` since the type is guaranteed in this class
            return std::static_pointer_cast<const matrix_type>(
                this->get_operators()[1]);
        } else {
            return std::static_pointer_cast<const matrix_type>(
                share(get_l_factor()->conj_transpose()));
        }
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

        /**
         * `true` will generate both L and L^H, `false` will only generate the L
         * factor, resulting in a Composition of only a single LinOp. This can
         * be used to avoid the transposition operation.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(both_factors, true);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Ic, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    Ic(const Factory* factory, std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate(system_matrix, parameters_.skip_sorting,
                 parameters_.both_factors)
            ->move_to(this);
    }

    std::unique_ptr<Composition<ValueType>> generate(
        const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
        bool both_factors) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_IC_HPP_
