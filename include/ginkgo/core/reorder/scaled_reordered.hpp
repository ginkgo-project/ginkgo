// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_SCALED_REORDERED_HPP_
#define GKO_PUBLIC_CORE_REORDER_SCALED_REORDERED_HPP_


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>


namespace gko {
namespace experimental {
namespace reorder {


/**
 * Provides an interface to wrap reorderings like Rcm and diagonal scaling
 * like equilibration around a LinOp like e.g. a sparse direct solver.
 *
 * Reorderings can be useful for reducing fill-in in the numerical factorization
 * phase of direct solvers, diagonal scaling can help improve the numerical
 * stability by reducing the condition number of the system matrix.
 *
 * With a permutation matrix P, a row scaling R and a column scaling C, the
 * inner operator is applied to the system matrix P*R*A*C*P^T instead of A.
 * Instead of A*x = b, the inner operator attempts to solve the equivalent
 * linear system P*R*A*C*P^T*y = P*R*b and retrieves the solution x = C*P^T*y.
 * Note: The inner system matrix is computed from a clone of A, so the original
 * system matrix is not changed.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ScaledReordered : public LinOp, public gko::detail::WorkspaceHolder {
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using ReorderingBaseFactory =
        AbstractFactory<gko::reorder::ReorderingBase<IndexType>,
                        gko::reorder::ReorderingBaseArgs>;

    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    std::shared_ptr<const LinOp> get_inner_operator() const
    {
        return inner_operator_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The inner operator factory that is to be generated on the scaled
         * and reordered system matrix.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            inner_operator, nullptr);

        /**
         * The reordering that is to be applied to the system matrix.
         * If a reordering is provided, the system matrix must be of type
         * `Permutable<IndexType>`.
         */
        std::shared_ptr<const ReorderingBaseFactory>
            GKO_FACTORY_PARAMETER_SCALAR(reordering, nullptr);

        /**
         * The row scaling that is to be applied to the system matrix.
         */
        std::shared_ptr<const matrix::Diagonal<value_type>>
            GKO_FACTORY_PARAMETER_SCALAR(row_scaling, nullptr);

        /**
         * The column scaling that is to be applied to the system matrix.
         */
        std::shared_ptr<const matrix::Diagonal<value_type>>
            GKO_FACTORY_PARAMETER_SCALAR(col_scaling, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ScaledReordered, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Creates an empty scaled reordered operator (0x0 operator).
     */
    explicit ScaledReordered(std::shared_ptr<const Executor> exec)
        : LinOp(std::move(exec)),
          gko::detail::WorkspaceHolder(this->get_executor()),
          permutation_array_{this->get_executor()}
    {}

    explicit ScaledReordered(const Factory* factory,
                             LinOpGenerateComponents components)
        : LinOp(factory->get_executor(), components.system_matrix->get_size()),
          gko::detail::WorkspaceHolder(factory->get_executor(), components),
          parameters_{factory->get_parameters()},
          permutation_array_{factory->get_executor()}
    {
        auto system_matrix = std::move(components.system_matrix);
        // For now only support square matrices.
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

        auto exec = this->get_executor();

        system_matrix_ = gko::clone(exec, system_matrix);

        // Scale the system matrix if scaling coefficients are provided
        if (parameters_.row_scaling) {
            GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.row_scaling,
                                        system_matrix_);
            row_scaling_ = parameters_.row_scaling;
            row_scaling_->apply(system_matrix_, system_matrix_);
        }
        if (parameters_.col_scaling) {
            GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.col_scaling,
                                        system_matrix_);
            col_scaling_ = parameters_.col_scaling;
            col_scaling_->rapply(system_matrix_, system_matrix_);
        }

        // If a reordering factory is provided, generate the reordering and
        // permute the system matrix accordingly.
        if (parameters_.reordering) {
            auto reordering = parameters_.reordering->generate(system_matrix_);
            permutation_array_ = reordering->get_permutation_array();
            system_matrix_ = as<Permutable<index_type>>(system_matrix_)
                                 ->permute(&permutation_array_);
        }

        // Generate the inner operator with the scaled and reordered system
        // matrix. If none is provided, use the Identity.
        if (parameters_.inner_operator) {
            inner_operator_ = gko::detail::generate_with_view(
                parameters_.inner_operator.get(), system_matrix_,
                this->get_workspace_node()->get_or_create_child(
                    "inner_operator"));
        } else {
            inner_operator_ = gko::matrix::Identity<value_type>::create(
                exec, this->get_size()[0]);
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    /** The vectors the inner operator works on, from the workspace. */
    struct inner_vectors {
        matrix::Dense<value_type>* b;
        matrix::Dense<value_type>* x;
        matrix::Dense<value_type>* intermediate;
    };

    /**
     * Returns the intermediate right hand side, solution and intermediate
     * vectors for the inner operator, sized like `b` and `x`, with `b` (and
     * `x`, if the inner operator uses an initial guess) copied in.
     */
    inner_vectors set_cache_to(const LinOp* b, const LinOp* x) const
    {
        using vec = matrix::Dense<value_type>;
        this->set_workspace_size(3, 0);
        const auto size = b->get_size();
        auto get = [&](int id) {
            return this->template create_workspace_op_if<vec>(
                id, [size](const vec& v) { return v.get_size() != size; },
                [&] { return vec::create(this->get_executor(), size); });
        };
        inner_vectors vecs{get(0), get(1), get(2)};
        vecs.b->copy_from(as<Cloneable>(b));
        if (inner_operator_->apply_uses_initial_guess()) {
            vecs.x->copy_from(as<Cloneable>(x));
        }
        return vecs;
    }

private:
    std::shared_ptr<LinOp> system_matrix_{};
    std::shared_ptr<const LinOp> inner_operator_{};
    std::shared_ptr<const matrix::Diagonal<value_type>> row_scaling_{};
    std::shared_ptr<const matrix::Diagonal<value_type>> col_scaling_{};
    array<index_type> permutation_array_{};
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_SCALED_REORDERED_HPP_
