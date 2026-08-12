// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_NULLSPACE_REMOVABLE_HPP_
#define GKO_PUBLIC_CORE_BASE_NULLSPACE_REMOVABLE_HPP_


#include <memory>

#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * A non-templated interface for operators that carry an (optional) nullspace.
 *
 * The nullspace is metadata: it is *not* applied inside the operator's apply.
 * Instead, iterative solvers query it (analogous to PETSc's MatSetNullSpace)
 * and project it out of the right-hand side and of the iterates themselves, so
 * that a singular but consistent system \f$A x = b\f$ (e.g. a pure-Neumann
 * problem, whose nullspace is the constant vector) is solved for the
 * minimum-norm solution orthogonal to the nullspace.
 *
 * This base class exposes only the type-erased query/projection entry points so
 * that a solver can handle any operator uniformly via `dynamic_cast`, without
 * knowing its value or vector type. Operators obtain a working implementation by
 * inheriting the EnableNullspaceRemoval mixin.
 *
 * @ingroup LinOp
 */
class NullspaceRemovable {
public:
    virtual ~NullspaceRemovable() = default;

    /** @return whether a nullspace is currently attached */
    virtual bool has_nullspace() const = 0;

    /**
     * Projects the attached nullspace out of `x` in place (like PETSc's
     * MatNullSpaceRemove). Does nothing if no nullspace is attached.
     *
     * @param x  the vector to project; must match the operator's vector type
     */
    virtual void remove_nullspace(ptr_param<LinOp> x) const = 0;
};


/**
 * The EnableNullspaceRemoval mixin equips a LinOp with an (optional) nullspace
 * stored as metadata, and implements the NullspaceRemovable interface.
 *
 * A nullspace is a single vector `n` spanning a one-dimensional subspace that a
 * solver should project out of the right-hand side and the iterates (e.g. the
 * constant vector for a pure-Neumann problem). The projection performed by
 * remove_nullspace() is the orthogonal projection
 * \f[
 *     x_j \leftarrow x_j - (n^H x_j)\, n
 * \f]
 * for every column \f$x_j\f$. The attached nullspace is normalized to
 * \f$\|n\|_2 = 1\f$ at attach time, so the projection does not need to divide
 * by \f$n^H n\f$.
 *
 * The nullspace can be either an arbitrary user-provided vector
 * (see set_nullspace()) or the constant all-ones vector (see
 * set_constant_nullspace()). In both cases the stored vector is a single-column
 * VectorType matching the operator's domain (a matrix::Dense in the
 * non-distributed case, an experimental::distributed::Vector otherwise).
 *
 * Concrete operators opt in by inheriting from this mixin and implementing the
 * two protected hooks create_constant_nullspace() and
 * create_nullspace_column_view().
 *
 * @tparam VectorType  the concrete (multi-)vector type the operator applies to,
 *                     e.g. matrix::Dense<ValueType> or
 *                     experimental::distributed::Vector<ValueType>
 *
 * @ingroup LinOp
 */
template <typename VectorType>
class EnableNullspaceRemoval : public NullspaceRemovable {
public:
    using vector_type = VectorType;
    using value_type = typename VectorType::value_type;

    /**
     * Attaches an arbitrary vector as the nullspace. The vector is copied and
     * normalized to unit 2-norm internally, so the caller retains ownership of
     * the passed object and may modify it afterwards.
     *
     * @param nullspace  a single-column vector of the operator's VectorType
     */
    void set_nullspace(std::shared_ptr<const LinOp> nullspace)
    {
        GKO_ASSERT(nullspace != nullptr);
        auto vec = gko::clone(as<const vector_type>(nullspace.get()));
        GKO_ASSERT_EQ(vec->get_size()[1], 1);
        this->normalize_nullspace(vec.get());
        nullspace_ = std::move(vec);
    }

    /**
     * Attaches the constant (all-ones) vector as the nullspace. The vector is
     * normalized to unit 2-norm internally, i.e. every entry becomes
     * \f$1/\sqrt{N}\f$ with \f$N\f$ the global size.
     */
    void set_constant_nullspace()
    {
        auto vec = this->create_constant_nullspace();
        this->normalize_nullspace(vec.get());
        nullspace_ = std::move(vec);
    }

    /**
     * @return the currently attached (normalized) nullspace, or `nullptr` if
     *         none is attached
     */
    std::shared_ptr<const vector_type> get_nullspace() const
    {
        return nullspace_;
    }

    /** Detaches any currently attached nullspace. */
    void clear_nullspace() { nullspace_.reset(); }

    bool has_nullspace() const override
    {
        return static_cast<bool>(nullspace_);
    }

    void remove_nullspace(ptr_param<LinOp> x) const override
    {
        if (!nullspace_) {
            return;
        }
        auto vec_x = dynamic_cast<vector_type*>(x.get());
        GKO_ASSERT(vec_x != nullptr);
        auto exec = vec_x->get_executor();
        const auto n = nullspace_.get();
        const auto num_cols = vec_x->get_size()[1];
        auto coef = matrix::Dense<value_type>::create(exec, dim<2>{1, 1});
        if (num_cols == 1) {
            // coef = n^H x
            vec_x->compute_conj_dot(n, coef);
            // x <- x - coef * n
            vec_x->sub_scaled(coef, n);
        } else {
            for (size_type col = 0; col < num_cols; ++col) {
                auto x_col = this->create_nullspace_column_view(vec_x, col);
                x_col->compute_conj_dot(n, coef);
                x_col->sub_scaled(coef, n);
            }
        }
    }

protected:
    /**
     * Creates the constant all-ones nullspace vector matching this operator's
     * domain (a single-column VectorType with the operator's row distribution).
     * The returned vector is normalized by the caller.
     */
    virtual std::unique_ptr<vector_type> create_constant_nullspace() const = 0;

    /**
     * Creates a single-column view of column `col` of `x` that shares its
     * memory, so that in-place updates on the view modify `x`. Only used for
     * multi-column right-hand sides.
     *
     * @param x  the (multi-)vector to take a column view of
     * @param col  the column index
     */
    virtual std::unique_ptr<vector_type> create_nullspace_column_view(
        vector_type* x, size_type col) const = 0;

private:
    /** Scales `v` in place so that its global 2-norm becomes 1. */
    void normalize_nullspace(vector_type* v) const
    {
        auto exec = v->get_executor();
        auto norm = matrix::Dense<remove_complex<value_type>>::create(
            exec, dim<2>{1, 1});
        v->compute_norm2(norm);
        auto host_norm = make_temporary_clone(exec->get_master(), norm.get());
        const auto norm_val = host_norm->at(0, 0);
        auto inv_scalar = matrix::Dense<value_type>::create(exec, dim<2>{1, 1});
        inv_scalar->fill(one<value_type>() / static_cast<value_type>(norm_val));
        v->scale(inv_scalar);
    }

    std::shared_ptr<const vector_type> nullspace_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_NULLSPACE_REMOVABLE_HPP_
