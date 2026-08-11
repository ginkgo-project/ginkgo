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
 * The EnableNullspaceRemoval mixin equips a LinOp with an (optional) nullspace
 * that is automatically projected out of the result of every `apply`.
 *
 * A nullspace is a single vector `n` spanning a one-dimensional subspace that
 * should not appear in the operator's output (e.g. the constant vector for a
 * pure-Neumann problem). Once a nullspace is attached, the concrete operator is
 * expected to call remove_nullspace() at the end of its apply_impl(), which
 * performs the orthogonal projection
 * \f[
 *     x_j \leftarrow x_j - (n^H x_j)\, n
 * \f]
 * for every column \f$x_j\f$ of the output. The attached nullspace is always
 * normalized to \f$\|n\|_2 = 1\f$ at attach time, so the projection does not
 * need to divide by \f$n^H n\f$.
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
class EnableNullspaceRemoval {
public:
    using vector_type = VectorType;
    using value_type = typename VectorType::value_type;

    virtual ~EnableNullspaceRemoval() = default;

    /**
     * Attaches an arbitrary vector as the nullspace to be removed after every
     * apply. The vector is copied and normalized to unit 2-norm internally, so
     * the caller retains ownership of the passed object and may modify it
     * afterwards.
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
     * Attaches the constant (all-ones) vector as the nullspace to be removed
     * after every apply. The vector is normalized to unit 2-norm internally,
     * i.e. every entry becomes \f$1/\sqrt{N}\f$ with \f$N\f$ the global size.
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

    /** @return whether a nullspace is currently attached */
    bool has_nullspace() const { return static_cast<bool>(nullspace_); }

    /** Detaches any currently attached nullspace. */
    void clear_nullspace() { nullspace_.reset(); }

    /**
     * Projects the attached nullspace out of `x` in place. Does nothing if no
     * nullspace is attached. This is called automatically at the end of the
     * operator's apply, but is also exposed publicly (analogous to PETSc's
     * MatNullSpaceRemove) so callers can project other vectors, e.g. a
     * right-hand side or a preconditioner's output.
     *
     * @param x  the vector to project; must be of the operator's VectorType
     */
    void remove_nullspace(ptr_param<LinOp> x) const
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
     * multi-column applies.
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
