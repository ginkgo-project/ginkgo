// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_NULL_SPACE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_NULL_SPACE_HPP_


#include <memory>
#include <vector>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * NullSpace represents the (right or left) nullspace of an operator and *is*
 * the orthogonal projector \( P = I - V V^H \) onto its complement, where
 * \( V \) is an orthonormal basis of the nullspace.
 *
 * The projector is assembled only from BLAS-1 operations shared by
 * matrix::Dense and experimental::distributed::Vector (`compute_conj_dot`,
 * `sub_scaled`, `inv_scale`, `compute_norm2`, `fill`), so it needs no dedicated
 * kernels and works for non-distributed and distributed vectors alike; the
 * all-reduce required in the distributed case happens inside
 * `compute_conj_dot`.
 *
 * @note v1 projects a single column (`nrhs == 1`).
 * @note A NullSpace is created once and shared (`shared_ptr<const NullSpace>`);
 *       it binds to the vector type (Dense vs. distributed) it is first applied
 *       to.
 *
 * @tparam ValueType  the value type of the vectors it projects.
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class NullSpace : public LinOp, public EnableCloneable<NullSpace<ValueType>> {
    friend class EnableCloneable<NullSpace>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableCloneable<NullSpace>::convert_to;
    using EnableCloneable<NullSpace>::move_to;

    using value_type = ValueType;

    /**
     * Creates a NullSpace from an explicit basis.
     *
     * @param exec  the executor
     * @param basis  the nullspace basis, each entry an `n x 1` vector
     *               (matrix::Dense or distributed::Vector). Orthonormalized
     *               (modified Gram-Schmidt) on first use; near-dependent
     *               columns are dropped.
     * @param contains_constant  whether the constant vector is also part of
     *                           the nullspace (materialized on first use).
     */
    static std::unique_ptr<NullSpace> create(
        std::shared_ptr<const Executor> exec,
        std::vector<std::shared_ptr<const LinOp>> basis,
        bool contains_constant = false)
    {
        GKO_ASSERT(!basis.empty());
        const auto n = basis[0]->get_size()[0];
        return std::unique_ptr<NullSpace>(
            new NullSpace(std::move(exec), dim<2>{n, n}, std::move(basis),
                          contains_constant));
    }

    /**
     * Creates a NullSpace consisting only of the constant vector (the
     * pure-Neumann case). The normalized all-ones column is materialized to
     * match the vector type/distribution on first use.
     *
     * @param exec  the executor
     * @param size  the (square, global) size `n x n` of the operator.
     */
    static std::unique_ptr<NullSpace> create_from_constant(
        std::shared_ptr<const Executor> exec, dim<2> size)
    {
        return std::unique_ptr<NullSpace>(
            new NullSpace(std::move(exec), size,
                          std::vector<std::shared_ptr<const LinOp>>{}, true));
    }

    /** @return whether the constant vector is part of the nullspace. */
    bool contains_constant() const noexcept { return contains_constant_; }

    /**
     * @return the number of orthonormal basis columns (valid after first use;
     *         a nominal count before).
     */
    size_type get_dimension() const noexcept
    {
        return prepared_ ? basis_.size()
                         : raw_basis_.size() + (contains_constant_ ? 1 : 0);
    }

    /**
     * In-place projector \( v \leftarrow (I - V V^H) v \). Requires `v` to have
     * a single column.
     *
     * @tparam VectorType  matrix::Dense<ValueType> or
     *                     experimental::distributed::Vector<ValueType>.
     */
    template <typename VectorType>
    void project(VectorType* v) const
    {
        if (v->get_size()[1] != 1) {
            GKO_NOT_IMPLEMENTED;
        }
        this->template prepare<VectorType>(v);
        auto coeff = matrix::Dense<ValueType>::create(this->get_executor(),
                                                      dim<2>{1, 1});
        for (const auto& col : basis_) {
            auto b = gko::as<VectorType>(col.get());
            // coeff = b^H v  (all-reduce happens inside for distributed
            // vectors)
            v->compute_conj_dot(b, coeff);
            // v = v - coeff * b
            v->sub_scaled(coeff, b);
        }
    }

protected:
    NullSpace(std::shared_ptr<const Executor> exec, dim<2> size = {})
        : LinOp(std::move(exec), size),
          contains_constant_{false},
          prepared_{false}
    {}

    NullSpace(std::shared_ptr<const Executor> exec, dim<2> size,
              std::vector<std::shared_ptr<const LinOp>> raw_basis,
              bool contains_constant)
        : LinOp(std::move(exec), size),
          contains_constant_{contains_constant},
          raw_basis_{std::move(raw_basis)},
          prepared_{false}
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    /**
     * Builds the orthonormal basis on first use, using `like` (an `n x 1`
     * vector) to fix the value type / distribution of the constant column.
     */
    template <typename VectorType>
    void prepare(const VectorType* like) const
    {
        if (prepared_) {
            return;
        }
        auto exec = this->get_executor();
        std::vector<std::shared_ptr<LinOp>> cols;
        if (contains_constant_) {
            auto ones = VectorType::create_with_config_of(like);
            ones->fill(one<ValueType>());
            auto nrm = matrix::Dense<remove_complex<ValueType>>::create(
                exec, dim<2>{1, 1});
            ones->compute_norm2(nrm);
            ones->inv_scale(nrm);  // normalize to unit length
            cols.push_back(std::move(ones));
        }
        for (const auto& raw : raw_basis_) {
            cols.push_back(gko::clone(gko::as<VectorType>(raw.get())));
        }

        // modified Gram-Schmidt orthonormalization
        auto coeff = matrix::Dense<ValueType>::create(exec, dim<2>{1, 1});
        auto nrm = matrix::Dense<remove_complex<ValueType>>::create(
            exec, dim<2>{1, 1});
        auto host_nrm = matrix::Dense<remove_complex<ValueType>>::create(
            exec->get_master(), dim<2>{1, 1});
        std::vector<std::shared_ptr<const LinOp>> ortho;
        for (auto& c : cols) {
            auto cv = gko::as<VectorType>(c.get());
            for (const auto& q : ortho) {
                auto qv = gko::as<VectorType>(q.get());
                cv->compute_conj_dot(qv, coeff);  // coeff = q^H c
                cv->sub_scaled(coeff, qv);
            }
            cv->compute_norm2(nrm);
            host_nrm->copy_from(nrm);
            if (host_nrm->at(0, 0) > remove_complex<ValueType>{1e-10}) {
                cv->inv_scale(nrm);
                ortho.push_back(std::move(c));
            }
            // else: near-dependent column, dropped
        }
        basis_ = std::move(ortho);
        prepared_ = true;
    }

private:
    bool contains_constant_;
    std::vector<std::shared_ptr<const LinOp>> raw_basis_;
    mutable std::vector<std::shared_ptr<const LinOp>> basis_;
    mutable bool prepared_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_NULL_SPACE_HPP_
