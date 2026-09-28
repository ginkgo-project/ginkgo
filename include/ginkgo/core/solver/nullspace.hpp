// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_NULLSPACE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_NULLSPACE_HPP_


#include <memory>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace solver {


/**
 * Nullspace represents the (right or left) nullspace of an operator and *is*
 * the orthogonal projector \( P = I - V V^H \) onto its complement, where
 * \( V \) is an orthonormal basis of the nullspace.
 *
 * - The constant vector is handled implicitly by subtracting the mean, so a
 *   constant-only nullspace works with any vector layout (`matrix::Dense` or
 *   `experimental::distributed::Vector` with any partition).
 * - An explicit basis is copied and orthonormalized once at creation time
 *   (twice-iterated Gram-Schmidt, also against the constant if it is part of
 *   the nullspace). Columns that are numerically linearly dependent (relative
 *   norm below \( \sqrt{\epsilon} \) after orthogonalization) are dropped;
 *   get_dimension() reports the remaining dimension.
 * - project() removes the nullspace components of all columns of a vector at
 *   once: the coefficients \( C = V^H X \) and the column means are computed
 *   in a single pass (and, for distributed vectors, a single all-reduce),
 *   followed by \( X \leftarrow X - V C \).
 *
 * The typical use is as the `nullspace` / `left_nullspace` parameter of an
 * iterative solver, see
 * enable_preconditioned_iterative_solver_factory_parameters.
 *
 * The object is immutable after creation apart from internal scratch buffers,
 * so it can be shared between solvers (e.g. as both the left and right
 * nullspace of a symmetric operator). Like other Ginkgo LinOps with internal
 * caches, a single instance must not be applied concurrently from multiple
 * threads.
 *
 * @tparam ValueType  the value type of the vectors it projects.
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Nullspace : public LinOp, public EnableCloneable<Nullspace<ValueType>> {
    friend class EnableCloneable<Nullspace>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableCloneable<Nullspace>::convert_to;
    using EnableCloneable<Nullspace>::move_to;

    using value_type = ValueType;
    using absolute_type = remove_complex<ValueType>;

    /**
     * Creates a Nullspace from an explicit basis.
     *
     * @param exec  the executor the projector (and its basis) lives on
     * @param basis  the nullspace basis vectors. Each entry is an `n x k_i`
     *               matrix::Dense<ValueType> or
     *               experimental::distributed::Vector<ValueType> (all entries
     *               of the same kind and, if distributed, the same
     *               partition); their columns together span the nullspace.
     *               They do not need to be orthonormal: they are copied and
     *               orthonormalized, dropping numerically dependent columns.
     * @param contains_constant  whether the constant vector is also part of
     *                           the nullspace.
     */
    static std::unique_ptr<Nullspace> create(
        std::shared_ptr<const Executor> exec,
        std::vector<std::shared_ptr<const LinOp>> basis,
        bool contains_constant = false);

    /**
     * Creates a Nullspace consisting only of the constant vector (e.g. a
     * pure-Neumann Poisson problem). The constant is never materialized, so
     * the result can project vectors of any type and distribution.
     *
     * Solvers adapt a constant-only nullspace to the size of their system
     * matrix, so the same object can be used in solver factories that are
     * generated for matrices of different sizes (e.g. on the levels of a
     * multigrid hierarchy). The size only matters when using the Nullspace
     * directly, and can be omitted otherwise.
     *
     * @param exec  the executor
     * @param size  the (square, global) size `n x n` of the operator.
     */
    static std::unique_ptr<Nullspace> create_from_constant(
        std::shared_ptr<const Executor> exec, dim<2> size = {});

    /** @return whether the constant vector is part of the nullspace. */
    bool contains_constant() const noexcept { return contains_constant_; }

    /**
     * @return the dimension of the nullspace: the number of orthonormal
     *         explicit basis vectors, plus one if the constant is included.
     */
    size_type get_dimension() const noexcept
    {
        return get_num_basis_vectors() + (contains_constant_ ? 1 : 0);
    }

    /**
     * @return the orthonormalized explicit basis as a single `n x k` vector
     *         of the same kind as the input basis (orthogonal to the constant
     *         if contains_constant()), or nullptr if there is none.
     */
    std::shared_ptr<const LinOp> get_basis() const noexcept { return basis_; }

    /**
     * Applies the projector in place: \( v \leftarrow (I - V V^H) v \), for
     * each column of `v` independently.
     *
     * @param v  a matrix::Dense<ValueType> or
     *           experimental::distributed::Vector<ValueType> with as many rows
     *           as this operator. If the Nullspace has an explicit basis, `v`
     *           must be of the same kind (and distribution) as that basis.
     */
    void project(ptr_param<LinOp> v) const;

    Nullspace& operator=(const Nullspace& other);

    Nullspace& operator=(Nullspace&& other);

    Nullspace(const Nullspace& other);

    Nullspace(Nullspace&& other);

protected:
    explicit Nullspace(std::shared_ptr<const Executor> exec);

    Nullspace(std::shared_ptr<const Executor> exec, dim<2> size,
              std::vector<std::shared_ptr<const LinOp>> basis,
              bool contains_constant);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    size_type get_num_basis_vectors() const noexcept
    {
        return basis_ ? basis_->get_size()[1] : 0;
    }

    template <typename VectorType>
    void setup_basis(const std::vector<std::shared_ptr<const LinOp>>& basis);

    template <typename VectorType>
    const matrix::Dense<ValueType>* get_local_basis(const VectorType* v) const;

    // stores [mean(v); V^H v] in coefficients_
    template <typename VectorType>
    void compute_components(const VectorType* v,
                            const matrix::Dense<ValueType>* basis_local,
                            bool has_constant) const;

    // v -= mean + V C with the coefficients from compute_components
    template <typename VectorType>
    void subtract_components(VectorType* v,
                             const matrix::Dense<ValueType>* basis_local,
                             bool has_constant) const;

    template <typename VectorType>
    void remove_components(VectorType* v,
                           const matrix::Dense<ValueType>* basis_local,
                           bool has_constant) const;

private:
    bool contains_constant_;
    // orthonormal explicit basis (n x k), Dense or distributed::Vector
    std::shared_ptr<const LinOp> basis_;
    // scratch: (1 + k) x nrhs coefficients (means, then V^H x)
    gko::detail::DenseCache<ValueType> coefficients_;
    gko::detail::DenseCache<ValueType> host_coefficients_;
    mutable array<char> reduction_tmp_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_NULLSPACE_HPP_
