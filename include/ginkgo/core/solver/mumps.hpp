// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_MUMPS_HPP_
#define GKO_PUBLIC_CORE_SOLVER_MUMPS_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_MUMPS


#include <memory>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace solver {


/**
 * Provides a wrapper around the MUMPS sparse direct solver.
 *
 * MUMPS (MUltifrontal Massively Parallel Sparse direct Solver) performs
 * LU or LDL^T factorization and triangular solves. This wrapper handles
 * the analysis, factorization, and solve phases internally.
 *
 * The system matrix is provided during generation (via the factory), and
 * the solve is performed in the apply step. MUMPS manages its own internal
 * factorization data structures, so no Ginkgo Factorization object is
 * produced.
 *
 * @note This solver only runs on the host (CPU). If the executor is not
 *       a ReferenceExecutor or OmpExecutor, data will be copied to the
 *       host for the MUMPS calls.
 *
 * @tparam ValueType  the type used to store values of the system matrix
 * @tparam IndexType  the type used to store sparsity pattern indices of the
 *                    system matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Mumps : public EnableLinOp<Mumps<ValueType, IndexType>> {
    friend class EnableLinOp<Mumps>;
    friend class EnablePolymorphicObject<Mumps, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    class Factory;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Factory> {
        /**
         * If set to true, MUMPS assumes the matrix is symmetric and
         * uses the symmetric (LDL^T) factorization. Otherwise, uses
         * the general (LU) factorization.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric, false);

        /**
         * MUMPS ICNTL(7) ordering option for the analysis phase.
         * Common values:
         *   0 = AMD (default, built-in to MUMPS)
         *   2 = AMF (Approximate Minimum Fill)
         *   3 = SCOTCH
         *   4 = PORD
         *   5 = METIS
         *   7 = automatic choice
         */
        int GKO_FACTORY_PARAMETER_SCALAR(ordering, 0);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Mumps, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /** Creates a copy of the solver (re-factorizes from stored matrix). */
    Mumps(const Mumps&);

    /** Moves from the given solver, leaving it empty. */
    Mumps(Mumps&&) noexcept;

    Mumps& operator=(const Mumps&);

    Mumps& operator=(Mumps&&) noexcept;

    ~Mumps() override;

protected:
    explicit Mumps(std::shared_ptr<const Executor> exec);

    Mumps(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    /**
     * Opaque handle to the MUMPS internal state. This avoids exposing
     * the MUMPS headers in the public API.
     */
    struct mumps_state;
    std::unique_ptr<mumps_state> state_;
    std::shared_ptr<const matrix_type> system_matrix_;

    /** Persistent host buffer for the MUMPS solve phase. */
    detail::DenseCache<ValueType> host_buffer_;
};


}  // namespace solver
}  // namespace experimental
}  // namespace gko


#endif  // GKO_HAVE_MUMPS


#endif  // GKO_PUBLIC_CORE_SOLVER_MUMPS_HPP_
