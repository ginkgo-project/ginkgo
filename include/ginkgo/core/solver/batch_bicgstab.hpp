// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_


#include <vector>


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/batch_solver_base.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace batch {
namespace solver {


/**
 * BiCGSTAB or the Bi-Conjugate Gradient-Stabilized is a Krylov subspace solver.
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices.
 *
 * This solver solves a batch of linear systems using the Bicgstab algorithm.
 * Each linear system in the batch can converge independently.
 *
 * Unless otherwise specified via the `preconditioner` factory parameter, this
 * implementation does not use any preconditioner by default. The type of
 * tolerance (absolute or relative) and the maximum number of iterations to be
 * used in the stopping criterion can be set via the factory parameters.
 *
 * @note The tolerance check is against the internal residual computed within
 * the solver process. This implicit (internal) residual, can diverge from the
 * true residual (||b - Ax||). A posterori checks (by computing the true
 * residual, ||b - Ax||) are recommended to ensure that the solution has
 * converged to the desired tolerance.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class Bicgstab final
    : public EnableBatchSolver<Bicgstab<ValueType>, ValueType> {
    friend class EnableBatchLinOp<Bicgstab>;
    friend class EnablePolymorphicObject<Bicgstab, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {};
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(Bicgstab, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

private:
    explicit Bicgstab(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<Bicgstab, ValueType>(std::move(exec))
    {}

    explicit Bicgstab(const Factory* factory,
                      std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchSolver<Bicgstab, ValueType>(factory->get_executor(),
                                                 std::move(system_matrix),
                                                 factory->get_parameters()),
          parameters_{factory->get_parameters()}
    {}

    void solver_apply(
        const MultiVector<ValueType>* b, MultiVector<ValueType>* x,
        log::detail::log_data<real_type>* log_data) const override;
};


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_
