/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_types.hpp>
#include <ginkgo/core/solver/batch_solver.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace solver {


/**
 * BiCGSTAB or the Bi-Conjugate Gradient-Stabilized is a Krylov subspace solver.
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices.
 *
 * This solver solves a batch of linear systems using Bicgstab algorithm.
 *
 * Unless otherwise specified via the `preconditioner` factory parameter, this
 * implementation does not use any preconditioner by default. The type of
 * tolerance( absolute or relative ) and the maximum number of iterations to be
 * used in the stopping criterion can be set via the factory parameters.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchBicgstab : public EnableBatchSolver<BatchBicgstab<ValueType>>,
                      public BatchTransposable {
    friend class EnableBatchLinOp<BatchBicgstab>;
    friend class EnablePolymorphicObject<BatchBicgstab, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchBicgstab<ValueType>;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Inner preconditioner descriptor.
         */
        preconditioner::batch::type GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, preconditioner::batch::type::none);

        /**
         * Maximum number iterations allowed.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100);


        /**
         * Residual tolerance.
         */
        real_type GKO_FACTORY_PARAMETER_SCALAR(residual_tol, 1e-11);


        /**
         * To specify which tolerance is to be considered.
         */
        ::gko::stop::batch::ToleranceType GKO_FACTORY_PARAMETER_SCALAR(
            tolerance_type, ::gko::stop::batch::ToleranceType::absolute);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchBicgstab, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void solver_apply(const BatchLinOp* const mtx, const BatchLinOp* const b,
                      BatchLinOp* const x, BatchInfo& info) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

    explicit BatchBicgstab(std::shared_ptr<const Executor> exec)
        : EnableBatchSolver<BatchBicgstab>(std::move(exec))
    {}

    explicit BatchBicgstab(const Factory* factory,
                           std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchSolver<BatchBicgstab>(factory->get_executor(),
                                           std::move(system_matrix)),
          parameters_{factory->get_parameters()}
    {}
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_BICGSTAB_HPP_
