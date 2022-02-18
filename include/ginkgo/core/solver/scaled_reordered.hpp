/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SOLVER_SCALED_REORDERED_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SCALED_REORDERED_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/reorder/mc64.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>


namespace gko {
namespace solver {


template <typename ValueType = default_precision>
class ScaledReordered : public EnableLinOp<ScaledReordered<ValueType>> {
    friend class EnableLinOp<ScaledReordered>;
    friend class EnablePolymorphicObject<ScaledReordered, LinOp>;

public:
    using value_type = ValueType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Returns the solver operator used as the inner solver.
     *
     * @return the solver operator used as the inner solver
     */
    std::shared_ptr<const LinOp> get_solver() const { return solver_; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Inner solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            solver, nullptr);

        std::shared_ptr<const reorder::ReorderingBase>
            GKO_FACTORY_PARAMETER_SCALAR(reordering, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(ScaledReordered, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const
    {
        auto mc64 = dynamic_cast<const reorder::Mc64<double, int>*>(
            parameters_.reordering.get());
        auto pb = as<matrix::Dense<double>>(b)->row_permute(
            mc64->get_permutation().get());
        mc64->get_row_scaling()->apply(pb.get(), pb.get());
        auto px = as<matrix::Dense<double>>(
            as<matrix::Dense<double>>(x)->inverse_row_permute(
                mc64->get_inverse_permutation().get()));

        solver_->apply(pb.get(), px.get());

        px = as<matrix::Dense<double>>(
            px->row_permute(mc64->get_inverse_permutation().get()));
        mc64->get_col_scaling()->apply(px.get(), x);
    }

    void apply_dense_impl(const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x) const
        GKO_NOT_IMPLEMENTED;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const GKO_NOT_IMPLEMENTED;

    explicit ScaledReordered(std::shared_ptr<const Executor> exec)
        : EnableLinOp<ScaledReordered>(std::move(exec))
    {}

    explicit ScaledReordered(const Factory* factory,
                             std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<ScaledReordered>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
        if (parameters_.reordering) {
            auto mc64 = dynamic_cast<const reorder::Mc64<double, int>*>(
                parameters_.reordering.get());
            if (mc64) {
                auto PA = as<matrix::Csr<double, int>>(
                    as<matrix::Csr<double, int>>(system_matrix)
                        ->row_permute(mc64->get_permutation().get()));
                mc64->get_row_scaling()->apply(PA.get(), PA.get());
                mc64->get_col_scaling()->rapply(PA.get(), PA.get());
                PA = as<matrix::Csr<double, int>>(PA->inverse_column_permute(
                    mc64->get_inverse_permutation().get()));
                system_matrix_ = gko::share(PA);
            }
        }
        if (parameters_.solver) {
            solver_ = parameters_.solver->generate(system_matrix_);
        } else {
            solver_ = matrix::Identity<ValueType>::create(this->get_executor(),
                                                          this->get_size());
        }
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const LinOp> solver_{};

    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
    std::shared_ptr<const matrix::Dense<ValueType>> relaxation_factor_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_IR_HPP_
