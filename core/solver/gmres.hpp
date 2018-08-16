/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_SOLVER_GMRES_HPP_
#define GKO_CORE_SOLVER_GMRES_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/log/logger.hpp"
#include "core/matrix/identity.hpp"
#include "core/stop/criterion.hpp"


namespace gko {
namespace solver {


constexpr size_type default_max_iter_num = 100u;


/**
 * GMRES or the generalized minimal residual method is an iterative type Krylov
 * subspace method which is suitable for nonsymmetric linear systems.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of GMRES are
 * merged into 2 separate steps.
 *
 * @tparam ValueType  precision of matrix elements
 */
template <typename ValueType = default_precision>
class Gmres : public EnableLinOp<Gmres<ValueType>>,
              public log::EnableLogging<Gmres<ValueType>> {
    friend class EnableLinOp<Gmres>;
    friend class EnablePolymorphicObject<Gmres, LinOp>;

public:
    using log::EnableLogging<Gmres<ValueType>>::log;
    using log::EnableLogging<Gmres<ValueType>>::add_logger;
    using value_type = ValueType;

    /**
     * Gets the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the preconditioner operator used by the solver.
     *
     * @return the preconditioner operator used by the solver
     */
    std::shared_ptr<const LinOp> get_preconditioner() const
    {
        return preconditioner_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factory
         */
        std::shared_ptr<const stop::CriterionFactory> GKO_FACTORY_PARAMETER(
            criterion, nullptr);

        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(
            preconditioner, nullptr);

        /**
         * krylov dimension factory.
         */
        size_type GKO_FACTORY_PARAMETER(krylov_dim, 0u);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Gmres, parameters, Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    explicit Gmres(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Gmres>(std::move(exec))
    {}

    explicit Gmres(const Factory *factory,
                   std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Gmres>(factory->get_executor(),
                             transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        if (parameters_.preconditioner) {
            preconditioner_ =
                parameters_.preconditioner->generate(system_matrix_);
        } else {
            preconditioner_ = matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]);
        }
        if (parameters_.criterion) {
            stop_criterion_factory_ = std::move(parameters_.criterion);
        } else {
            NOT_SUPPORTED(nullptr);
        }
        if (parameters_.krylov_dim) {
            krylov_dim_ = parameters_.krylov_dim;
        } else {
            krylov_dim_ = default_max_iter_num;
        }
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const LinOp> preconditioner_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
    size_type krylov_dim_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_HPP
