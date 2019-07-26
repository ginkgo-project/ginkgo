/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_PRECONDITIONER_ILU_HPP_
#define GKO_CORE_PRECONDITIONER_ILU_HPP_

#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


namespace gko {
namespace preconditioner {


/**
 * Incomplete LU (ILU) is ... TODO
 *
 * @note This class is not thread safe (even a const object is not) because it
 *       uses an internal cache to accelerate multiple applies
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename LSolverType, typename USolverType,
          typename ValueType = default_precision, bool ReverseApply = false>
class AbstractIlu
    : public EnableLinOp<
          AbstractIlu<LSolverType, USolverType, ValueType, ReverseApply>> {
    friend class EnableLinOp<AbstractIlu>;
    friend class EnablePolymorphicObject<AbstractIlu, LinOp>;

public:
    using value_type = ValueType;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factory for the L solver
         */
        std::shared_ptr<typename LSolverType::Factory> GKO_FACTORY_PARAMETER(
            l_solver_factory, nullptr);

        /**
         * Factory for the U solver
         */
        std::shared_ptr<typename USolverType::Factory> GKO_FACTORY_PARAMETER(
            u_solver_factory, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(AbstractIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    class LUManager {
    public:
        LUManager(std::shared_ptr<const LinOp> single)
        {
            // TODO: dynamic_cast to Composition, followed by setting l_factor_
            // and u_factor_ Also: Throw error on failure / incompatability
        }
        LUManager(std::shared_ptr<const LinOp> l_factor,
                  std::shared_ptr<const LinOp> u_factor)
            : l_factor_{std::move(l_factor)}, u_factor_{std::move(u_factor)}
        {}
        // TODO: add get_size
    private:
        std::shared_ptr<const LinOp> l_factor_;
        std::shared_ptr<const LinOp> u_factor_;
    };

public:
    // The Factory might have to be hand written since we want to have multiple
    // `generate` functions (e.g. with ParIlu and user chosen L and U matrices)
    /*
    class Factory
        : public EnablePolymorphicObject<Factory, >, {
    friend class ::gko::enable_parameters_type<parameters_type, Factory>;
    public:
    };
    GKO_ENABLE_BUILD_METHOD(Factory);

// Required since we did not use the `GKO_ENABLE_LIN_OP_FACTORY` macro
private:
    parameters_type parameters_;
    */

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        ensure_cache_support_for(b);
        if (!ReverseApply) {
            l_solver_->apply(b, cache_.intermediate.get());
            u_solver_->apply(cache_.intermediate.get(), x);
        } else {
            u_solver_->apply(b, cache_.intermediate.get());
            l_solver_->apply(cache_.intermediate.get(), x);
        }
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {
        ensure_cache_support_for(b);
        if (!ReverseApply) {
            l_solver_->apply(b, cache_.intermediate.get());
            u_solver_->apply(alpha, cache_.intermediate.get(), beta, x);
        } else {
            // TODO: Might be wrong order (for alpha and beta), needs a sanity
            // check!
            u_solver_->apply(b, cache_.intermediate.get());
            l_solver_->apply(alpha, cache_.intermediate.get(), beta, x);
        }
    }

    explicit AbstractIlu(std::shared_ptr<const Executor> exec)
        : EnableLinOp<AbstractIlu>(std::move(exec))
    {}

    explicit AbstractIlu(const Factory *factory,
                         std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<AbstractIlu>(factory->get_executor(),
                                   // TODO: read dimensions from struct
                                   transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          factors_{std::dynamic_pointer_cast<const Composition<ValueType>>(
              system_matrix)}
    {
        // TODO: For reverse apply, L and U must have the same dimensions!
        // Throw if it is not a Composition object
        if (!factors_ || factors_->get_operators().size() < 2) {
            throw GKO_NOT_SUPPORTED(*(system_matrix.get()));
        }
        auto exec = this->get_executor();
        // If it was not set, use a default generated one of `LSolverType`
        if (!parameters_.l_solver_factory) {
            l_solver_ =
                LSolverType::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(40u).on(
                            exec),
                        gko::stop::ResidualNormReduction<>::build()
                            .with_reduction_factor(1e-4)
                            .on(exec))
                    .on(exec)
                    ->generate(factors_->get_operators()[0]);
        } else {
            l_solver_ = parameters_.l_solver_factory->generate(
                factors_->get_operators()[0]);
        }
        if (!parameters_.u_solver_factory) {
            u_solver_ =
                USolverType::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(40u).on(
                            exec),
                        gko::stop::ResidualNormReduction<>::build()
                            .with_reduction_factor(1e-4)
                            .on(exec))
                    .on(exec)
                    ->generate(factors_->get_operators()[1]);
        } else {
            u_solver_ = parameters_.u_solver_factory->generate(
                factors_->get_operators()[1]);
        }
    }

    void ensure_cache_support_for(const LinOp *b) const
    {
        dim<2> expected_size =
            ReverseApply ? dim<2>{u_solver_->get_size()[0], b->get_size()[1]}
                         : dim<2>{l_solver_->get_size()[0], b->get_size()[1]};
        if (cache_.intermediate == nullptr ||
            cache_.intermediate->get_size() != expected_size) {
            cache_.intermediate = matrix::Dense<ValueType>::create(
                this->get_executor(), expected_size);
        }
    }

private:
    // Temporary solution, later it should be replaced with a wrapper object
    // (PolymorphicBase::components_type)
    std::shared_ptr<const Composition<ValueType>> factors_{};
    /**
     * Manages a vector as a cache, so there is no need to allocate one every
     * time an intermediate vector is required.
     * Copying an instance will only yield an empty object since copying the
     * cached vector would not make sense.
     *
     * @internal  The struct is necessary, so the whole class can be copiable
     *            (could also be done with writing `operator=` and copy
     *             constructor by hand)
     */
    mutable struct cache_struct {
        cache_struct() = default;
        cache_struct(const cache_struct &other) {}
        cache_struct &operator=(const cache_struct &) { return *this; }
        std::unique_ptr<LinOp> intermediate{};
    } cache_;
    std::shared_ptr<const LSolverType> l_solver_{};
    std::shared_ptr<const USolverType> u_solver_{};
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ILU_HPP_
