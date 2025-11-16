// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_ITERATION_HPP_
#define GKO_PUBLIC_CORE_STOP_ITERATION_HPP_


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {

/**
 * The Iteration class is a stopping criterion which stops the iteration process
 * after a preset number of iterations.
 *
 * @note to use this stopping criterion, it is required to update the iteration
 * count for the ::check() method.
 *
 * @ingroup stop
 */
class Iteration : public EnablePolymorphicObject<Iteration, Criterion> {
    friend class EnablePolymorphicObject<Iteration, Criterion>;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Maximum number of iterations
         */
        size_type max_iters{0};

        parameters_type& with_max_iters(size_type value)
        {
            this->max_iters = value;
            return *this;
        }
    };
    GKO_ENABLE_CRITERION_FACTORY(Iteration, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Updater& updater) override;

    explicit Iteration(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Iteration, Criterion>(std::move(exec))
    {}

    explicit Iteration(const Factory* factory, const CriterionArgs& args)
        : EnablePolymorphicObject<Iteration, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * Creates the precursor to an Iteration stopping criterion factory, to be used
 * in conjunction with `.with_criteria(...)` function calls when building a
 * solver factory. This stopping criterion will stop the iteration after `count`
 * iterations of the solver have finished.
 *
 * Full usage example: Stop after 100 iterations or when the relative residual
 * norm is below $10^{-10}$, whichever happens first.
 * ```cpp
 * auto factory = gko::solver::Cg<double>::build()
 *                    .with_criteria(
 *                        gko::stop::max_iters(100),
 *                        gko::stop::relative_residual_norm(1e-10))
 *                    .on(exec);
 * ```
 *
 * @param count  the number of iterations after which to stop
 * @return a deferred_factory_parameter that can be passed to the
 *         `with_criteria` function when building a solver.
 */
deferred_factory_parameter<Iteration::Factory> max_iters(size_type count);


deferred_factory_parameter<CriterionFactory> min_iters(
    size_type count, deferred_factory_parameter<CriterionFactory> criterion);


template <typename... Args>
std::enable_if_t<sizeof...(Args) >= 2,
                 deferred_factory_parameter<CriterionFactory>>
min_iters(size_type count, Args&&... criteria)
{
    std::vector<deferred_factory_parameter<CriterionFactory>> criterion_vec{
        std::forward<Args>(criteria)...};
    return min_iters(count, Combined::build().with_criteria(criterion_vec));
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_ITERATION_HPP_
