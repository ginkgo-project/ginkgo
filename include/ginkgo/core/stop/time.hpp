// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_TIME_HPP_
#define GKO_PUBLIC_CORE_STOP_TIME_HPP_


#include <chrono>

#include <ginkgo/core/stop/criterion.hpp>

#include "ginkgo/core/base/abstract_factory.hpp"


namespace gko {
namespace stop {

/**
 * The Time class is a stopping criterion which stops the iteration process
 * after a certain amount of time has passed.
 *
 * @ingroup stop
 */
class Time : public EnablePolymorphicObject<Time, Criterion> {
    friend class EnablePolymorphicObject<Time, Criterion>;

public:
    using clock = std::chrono::steady_clock;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Amount of seconds to wait (default value: 10 seconds)
         */
        std::chrono::nanoseconds GKO_FACTORY_PARAMETER_SCALAR(time_limit,
                                                              10000000000LL);
    };
    GKO_ENABLE_CRITERION_FACTORY(Time, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Updater&) override;

    explicit Time(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Time, Criterion>(std::move(exec)),
          time_limit_{},
          start_{}
    {}

    explicit Time(const Factory* factory, const CriterionArgs args)
        : EnablePolymorphicObject<Time, Criterion>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          time_limit_{std::chrono::duration<double>(
              factory->get_parameters().time_limit)},
          start_{clock::now()}
    {}

private:
    /**
     * @internal in order to improve the interface, we store a `double` in the
     * parameters and here properly convert the double to a
     * std::chrono::duration type
     */
    std::chrono::duration<double> time_limit_;
    clock::time_point start_;
};


/**
 * Creates the precursor to a Time stopping criterion factory, to be used
 * in conjunction with `.with_criteria(...)` function calls when building a
 * solver factory. This stopping criterion will stop the iteration after the
 * specified amount of time since the start of the solver run has elapsed.
 *
 * Full usage example: Stop after 1 second or when the relative residual norm is
 * below $10^{-10}$, whichever happens first.
 * ```cpp
 * auto factory = gko::solver::Cg<double>::build()
 *                    .with_criteria(
 *                        gko::stop::time_limit(std::chrono::seconds(1)),
 *                        gko::stop::relative_residual_norm(1e-10))
 *                    .on(exec);
 * ```
 *
 * @param duration  the time limit after which to stop the iteration.
 *                  Thanks to std::chrono's converting constructors, you can
 *                  specify any time units, and they will be converted to
 *                  nanoseconds automatically.
 * @return a deferred_factory_parameter that can be passed to the
 *         `with_criteria` function when building a solver.
 */
deferred_factory_parameter<Time::Factory> time_limit(
    std::chrono::nanoseconds duration);


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_TIME_HPP_
