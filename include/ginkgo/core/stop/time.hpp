// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_TIME_HPP_
#define GKO_PUBLIC_CORE_STOP_TIME_HPP_


#include <chrono>


#include <ginkgo/core/stop/criterion.hpp>


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


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_TIME_HPP_
