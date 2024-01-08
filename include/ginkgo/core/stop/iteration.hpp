// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_ITERATION_HPP_
#define GKO_PUBLIC_CORE_STOP_ITERATION_HPP_


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
        size_type GKO_FACTORY_PARAMETER_SCALAR(max_iters, 0);
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


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_ITERATION_HPP_
