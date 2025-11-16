// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/stop/iteration.hpp"

#include <ginkgo/core/base/abstract_factory.hpp>


namespace gko {
namespace stop {


bool Iteration::check_impl(uint8 stoppingId, bool setFinalized,
                           array<stopping_status>* stop_status,
                           bool* one_changed, const Updater& updater)
{
    bool result = updater.num_iterations_ >= parameters_.max_iters;
    if (result) {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }
    return result;
}


deferred_factory_parameter<Iteration::Factory> max_iters(size_type count)
{
    return Iteration::build().with_max_iters(count);
}


class MinIterationWrapper
    : public EnablePolymorphicObject<MinIterationWrapper, Criterion> {
    friend class EnablePolymorphicObject<MinIterationWrapper, Criterion>;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Minimum number of iterations, after which we check the inner
         * criterion
         */
        size_type min_iters{0};

        parameters_type& with_min_iters(size_type value)
        {
            this->min_iters = value;
            return *this;
        }

        std::shared_ptr<const CriterionFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            inner_criterion);
    };
    GKO_ENABLE_CRITERION_FACTORY(MinIterationWrapper, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(uint8 stopping_id, bool set_finalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Updater& updater) override
    {
        if (updater.num_iterations_ < this->get_parameters().min_iters) {
            return false;
        }
        return inner_criterion_->check(stopping_id, set_finalized, stop_status,
                                       one_changed, updater);
    }

    explicit MinIterationWrapper(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<MinIterationWrapper, Criterion>(
              std::move(exec))
    {}

    explicit MinIterationWrapper(const Factory* factory,
                                 const CriterionArgs& args)
        : EnablePolymorphicObject<MinIterationWrapper, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()},
          inner_criterion_{
              factory->get_parameters().inner_criterion->generate(args)}
    {}

    std::shared_ptr<Criterion> inner_criterion_;
};


deferred_factory_parameter<CriterionFactory> min_iters(
    size_type count, deferred_factory_parameter<CriterionFactory> criterion)
{
    return MinIterationWrapper::build()
        .with_min_iters(count)
        .with_inner_criterion(criterion);
}


}  // namespace stop
}  // namespace gko
