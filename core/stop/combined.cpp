// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/combined.hpp>


namespace gko {
namespace stop {


Combined::Combined(std::shared_ptr<const gko::Executor> exec)
    : EnablePolymorphicObject<Combined, Criterion>(std::move(exec))
{}


Combined::Combined(const Combined::Factory* factory, const CriterionArgs& args)
    : EnablePolymorphicObject<Combined, Criterion>(factory->get_executor()),
      parameters_{factory->get_parameters()}
{
    for (const auto& f : parameters_.criteria) {
        // Ignore the nullptr from the list
        if (f != nullptr) {
            criteria_.push_back(f->generate(args));
        }
    }
    // If the list are empty or all nullptr, throw gko::NotSupported
    if (criteria_.size() == 0) {
        GKO_NOT_SUPPORTED(this);
    }
}


bool Combined::check_impl(uint8 stoppingId, bool setFinalized,
                          array<stopping_status>* stop_status,
                          bool* one_changed, const Updater& updater)
{
    bool one_converged = false;
    gko::uint8 ids{1};
    *one_changed = false;
    for (auto& c : criteria_) {
        bool local_one_changed = false;
        one_converged |= c->check(ids, setFinalized, stop_status,
                                  &local_one_changed, updater);
        *one_changed |= local_one_changed;
        if (one_converged) {
            break;
        }
        ids++;
    }
    return one_converged;
}


Combined::Factory::Factory(std::shared_ptr<const ::gko::Executor> exec)
    : Base(std::move(exec))
{}


Combined::Factory::Factory(std::shared_ptr<const ::gko::Executor> exec,
                           const Combined::parameters_type& parameters)
    : Base(std::move(exec), parameters)
{}


Combined::Factory& Combined::Factory::operator=(const Combined::Factory& other)
{
    if (this != &other) {
        parameters_type new_parameters;
        new_parameters.criteria.clear();
        for (auto criterion : other.get_parameters().criteria) {
            new_parameters.criteria.push_back(
                gko::clone(this->get_executor(), criterion));
        }
        Base::operator=(Factory(this->get_executor(), new_parameters));
    }
    return *this;
}


}  // namespace stop
}  // namespace gko
