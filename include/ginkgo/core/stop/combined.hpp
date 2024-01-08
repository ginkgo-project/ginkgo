// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_COMBINED_HPP_
#define GKO_PUBLIC_CORE_STOP_COMBINED_HPP_


#include <vector>


#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {


/**
 * The Combined class is used to combine multiple criterions together through an
 * OR operation. The typical use case is to stop the iteration process if any of
 * the criteria is fulfilled, e.g. a number of iterations, the relative residual
 * norm has reached a threshold, etc.
 *
 * @ingroup stop
 */
class Combined : public EnablePolymorphicObject<Combined, Criterion> {
    friend class EnablePolymorphicObject<Combined, Criterion>;

public:
    class Factory;

    struct parameters_type
        : public ::gko::enable_parameters_type<parameters_type, Factory> {
        /**
         * Criterion factories to combine
         *
         * @internal In order to simplify the factory, the factories are passed
         * as `shared_ptr`. This way there is no problem when copying the whole
         * vector. Technically it is a vector (similar to a traditional array)
         * of pointers, so copying it when creating the factories should not be
         * too costly.
         */
        std::vector<std::shared_ptr<const CriterionFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(criteria);
    };

    class Factory
        : public ::gko::stop::EnableDefaultCriterionFactory<Factory, Combined,
                                                            parameters_type> {
        friend class ::gko::EnablePolymorphicObject<
            Factory, ::gko::stop::CriterionFactory>;
        friend class ::gko::enable_parameters_type<parameters_type, Factory>;

        using Base =
            ::gko::stop::EnableDefaultCriterionFactory<Factory, Combined,
                                                       parameters_type>;

    public:
        explicit Factory(std::shared_ptr<const ::gko::Executor> exec);
        explicit Factory(std::shared_ptr<const ::gko::Executor> exec,
                         const parameters_type& parameters);

        Factory(const Factory& other) = default;
        Factory(Factory&& other) = default;

        Factory& operator=(const Factory& other);
    };

    static parameters_type build() { return {}; }

    const parameters_type& get_parameters() const { return parameters_; }

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Updater&) override;

    explicit Combined(std::shared_ptr<const gko::Executor> exec);

    explicit Combined(const Factory* factory, const CriterionArgs& args);

private:
    friend ::gko::stop::EnableDefaultCriterionFactory<Factory, Combined,
                                                      parameters_type>;

    parameters_type parameters_;

    std::vector<std::unique_ptr<Criterion>> criteria_{};
};


/**
 * Combines multiple criterion factories into a single combined criterion
 * factory.
 *
 * This function treats a singleton container as a special case and avoids
 * creating an additional object and just returns the input factory.
 *
 * @tparam FactoryContainer  a random access container type
 *
 * @param factories  a list of factories to combined
 *
 * @return a combined criterion factory if the input contains multiple factories
 *         or the input factory if the input contains only one factory
 *
 * @ingroup stop
 */
template <typename FactoryContainer>
std::shared_ptr<const CriterionFactory> combine(FactoryContainer&& factories)
{
    switch (factories.size()) {
    case 0:
        GKO_NOT_SUPPORTED(nullptr);
    case 1:
        if (factories[0] == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        return factories[0];
    default:
        if (factories[0] == nullptr) {
            // first factory must be valid to capture executor
            GKO_NOT_SUPPORTED(nullptr);
        } else {
            auto exec = factories[0]->get_executor();
            return Combined::build()
                .with_criteria(std::forward<FactoryContainer>(factories))
                .on(exec);
        }
    }
}


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_COMBINED_HPP_
