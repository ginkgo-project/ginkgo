#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>
#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"

namespace gko {
namespace config {


inline std::unique_ptr<stop::CriterionFactory> configure_time(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    auto factory = stop::Time::build();
    SET_VALUE(factory, long long int, time_limit, item);
    return factory.on(exec);
}


inline std::unique_ptr<stop::CriterionFactory> configure_iter(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    auto factory = stop::Iteration::build();
    SET_VALUE(factory, size_type, max_iters, item);
    return factory.on(exec);
}


stop::mode get_mode(const std::string& str)
{
    if (str == "absolute") {
        return stop::mode::absolute;
    } else if (str == "initial_resnorm") {
        return stop::mode::initial_resnorm;
    } else if (str == "rhs_norm") {
        return stop::mode::rhs_norm;
    }
}


template <typename ValueType>
class ResidualNormConfigurer {
public:
    static std::unique_ptr<typename stop::ResidualNorm<ValueType>::Factory>
    build_from_config(const gko::config::Config& config,
                      const gko::config::registry& context,
                      std::shared_ptr<const Executor> exec,
                      gko::config::TypeDescriptor td_for_child)
    {
        auto factory = stop::ResidualNorm<ValueType>::build();
        // SET_VALUE(factory, remove_complex<ValueType>, reduction_factor,
        // item);
        if (config.contains("baseline")) {
            factory.with_baseline(
                get_mode(config.at("baseline").get_data<std::string>()));
        }
        return factory.on(exec);
    }
};


inline std::unique_ptr<stop::CriterionFactory> configure_residual(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    auto updated = update_type(item, td);
    return dispatch<stop::CriterionFactory, ResidualNormConfigurer>(
        updated.first, item, context, exec, updated, value_type_list());
}


template <typename ValueType>
class ImplicitResidualNormConfigurer {
public:
    static std::unique_ptr<
        typename stop::ImplicitResidualNorm<ValueType>::Factory>
    build_from_config(const gko::config::Config& config,
                      const gko::config::registry& context,
                      std::shared_ptr<const Executor> exec,
                      gko::config::TypeDescriptor td_for_child)
    {
        auto factory = stop::ImplicitResidualNorm<ValueType>::build();
        // SET_VALUE(factory, remove_complex<ValueType>, reduction_factor,
        // item);
        if (config.contains("baseline")) {
            factory.with_baseline(
                get_mode(config.at("baseline").get_data<std::string>()));
        }
        return factory.on(exec);
    }
};


inline std::unique_ptr<stop::CriterionFactory> configure_implicit_residual(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    auto updated = update_type(item, td);
    return dispatch<stop::CriterionFactory, ImplicitResidualNormConfigurer>(
        updated.first, item, context, exec, updated, value_type_list());
}


template <>
std::shared_ptr<const stop::CriterionFactory>
get_pointer<const stop::CriterionFactory>(const item& item,
                                          const registry& context,
                                          std::shared_ptr<const Executor> exec,
                                          TypeDescriptor td)
{
    std::shared_ptr<const stop::CriterionFactory> ptr;
    if (item.is(pnode::status_t::object)) {
        return context.search_data<stop::CriterionFactory>(
            item.get_data<std::string>());
    } else if (item.is(pnode::status_t::list)) {
        static std::map<
            std::string,
            std::function<std::unique_ptr<gko::stop::CriterionFactory>(
                const Config&, const registry&,
                std::shared_ptr<const Executor>&, TypeDescriptor)>>
            criterion_map{
                {{"Time", configure_time},
                 {"Iteration", configure_iter},
                 {"ResidualNorm", configure_residual},
                 {"ImplicitResidualNorm", configure_implicit_residual}}};
        return criterion_map.at(item.at("Type").get_data<std::string>())(
            item, context, exec, td);
    }
    // handle object is item
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::vector<std::shared_ptr<const stop::CriterionFactory>>
get_pointer_vector<const stop::CriterionFactory>(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    std::vector<std::shared_ptr<const stop::CriterionFactory>> res;
    if (item.is(pnode::status_t::array)) {
        for (const auto& it : item.get_array()) {
            res.push_back(get_pointer<const stop::CriterionFactory>(it, context,
                                                                    exec, td));
        }
    } else {
        res.push_back(
            get_pointer<const stop::CriterionFactory>(item, context, exec, td));
    }
    // TODO: handle shortcut version by item.is(pnode::status_t::list) &&
    // !item.contains("Type")

    return res;
}


}  // namespace config
}  // namespace gko