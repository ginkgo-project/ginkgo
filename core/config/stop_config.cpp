/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/base/exception_helpers.hpp>
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
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    auto factory = stop::Time::build();
    SET_VALUE(factory, long long int, time_limit, config);
    return factory.on(exec);
}


inline std::unique_ptr<stop::CriterionFactory> configure_iter(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    auto factory = stop::Iteration::build();
    SET_VALUE(factory, size_type, max_iters, config);
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
    GKO_INVALID_STATE("Not valid " + str);
}


template <typename ValueType>
class ResidualNormConfigurer {
public:
    static std::unique_ptr<typename stop::ResidualNorm<ValueType>::Factory>
    build_from_config(const gko::config::pnode& config,
                      const gko::config::registry& context,
                      std::shared_ptr<const Executor> exec,
                      gko::config::type_descriptor td_for_child)
    {
        auto factory = stop::ResidualNorm<ValueType>::build();
        SET_VALUE(factory, remove_complex<ValueType>, reduction_factor, config);
        if (config.contains("baseline")) {
            factory.with_baseline(
                get_mode(config.at("baseline").get_data<std::string>()));
        }
        return factory.on(exec);
    }
};


inline std::unique_ptr<stop::CriterionFactory> configure_residual(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<stop::CriterionFactory, ResidualNormConfigurer>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType>
class ImplicitResidualNormConfigurer {
public:
    static std::unique_ptr<
        typename stop::ImplicitResidualNorm<ValueType>::Factory>
    build_from_config(const gko::config::pnode& config,
                      const gko::config::registry& context,
                      std::shared_ptr<const Executor> exec,
                      gko::config::type_descriptor td_for_child)
    {
        auto factory = stop::ImplicitResidualNorm<ValueType>::build();
        SET_VALUE(factory, remove_complex<ValueType>, reduction_factor, config);
        if (config.contains("baseline")) {
            factory.with_baseline(
                get_mode(config.at("baseline").get_data<std::string>()));
        }
        return factory.on(exec);
    }
};


inline std::unique_ptr<stop::CriterionFactory> configure_implicit_residual(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<stop::CriterionFactory, ImplicitResidualNormConfigurer>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <>
std::shared_ptr<const stop::CriterionFactory>
get_pointer<const stop::CriterionFactory>(const pnode& config,
                                          const registry& context,
                                          std::shared_ptr<const Executor> exec,
                                          type_descriptor td)
{
    std::shared_ptr<const stop::CriterionFactory> ptr;
    if (config.is(pnode::status_t::object)) {
        return context.search_data<stop::CriterionFactory>(
            config.get_data<std::string>());
    } else if (config.is(pnode::status_t::list)) {
        static std::map<
            std::string,
            std::function<std::unique_ptr<gko::stop::CriterionFactory>(
                const pnode&, const registry&, std::shared_ptr<const Executor>&,
                type_descriptor)>>
            criterion_map{
                {{"Time", configure_time},
                 {"Iteration", configure_iter},
                 {"ResidualNorm", configure_residual},
                 {"ImplicitResidualNorm", configure_implicit_residual}}};
        return criterion_map.at(config.at("Type").get_data<std::string>())(
            config, context, exec, td);
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::vector<std::shared_ptr<const stop::CriterionFactory>>
get_pointer_vector<const stop::CriterionFactory>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    std::vector<std::shared_ptr<const stop::CriterionFactory>> res;
    if (config.is(pnode::status_t::array)) {
        for (const auto& it : config.get_array()) {
            res.push_back(get_pointer<const stop::CriterionFactory>(it, context,
                                                                    exec, td));
        }
    } else {
        res.push_back(get_pointer<const stop::CriterionFactory>(config, context,
                                                                exec, td));
    }
    // TODO: handle shortcut version by config.is(pnode::status_t::list) &&
    // !config.contains("Type")

    return res;
}


}  // namespace config
}  // namespace gko
