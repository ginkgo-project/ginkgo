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

#ifndef GKO_CORE_CONFIG_CONFIG_HPP_
#define GKO_CORE_CONFIG_CONFIG_HPP_


#include <ginkgo/core/config/config.hpp>


#include <string>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {

/**
 * This function is to update the default type setting from current config.
 *
 * @note It might update the unused type for the current class.
 */
inline type_descriptor update_type(const pnode& config,
                                   const type_descriptor& td)
{
    type_descriptor updated = td;

    if (config.contains("ValueType")) {
        updated.first = config.at("ValueType").get_data<std::string>();
    }
    if (config.contains("IndexType")) {
        updated.second = config.at("IndexType").get_data<std::string>();
    }
    return updated;
}


template <typename T>
inline std::shared_ptr<T> get_pointer(const pnode& config,
                                      const registry& context,
                                      std::shared_ptr<const Executor> exec,
                                      type_descriptor td)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    ptr = context.search_data<T_non_const>(config.get_data<std::string>());
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
inline std::shared_ptr<const LinOpFactory> get_pointer<const LinOpFactory>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    std::shared_ptr<const LinOpFactory> ptr;
    if (config.is(pnode::status_t::object)) {
        ptr = context.search_data<LinOpFactory>(config.get_data<std::string>());
    } else if (config.is(pnode::status_t::list)) {
        ptr = build_from_config(config, context, exec, td);
    }
    // handle object is config
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::shared_ptr<const stop::CriterionFactory>
get_pointer<const stop::CriterionFactory>(const pnode& config,
                                          const registry& context,
                                          std::shared_ptr<const Executor> exec,
                                          type_descriptor td);


template <typename T>
inline std::vector<std::shared_ptr<T>> get_pointer_vector(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    std::vector<std::shared_ptr<T>> res;
    // for loop in config
    if (config.is(pnode::status_t::array)) {
        for (const auto& it : config.get_array()) {
            res.push_back(get_pointer<T>(it, context, exec, td));
        }
    } else {
        // only one config can be passed without array
        res.push_back(get_pointer<T>(config, context, exec, td));
    }

    return res;
}

template <>
std::vector<std::shared_ptr<const stop::CriterionFactory>>
get_pointer_vector<const stop::CriterionFactory>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td);


template <typename IndexType>
inline
    typename std::enable_if<std::is_integral<IndexType>::value, IndexType>::type
    get_value(const pnode& config)
{
    auto val = config.get_data<long long int>();
    assert(val <= std::numeric_limits<IndexType>::max() &&
           val >= std::numeric_limits<IndexType>::min());
    return static_cast<IndexType>(val);
}

template <typename ValueType>
inline typename std::enable_if<std::is_floating_point<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config)
{
    auto val = config.get_data<double>();
    assert(val <= std::numeric_limits<ValueType>::max() &&
           val >= -std::numeric_limits<ValueType>::max());
    return static_cast<ValueType>(val);
}

template <typename ValueType>
inline typename std::enable_if<gko::is_complex_s<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config)
{
    using real_type = gko::remove_complex<ValueType>;
    if (config.is(pnode::status_t::object)) {
        return static_cast<ValueType>(get_value<real_type>(config));
    } else if (config.is(pnode::status_t::array)) {
        return ValueType{get_value<real_type>(config.at(0)),
                         get_value<real_type>(config.at(1))};
    }
    GKO_INVALID_STATE("Can not get complex value");
}


#define SET_POINTER(_factory, _param_type, _param_name, _config, _context,     \
                    _exec, _td)                                                \
    {                                                                          \
        if (_config.contains(#_param_name)) {                                  \
            _factory.with_##_param_name(gko::config::get_pointer<_param_type>( \
                _config.at(#_param_name), _context, _exec, _td));              \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


#define SET_POINTER_VECTOR(_factory, _param_type, _param_name, _config,      \
                           _context, _exec, _td)                             \
    {                                                                        \
        if (_config.contains(#_param_name)) {                                \
            _factory.with_##_param_name(                                     \
                gko::config::get_pointer_vector<_param_type>(                \
                    _config.at(#_param_name), _context, _exec, _td));        \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define SET_VALUE(_factory, _param_type, _param_name, _config)               \
    {                                                                        \
        if (_config.contains(#_param_name)) {                                \
            _factory.with_##_param_name(gko::config::get_value<_param_type>( \
                _config.at(#_param_name)));                                  \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


// If we do not put the build_from_config in the class directly, the following
// can also be in internal header.
template <typename T>
struct type_string {
    static std::string str() { return "N"; };
};

#define TYPE_STRING_OVERLOAD(_type, _str)         \
    template <>                                   \
    struct type_string<_type> {                   \
        static std::string str() { return _str; } \
    }

TYPE_STRING_OVERLOAD(double, "double");
TYPE_STRING_OVERLOAD(float, "float");
TYPE_STRING_OVERLOAD(std::complex<double>, "complex<double>");
TYPE_STRING_OVERLOAD(std::complex<float>, "complex<float>");


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HPP_
