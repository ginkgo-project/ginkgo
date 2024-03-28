// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_CONFIG_HPP_
#define GKO_CORE_CONFIG_CONFIG_HPP_


#include <ginkgo/core/config/config.hpp>


#include <string>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
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

    if (auto& obj = config.get("ValueType")) {
        updated.first = obj.get_string();
    }
    if (auto& obj = config.get("IndexType")) {
        updated.second = obj.get_string();
    }
    return updated;
}


template <typename T>
inline std::shared_ptr<T> get_pointer(const pnode& config,
                                      const registry& context,
                                      type_descriptor td)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    ptr = context.search_data<T_non_const>(config.get_string());
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}


template <typename T>
inline deferred_factory_parameter<T> get_factory(const pnode& config,
                                                 const registry& context,
                                                 type_descriptor td);

template <>
inline deferred_factory_parameter<const LinOpFactory>
get_factory<const LinOpFactory>(const pnode& config, const registry& context,
                                type_descriptor td)
{
    deferred_factory_parameter<const LinOpFactory> ptr;
    if (config.get_status() == pnode::status_t::string) {
        ptr = context.search_data<LinOpFactory>(config.get_string());
    } else if (config.get_status() == pnode::status_t::map) {
        ptr = build_from_config(config, context, td);
    }
    // handle object is config
    assert(!ptr.is_empty());
    return std::move(ptr);
}

template <>
deferred_factory_parameter<const stop::CriterionFactory>
get_factory<const stop::CriterionFactory>(const pnode& config,
                                          const registry& context,
                                          type_descriptor td);


template <typename T>
inline std::vector<deferred_factory_parameter<T>> get_factory_vector(
    const pnode& config, const registry& context, type_descriptor td)
{
    std::vector<deferred_factory_parameter<T>> res;
    // for loop in config
    if (config.get_status() == pnode::status_t::array) {
        for (const auto& it : config.get_array()) {
            res.push_back(get_factory<T>(it, context, td));
        }
    } else {
        // only one config can be passed without array
        res.push_back(get_factory<T>(config, context, td));
    }

    return res;
}

// template <>
// std::vector<std::shared_ptr<const stop::CriterionFactory>>
// get_pointer_vector<const stop::CriterionFactory>(
//     const pnode& config, const registry& context,
//     std::shared_ptr<const Executor> exec, type_descriptor td);


template <typename IndexType>
inline
    typename std::enable_if<std::is_integral<IndexType>::value, IndexType>::type
    get_value(const pnode& config)
{
    auto val = config.get_integer();
    assert(val <= std::numeric_limits<IndexType>::max() &&
           val >= std::numeric_limits<IndexType>::min());
    return static_cast<IndexType>(val);
}

template <typename ValueType>
inline typename std::enable_if<std::is_floating_point<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config)
{
    auto val = config.get_real();
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
    if (config.get_status() == pnode::status_t::real) {
        return static_cast<ValueType>(get_value<real_type>(config));
    } else if (config.get_status() == pnode::status_t::array) {
        return ValueType{get_value<real_type>(config.get(0)),
                         get_value<real_type>(config.get(1))};
    }
    GKO_INVALID_STATE("Can not get complex value");
}


#define SET_POINTER(_factory, _param_type, _param_name, _config, _context,   \
                    _td)                                                     \
    {                                                                        \
        if (auto& obj = _config.get(#_param_name)) {                         \
            _factory.with_##_param_name(                                     \
                gko::config::get_pointer<_param_type>(obj, _context, _td));  \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define SET_FACTORY(_factory, _param_type, _param_name, _config, _context,   \
                    _td)                                                     \
    {                                                                        \
        if (auto& obj = _config.get(#_param_name)) {                         \
            _factory.with_##_param_name(                                     \
                gko::config::get_factory<_param_type>(obj, _context, _td));  \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define SET_FACTORY_VECTOR(_factory, _param_type, _param_name, _config,      \
                           _context, _td)                                    \
    {                                                                        \
        if (auto& obj = _config.get(#_param_name)) {                         \
            _factory.with_##_param_name(                                     \
                gko::config::get_factory_vector<_param_type>(obj, _context,  \
                                                             _td));          \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define SET_VALUE(_factory, _param_type, _param_name, _config)               \
    {                                                                        \
        if (auto& obj = _config.get(#_param_name)) {                         \
            _factory.with_##_param_name(                                     \
                gko::config::get_value<_param_type>(obj));                   \
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
