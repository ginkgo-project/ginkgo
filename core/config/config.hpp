// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_CONFIG_HPP_
#define GKO_CORE_CONFIG_CONFIG_HPP_


#include <ginkgo/core/config/config.hpp>


#include <string>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
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
        updated.first = obj.get_data<std::string>();
    }
    if (auto& obj = config.get("IndexType")) {
        updated.second = obj.get_data<std::string>();
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
    if (config.is(pnode::status_t::object)) {
        ptr = context.search_data<T_non_const>(config.get_data<std::string>());
    } else if (config.is(pnode::status_t::list) &&
               std::is_convertible<T_non_const*, LinOpFactory*>::value) {
        ptr = std::dynamic_pointer_cast<T_non_const>(
            gko::share(build_from_config(config, context, exec, td)));
    }
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
    if (config.is(pnode::status_t::data)) {
        ptr = context.search_data<LinOpFactory>(config.get_data<std::string>());
    } else if (config.is(pnode::status_t::map)) {
        ptr = build_from_config(config, context, td);
    }
    // handle object is config
    assert(!ptr.is_empty());
    return std::move(ptr);
}


template <typename Csr>
inline std::shared_ptr<typename Csr::strategy_type> get_strategy(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor> exec, type_descriptor td)
{
    auto str = config.get_data<std::string>();
    std::shared_ptr<typename Csr::strategy_type> strategy_ptr;
    if (str == "sparselib" || str == "cusparse") {
        strategy_ptr = std::make_shared<typename Csr::sparselib>();
    } else if (str == "automatical") {
        if (auto explicit_exec =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
            strategy_ptr =
                std::make_shared<typename Csr::automatical>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::HipExecutor>(
                           exec)) {
            strategy_ptr =
                std::make_shared<typename Csr::automatical>(explicit_exec);
        } else {
            strategy_ptr = std::make_shared<typename Csr::automatical>(256);
        }
    } else if (str == "load_balance") {
        if (auto explicit_exec =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
            strategy_ptr =
                std::make_shared<typename Csr::load_balance>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::HipExecutor>(
                           exec)) {
            strategy_ptr =
                std::make_shared<typename Csr::load_balance>(explicit_exec);
        } else {
            strategy_ptr = std::make_shared<typename Csr::load_balance>(256);
        }

    } else if (str == "merge_path") {
        strategy_ptr = std::make_shared<typename Csr::merge_path>();
    } else if (str == "classical") {
        strategy_ptr = std::make_shared<typename Csr::classical>();
    }
    return std::move(strategy_ptr);
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
    if (config.is(pnode::status_t::array)) {
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


template <typename ValueType>
inline typename std::enable_if<std::is_same<ValueType, bool>::value, bool>::type
get_value(const pnode& config)
{
    auto val = config.get_data<bool>();
    return val;
}

template <typename IndexType>
inline typename std::enable_if<std::is_integral<IndexType>::value &&
                                   !std::is_same<IndexType, bool>::value,
                               IndexType>::type
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
    if (config.is(pnode::status_t::data)) {
        return static_cast<ValueType>(get_value<real_type>(config));
    } else if (config.is(pnode::status_t::array)) {
        return ValueType{get_value<real_type>(config.at(0)),
                         get_value<real_type>(config.at(1))};
    }
    GKO_INVALID_STATE("Can not get complex value");
}

template <typename Type>
inline typename std::enable_if<std::is_same<Type, precision_reduction>::value,
                               Type>::type
get_value(const pnode& config)
{
    using T = typename Type::storage_type;
    if (config.is(pnode::status_t::array) && config.get_array().size() == 2) {
        return Type(get_value<T>(config.at(0)), get_value<T>(config.at(1)));
    }
    GKO_INVALID_STATE("should use size 2 array");
}

template <typename ValueType>
inline typename std::enable_if<
    std::is_same<ValueType, solver::initial_guess_mode>::value,
    solver::initial_guess_mode>::type
get_value(const pnode& config)
{
    auto val = config.get_data<std::string>();
    if (val == "zero") {
        return solver::initial_guess_mode::zero;
    } else if (val == "rhs") {
        return solver::initial_guess_mode::rhs;
    } else if (val == "provided") {
        return solver::initial_guess_mode::provided;
    }
    GKO_INVALID_STATE("Wrong value for initial_guess_mode");
}


template <typename T>
struct is_array_t : std::false_type {};

template <typename V>
struct is_array_t<array<V>> : std::true_type {};

template <typename ArrayType>
inline typename std::enable_if<is_array_t<ArrayType>::value, ArrayType>::type
get_value(const pnode& config, std::shared_ptr<const Executor> exec)
{
    using T = typename ArrayType::value_type;
    std::vector<T> res;
    // for loop in config
    if (config.is(pnode::status_t::array)) {
        for (const auto& it : config.get_array()) {
            res.push_back(get_value<T>(it));
        }
    } else {
        // only one config can be passed without array
        res.push_back(get_value<T>(config));
    }
    return ArrayType(exec, res.begin(), res.end());
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

#define SET_CSR_STRATEGY(_factory, _csr_type, _param_name, _config, _context, \
                         _exec, _td)                                          \
    {                                                                         \
        if (_config.contains(#_param_name)) {                                 \
            _factory.with_##_param_name(gko::config::get_strategy<_csr_type>( \
                _config.at(#_param_name), _context, _exec, _td));             \
        }                                                                     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
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

#define SET_VALUE_ARRAY(_factory, _param_type, _param_name, _config, _exec)  \
    {                                                                        \
        if (_config.contains(#_param_name)) {                                \
            _factory.with_##_param_name(gko::config::get_value<_param_type>( \
                _config.at(#_param_name), _exec));                           \
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
TYPE_STRING_OVERLOAD(gko::int32, "int");
TYPE_STRING_OVERLOAD(gko::int64, "int64");


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HPP_
