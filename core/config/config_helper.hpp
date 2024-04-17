// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_CONFIG_HELPER_HPP_
#define GKO_CORE_CONFIG_CONFIG_HELPER_HPP_


#include <string>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


/**
 * LinOpFactoryType enum is to avoid forward declartion, linopfactory header,
 * two template versions of build_from_config
 */
enum class LinOpFactoryType : int { Cg = 0 };


/**
 * It is only an intermediate step after dispatching the class base type. Each
 * implementation needs to deal with the template selection.
 */
template <LinOpFactoryType flag>
deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context,
    type_descriptor td = type_descriptor{"void", "void"});

/**
 * This function is to update the default type setting from current config.
 *
 * @note It might update the unused type for the current class.
 */
type_descriptor update_type(const pnode& config, const type_descriptor& td);

/**
 * get_pointer searches the pointer in the registry by string
 */
template <typename T>
inline std::shared_ptr<T> get_pointer(const pnode& config,
                                      const registry& context,
                                      type_descriptor td);


/**
 * get_factory builds the factory from config (map) or searches the pointers in
 * the registry by string.
 */
template <typename T>
deferred_factory_parameter<T> get_factory(const pnode& config,
                                          const registry& context,
                                          type_descriptor td);

/**
 * specialize for const LinOpFactory
 */
template <>
deferred_factory_parameter<const LinOpFactory> get_factory<const LinOpFactory>(
    const pnode& config, const registry& context, type_descriptor td);

/**
 * specialize for const stop::CriterionFactory
 */
template <>
deferred_factory_parameter<const stop::CriterionFactory>
get_factory<const stop::CriterionFactory>(const pnode& config,
                                          const registry& context,
                                          type_descriptor td);

/**
 * get_factory_vector will gives a vector of factory by calling get_factory.
 */
template <typename T>
inline std::vector<deferred_factory_parameter<T>> get_factory_vector(
    const pnode& config, const registry& context, type_descriptor td);


/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for integral type
 */
template <typename IndexType>
inline
    typename std::enable_if<std::is_integral<IndexType>::value, IndexType>::type
    get_value(const pnode& config);

/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for floating point type
 */
template <typename ValueType>
inline typename std::enable_if<std::is_floating_point<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config);

/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for complex type
 */
template <typename ValueType>
inline typename std::enable_if<gko::is_complex_s<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config);


// type_string providing the mapping from type to string.
template <typename T>
struct type_string {};

#define TYPE_STRING_OVERLOAD(_type, _str)         \
    template <>                                   \
    struct type_string<_type> {                   \
        static std::string str() { return _str; } \
    }

TYPE_STRING_OVERLOAD(void, "void");
TYPE_STRING_OVERLOAD(double, "double");
TYPE_STRING_OVERLOAD(float, "float");
TYPE_STRING_OVERLOAD(std::complex<double>, "complex<double>");
TYPE_STRING_OVERLOAD(std::complex<float>, "complex<float>");
TYPE_STRING_OVERLOAD(int32, "int");
TYPE_STRING_OVERLOAD(int64, "int64");


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
inline std::vector<deferred_factory_parameter<T>> get_factory_vector(
    const pnode& config, const registry& context, type_descriptor td)
{
    std::vector<deferred_factory_parameter<T>> res;
    // for loop in config
    if (config.get_tag() == pnode::tag_t::array) {
        for (const auto& it : config.get_array()) {
            res.push_back(get_factory<T>(it, context, td));
        }
    } else {
        // only one config can be passed without array
        res.push_back(get_factory<T>(config, context, td));
    }

    return res;
}


template <typename IndexType>
inline
    typename std::enable_if<std::is_integral<IndexType>::value, IndexType>::type
    get_value(const pnode& config)
{
    auto val = config.get_integer();
    GKO_THROW_IF_INVALID(
        val <= std::numeric_limits<IndexType>::max() &&
            val >= std::numeric_limits<IndexType>::min(),
        "the config value is out of the range of the require type.");
    return static_cast<IndexType>(val);
}

template <typename ValueType>
inline typename std::enable_if<std::is_floating_point<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config)
{
    auto val = config.get_real();
    // the max, min of floating point only consider positive value.
    GKO_THROW_IF_INVALID(
        val <= std::numeric_limits<ValueType>::max() &&
            val >= -std::numeric_limits<ValueType>::max(),
        "the config value is out of the range of the require type.");
    return static_cast<ValueType>(val);
}

template <typename ValueType>
inline typename std::enable_if<gko::is_complex_s<ValueType>::value,
                               ValueType>::type
get_value(const pnode& config)
{
    using real_type = gko::remove_complex<ValueType>;
    if (config.get_tag() == pnode::tag_t::real) {
        return static_cast<ValueType>(get_value<real_type>(config));
    } else if (config.get_tag() == pnode::tag_t::array) {
        return ValueType{get_value<real_type>(config.get(0)),
                         get_value<real_type>(config.get(1))};
    }
    GKO_INVALID_STATE("Can not get complex value");
}


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HELPER_HPP_
