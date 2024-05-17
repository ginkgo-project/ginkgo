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


#include "core/config/registry_accessor.hpp"


namespace gko {
namespace config {


#define GKO_INVALID_CONFIG_VALUE(_entry, _value)                            \
    GKO_INVALID_STATE(std::string("The value >" + _value +                  \
                                  "< is invalid for the entry >" + _entry + \
                                  "<"))


#define GKO_MISS_CONFIG_ENTRY(_entry) \
    GKO_INVALID_STATE(std::string("The entry >") + _entry + "< is missing")


/**
 * LinOpFactoryType enum is to avoid forward declaration, linopfactory header,
 * two template versions of parse
 */
enum class LinOpFactoryType : int {
    Cg = 0,
    Bicg,
    Bicgstab,
    Fcg,
    Cgs,
    Ir,
    Idr,
    Gcr,
    Gmres,
    CbGmres,
    Direct,
    LowerTrs,
    UpperTrs,
    Factorization_Ic,
    Factorization_Ilu,
    Cholesky,
    Lu,
    ParIc,
    ParIct,
    ParIlu,
    ParIlut,
    Ic,
    Ilu,
    Isai,
    Jacobi
};


/**
 * It is only an intermediate step after dispatching the class base type. Each
 * implementation needs to deal with the template selection.
 */
template <LinOpFactoryType flag>
deferred_factory_parameter<gko::LinOpFactory> parse(
    const pnode& config, const registry& context,
    const type_descriptor& td = make_type_descriptor<>());


/**
 * get_stored_obj searches the object pointer stored in the registry by string
 */
template <typename T>
inline std::shared_ptr<T> get_stored_obj(const pnode& config,
                                         const registry& context)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    ptr = detail::registry_accessor::get_data<T_non_const>(context,
                                                           config.get_string());
    GKO_THROW_IF_INVALID(ptr.get() != nullptr, "Do not get the stored data");
    return ptr;
}


/**
 * Build the factory from config (map) or search the pointers in
 * the registry by string.
 */
template <typename T>
deferred_factory_parameter<T> parse_or_get_factory(const pnode& config,
                                                   const registry& context,
                                                   const type_descriptor& td);

template <typename T>
inline deferred_factory_parameter<typename T::Factory>
parse_or_get_specific_factory(const pnode& config, const registry& context,
                              const type_descriptor& td)
{
    using T_non_const = std::remove_const_t<T>;

    if (config.get_tag() == pnode::tag_t::string) {
        return detail::registry_accessor::get_data<
            typename T_non_const::Factory>(context, config.get_string());
    } else if (config.get_tag() == pnode::tag_t::map) {
        return T_non_const::parse(config, context, td);
    } else {
        GKO_INVALID_STATE("The data of config is not valid.");
    }
}


template <>
deferred_factory_parameter<const LinOpFactory>
parse_or_get_factory<const LinOpFactory>(const pnode& config,
                                         const registry& context,
                                         const type_descriptor& td);


/**
 * specialize for const stop::CriterionFactory
 */
template <>
deferred_factory_parameter<const stop::CriterionFactory>
parse_or_get_factory<const stop::CriterionFactory>(const pnode& config,
                                                   const registry& context,
                                                   const type_descriptor& td);

/**
 * give a vector of factory by calling parse_or_get_factory.
 */
template <typename T>
inline std::vector<deferred_factory_parameter<T>> parse_or_get_factory_vector(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    std::vector<deferred_factory_parameter<T>> res;
    if (config.get_tag() == pnode::tag_t::array) {
        for (const auto& it : config.get_array()) {
            res.push_back(parse_or_get_factory<T>(it, context, td));
        }
    } else {
        // only one config can be passed without array
        res.push_back(parse_or_get_factory<T>(config, context, td));
    }

    return res;
}


/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for bool type
 */
template <typename ValueType>
inline std::enable_if_t<std::is_same<ValueType, bool>::value, bool> get_value(
    const pnode& config)
{
    auto val = config.get_boolean();
    return val;
}


/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for integral type
 */
template <typename IndexType>
inline std::enable_if_t<std::is_integral<IndexType>::value &&
                            !std::is_same<IndexType, bool>::value,
                        IndexType>
get_value(const pnode& config)
{
    auto val = config.get_integer();
    GKO_THROW_IF_INVALID(
        val <= std::numeric_limits<IndexType>::max() &&
            val >= std::numeric_limits<IndexType>::min(),
        "the config value is out of the range of the require type.");
    return static_cast<IndexType>(val);
}


/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for floating point type
 */
template <typename ValueType>
inline std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType>
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

/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for complex type
 */
template <typename ValueType>
inline std::enable_if_t<gko::is_complex_s<ValueType>::value, ValueType>
get_value(const pnode& config)
{
    using real_type = gko::remove_complex<ValueType>;
    if (config.get_tag() == pnode::tag_t::real) {
        return static_cast<ValueType>(get_value<real_type>(config));
    } else if (config.get_tag() == pnode::tag_t::array) {
        real_type real(0);
        real_type imag(0);
        if (config.get_array().size() >= 1) {
            real = get_value<real_type>(config.get(0));
        }
        if (config.get_array().size() >= 2) {
            imag = get_value<real_type>(config.get(1));
        }
        GKO_THROW_IF_INVALID(
            config.get_array().size() <= 2,
            "complex value array expression only accept up to two elements");
        return ValueType{real, imag};
    }
    GKO_INVALID_STATE("Can not get complex value");
}


/**
 * get_value gets the corresponding type value from config.
 *
 * This is specialization for initial_guess_mode
 */
template <typename ValueType>
inline std::enable_if_t<
    std::is_same<ValueType, solver::initial_guess_mode>::value,
    solver::initial_guess_mode>
get_value(const pnode& config)
{
    auto val = config.get_string();
    if (val == "zero") {
        return solver::initial_guess_mode::zero;
    } else if (val == "rhs") {
        return solver::initial_guess_mode::rhs;
    } else if (val == "provided") {
        return solver::initial_guess_mode::provided;
    }
    GKO_INVALID_CONFIG_VALUE("default_initial_guess", val);
}


template <typename Type>
inline typename std::enable_if<std::is_same<Type, precision_reduction>::value,
                               Type>::type
get_value(const pnode& config)
{
    using T = typename Type::storage_type;
    if (config.get_tag() == pnode::tag_t::array &&
        config.get_array().size() == 2) {
        return Type(get_value<T>(config.get(0)), get_value<T>(config.get(1)));
    }
    GKO_INVALID_STATE("should use size 2 array");
}


template <typename Csr>
inline std::shared_ptr<typename Csr::strategy_type> get_strategy(
    const pnode& config)
{
    auto str = config.get_string();
    std::shared_ptr<typename Csr::strategy_type> strategy_ptr;
    // automatical and load_balance requires the executor
    if (str == "sparselib" || str == "cusparse") {
        strategy_ptr = std::make_shared<typename Csr::sparselib>();
    } else if (str == "merge_path") {
        strategy_ptr = std::make_shared<typename Csr::merge_path>();
    } else if (str == "classical") {
        strategy_ptr = std::make_shared<typename Csr::classical>();
    } else {
        GKO_INVALID_CONFIG_VALUE("strategy", str);
    }
    return std::move(strategy_ptr);
}


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HELPER_HPP_
