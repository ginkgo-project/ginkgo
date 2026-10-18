// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_
#define GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_


#include <cmath>
#include <complex>
#include <iostream>
#include <typeinfo>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/json.hpp"


/**
 * Converts a config property tree to json, with empty nodes and non-finite
 * reals as null.
 */
inline json pnode_to_json(const gko::config::pnode& node)
{
    using tag_t = gko::config::pnode::tag_t;
    switch (node.get_tag()) {
    case tag_t::map: {
        auto result = json::object();
        for (const auto& [key, child] : node.get_map()) {
            result[key] = pnode_to_json(child);
        }
        return result;
    }
    case tag_t::array: {
        auto result = json::array();
        for (const auto& child : node.get_array()) {
            result.push_back(pnode_to_json(child));
        }
        return result;
    }
    case tag_t::boolean:
        return node.get_boolean();
    case tag_t::integer:
        return node.get_integer();
    case tag_t::real: {
        auto value = node.get_real();
        if (!std::isfinite(value)) {
            return nullptr;
        }
        return value;
    }
    case tag_t::string:
        return node.get_string();
    default:
        return nullptr;
    }
}


namespace detail {


template <typename ValueType>
bool is_identity_factory(const gko::LinOpFactory& factory)
{
    return dynamic_cast<const gko::matrix::IdentityFactory<ValueType>*>(
               &factory) != nullptr;
}


}  // namespace detail


/**
 * The parameters of a factory as json, or null if they can not be written.
 */
inline json parameters_json(const gko::LinOpFactory& factory)
{
    if (detail::is_identity_factory<float>(factory) ||
        detail::is_identity_factory<double>(factory) ||
        detail::is_identity_factory<std::complex<float>>(factory) ||
        detail::is_identity_factory<std::complex<double>>(factory)) {
        return nullptr;
    }
    try {
        return pnode_to_json(gko::config::write_config(factory));
    } catch (const gko::NotSupported&) {
        std::cerr << "No config writer for "
                  << gko::name_demangling::get_type_name(typeid(factory))
                  << std::endl;
        return nullptr;
    }
}


#endif  // GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_
