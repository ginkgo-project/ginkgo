// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_REGISTRY_ACCESSOR_HPP_
#define GKO_CORE_CONFIG_REGISTRY_ACCESSOR_HPP_


#include <string>

#include <ginkgo/core/config/registry.hpp>


namespace gko {
namespace config {
namespace detail {


class registry_accessor {
public:
    template <typename T>
    static inline std::shared_ptr<T> get_data(const registry& reg,
                                              std::string key)
    {
        return reg.get_data<T>(key);
    }

    static inline const configuration_map& get_build_map(const registry& reg)
    {
        return reg.get_build_map();
    }
};


}  // namespace detail
}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_REGISTRY_ACCESSOR_HPP_
