// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_


#include <map>
#include <string>
#include <unordered_map>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace config {


/**
 * build_from_config is the main function for file config. It can read the
 * property tree to create the desired type. `build_from_config(...).on(exec) ->
 * LinOpFactory`.
 *
 * @param config  The property tree which must include `Type` for the class base
 * and the corresponding template selection.
 * @param context  The registry which stores the building function map and the
 * storage for generated object.
 * @param type_descriptor  The default common type. If the ValueType or
 * IndexType is required by the class base but user does not provide it, this
 * function will take this input as the default.
 *
 * @return deferred_factory_parameter, user can get LinOpFactory after giving
 * the executor.
 */
deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context,
    type_descriptor td = {"", ""});


/**
 * Generate the configuration map.
 */
buildfromconfig_map generate_config_map();


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
