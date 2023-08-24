// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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


enum LinOpFactoryType : int {
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


// It is only an intermediate step. If we do not provide the SolverType with VT,
// IT selection, it can be in detail namespace or hide it by structure
template <int flag>
deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context,
    type_descriptor td = {"", ""});


// The main function
deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context,
    type_descriptor td = {"", ""});


buildfromconfig_map generate_config_map();


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
