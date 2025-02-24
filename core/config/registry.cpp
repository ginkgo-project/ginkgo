// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/registry.hpp"

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>

#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


configuration_map generate_config_map()
{
    return
    {
        {"solver::Cg", parse<LinOpFactoryType::Cg>},
            {"solver::Bicg", parse<LinOpFactoryType::Bicg>},
            {"solver::Bicgstab", parse<LinOpFactoryType::Bicgstab>},
            {"solver::Fcg", parse<LinOpFactoryType::Fcg>},
            {"solver::Cgs", parse<LinOpFactoryType::Cgs>},
            {"solver::Ir", parse<LinOpFactoryType::Ir>},
            {"solver::Idr", parse<LinOpFactoryType::Idr>},
            {"solver::Gcr", parse<LinOpFactoryType::Gcr>},
            {"solver::Gmres", parse<LinOpFactoryType::Gmres>},
            {"solver::CbGmres", parse<LinOpFactoryType::CbGmres>},
            {"solver::Minres", parse<LinOpFactoryType::Minres>},
            {"solver::Direct", parse<LinOpFactoryType::Direct>},
            {"solver::LowerTrs", parse<LinOpFactoryType::LowerTrs>},
            {"solver::UpperTrs", parse<LinOpFactoryType::UpperTrs>},
            {"factorization::Ic", parse<LinOpFactoryType::Factorization_Ic>},
            {"factorization::Ilu", parse<LinOpFactoryType::Factorization_Ilu>},
            {"factorization::Cholesky", parse<LinOpFactoryType::Cholesky>},
            {"factorization::Lu", parse<LinOpFactoryType::Lu>},
            {"factorization::ParIc", parse<LinOpFactoryType::ParIc>},
            {"factorization::ParIct", parse<LinOpFactoryType::ParIct>},
            {"factorization::ParIlu", parse<LinOpFactoryType::ParIlu>},
            {"factorization::ParIlut", parse<LinOpFactoryType::ParIlut>},
            {"preconditioner::GaussSeidel",
             parse<LinOpFactoryType::GaussSeidel>},
            {"preconditioner::Ic", parse<LinOpFactoryType::Ic>},
            {"preconditioner::Ilu", parse<LinOpFactoryType::Ilu>},
            {"preconditioner::Isai", parse<LinOpFactoryType::Isai>},
            {"preconditioner::Jacobi", parse<LinOpFactoryType::Jacobi>},
            {"preconditioner::Sor", parse<LinOpFactoryType::Sor>},
            {"solver::Multigrid", parse<LinOpFactoryType::Multigrid>},
            {"multigrid::Pgm", parse<LinOpFactoryType::Pgm>},
#if GINKGO_BUILD_MPI
        {
            "preconditioner::Schwarz", parse<LinOpFactoryType::Schwarz>
        }
#endif
    };
}


registry::registry(const configuration_map& additional_map)
    : registry({}, additional_map)
{}


registry::registry(
    const std::unordered_map<std::string, detail::allowed_ptr>& stored_map,
    const configuration_map& additional_map)
    : stored_map_(stored_map), build_map_(generate_config_map())
{
    // merge additional_map into build_map_
    for (auto& item : additional_map) {
        auto res = build_map_.emplace(item.first, item.second);
        GKO_THROW_IF_INVALID(res.second,
                             "failed when adding the key " + item.first);
    }
}


}  // namespace config
}  // namespace gko
