// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/config.hpp>


#include <map>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace config {


buildfromconfig_map generate_config_map()
{
    return {{"Cg", build_from_config<LinOpFactoryType::Cg>},
            {"Bicg", build_from_config<LinOpFactoryType::Bicg>},
            {"Bicgstab", build_from_config<LinOpFactoryType::Bicgstab>},
            {"Cgs", build_from_config<LinOpFactoryType::Cgs>},
            {"Fcg", build_from_config<LinOpFactoryType::Fcg>},
            {"Ir", build_from_config<LinOpFactoryType::Ir>},
            {"Idr", build_from_config<LinOpFactoryType::Idr>},
            {"Gcr", build_from_config<LinOpFactoryType::Gcr>},
            {"Gmres", build_from_config<LinOpFactoryType::Gmres>},
            {"CbGmres", build_from_config<LinOpFactoryType::CbGmres>},
            {"Direct", build_from_config<LinOpFactoryType::Direct>},
            {"LowerTrs", build_from_config<LinOpFactoryType::LowerTrs>},
            {"UpperTrs", build_from_config<LinOpFactoryType::UpperTrs>},
            {"Factorization_Ic",
             build_from_config<LinOpFactoryType::Factorization_Ic>},
            {"Factorization_Ilu",
             build_from_config<LinOpFactoryType::Factorization_Ilu>},
            {"Cholesky", build_from_config<LinOpFactoryType::Cholesky>},
            {"Lu", build_from_config<LinOpFactoryType::Lu>},
            {"ParIc", build_from_config<LinOpFactoryType::ParIc>},
            {"ParIct", build_from_config<LinOpFactoryType::ParIct>},
            {"ParIlu", build_from_config<LinOpFactoryType::ParIlu>},
            {"ParIlut", build_from_config<LinOpFactoryType::ParIlut>},
            {"Ic", build_from_config<LinOpFactoryType::Ic>},
            // {"Ilu", build_from_config<LinOpFactoryType::Ilu>},
            {"Isai", build_from_config<LinOpFactoryType::Isai>},
            {"Jacobi", build_from_config<LinOpFactoryType::Jacobi>}};
}


deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context, type_descriptor td)
{
    if (auto& obj = config.get("Type")) {
        auto func = context.get_build_map().at(obj.get_data<std::string>());
        return func(config, context, td);
    }
    GKO_INVALID_STATE("Should contain Type property");
}


}  // namespace config
}  // namespace gko
