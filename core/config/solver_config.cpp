// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/solver_config.hpp"


namespace gko {
namespace config {


PARSE_VALUE_TYPE(Cg, gko::solver::Cg);
PARSE_VALUE_TYPE(Bicg, gko::solver::Bicg);
PARSE_VALUE_TYPE(Bicgstab, gko::solver::Bicgstab);
PARSE_VALUE_TYPE(Cgs, gko::solver::Cgs);
PARSE_VALUE_TYPE(Fcg, gko::solver::Fcg);
PARSE_VALUE_TYPE(Ir, gko::solver::Ir);
PARSE_VALUE_TYPE(Idr, gko::solver::Idr);
PARSE_VALUE_TYPE(Gcr, gko::solver::Gcr);
PARSE_VALUE_TYPE(Gmres, gko::solver::Gmres);
PARSE_VALUE_TYPE(CbGmres, gko::solver::CbGmres);
PARSE_VALUE_AND_INDEX_TYPE(Direct, gko::experimental::solver::Direct);
PARSE_VALUE_AND_INDEX_TYPE(LowerTrs, gko::solver::LowerTrs);
PARSE_VALUE_AND_INDEX_TYPE(UpperTrs, gko::solver::UpperTrs);


}  // namespace config
}  // namespace gko
