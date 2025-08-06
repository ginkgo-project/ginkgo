// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


GKO_PARSE_VALUE_AND_INDEX_TYPE_BASE(GaussSeidel,
                                    gko::preconditioner::GaussSeidel);
GKO_PARSE_VALUE_AND_INDEX_TYPE(Jacobi, gko::preconditioner::Jacobi);
GKO_PARSE_VALUE_AND_INDEX_TYPE_BASE(Sor, gko::preconditioner::Sor);


}  // namespace config
}  // namespace gko
