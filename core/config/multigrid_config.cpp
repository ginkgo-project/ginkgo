// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/config/parse_macro.hpp"
#include "ginkgo/core/multigrid/pgm.hpp"


namespace gko {
namespace config {


GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(Pgm, gko::multigrid::Pgm);


}  // namespace config
}  // namespace gko
