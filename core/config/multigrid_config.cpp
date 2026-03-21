// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/config/parse_macro.hpp"
#include "ginkgo/core/multigrid/hmis.hpp"
#include "ginkgo/core/multigrid/pgm.hpp"


namespace gko {
namespace config {


GKO_PARSE_VALUE_AND_INDEX_TYPE(Hmis, gko::multigrid::Hmis);
GKO_PARSE_VALUE_AND_INDEX_TYPE(Pgm, gko::multigrid::Pgm);


}  // namespace config
}  // namespace gko
