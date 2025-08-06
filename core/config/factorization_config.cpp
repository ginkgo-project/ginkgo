// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/factorization/par_ict.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"


namespace gko {
namespace config {


GKO_PARSE_VALUE_AND_INDEX_TYPE(Factorization_Ic, gko::factorization::Ic);
GKO_PARSE_VALUE_AND_INDEX_TYPE(Factorization_Ilu, gko::factorization::Ilu);
GKO_PARSE_VALUE_AND_INDEX_TYPE(Cholesky,
                               gko::experimental::factorization::Cholesky);
GKO_PARSE_VALUE_AND_INDEX_TYPE(Lu, gko::experimental::factorization::Lu);
GKO_PARSE_VALUE_AND_INDEX_TYPE(ParIlu, gko::factorization::ParIlu);
GKO_PARSE_VALUE_AND_INDEX_TYPE(ParIlut, gko::factorization::ParIlut);
GKO_PARSE_VALUE_AND_INDEX_TYPE(ParIc, gko::factorization::ParIc);
GKO_PARSE_VALUE_AND_INDEX_TYPE(ParIct, gko::factorization::ParIct);


}  // namespace config
}  // namespace gko
