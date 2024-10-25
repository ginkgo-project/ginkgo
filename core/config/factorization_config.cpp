// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(Factorization_Ic,
                                         gko::factorization::Ic);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(Factorization_Ilu,
                                         gko::factorization::Ilu);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(
    Cholesky, gko::experimental::factorization::Cholesky);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(Lu,
                                         gko::experimental::factorization::Lu);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(ParIlu, gko::factorization::ParIlu);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(ParIlut, gko::factorization::ParIlut);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(ParIc, gko::factorization::ParIc);
GKO_PARSE_VALUE_AND_INDEX_TYPE_WITH_HALF(ParIct, gko::factorization::ParIct);


}  // namespace config
}  // namespace gko
