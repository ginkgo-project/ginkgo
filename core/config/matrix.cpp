// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"


namespace gko {
namespace config {


template <typename ValueType>
struct IdentityParser {
    static typename matrix::IdentityFactory<ValueType>::parameters_type parse(
        const pnode& config, const registry& context,
        const type_descriptor& td_for_child)
    {
        return {};
    }
};

GKO_PARSE_VALUE_TYPE(Identity, IdentityParser);


}  // namespace config
}  // namespace gko
