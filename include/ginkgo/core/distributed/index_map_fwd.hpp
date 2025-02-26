// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_INDEX_MAP_FWD_HPP
#define GKO_PUBLIC_CORE_INDEX_MAP_FWD_HPP

#include <variant>

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
class index_map;

using index_map_variant =
    std::variant<index_map<int32, int32>, index_map<int32, int64>,
                 index_map<int64, int64>>;


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_INDEX_MAP_FWD_HPP
