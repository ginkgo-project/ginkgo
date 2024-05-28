// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_INDEX_MAP_FWD_HPP
#define GINKGO_INDEX_MAP_FWD_HPP

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

#endif  // GINKGO_INDEX_MAP_FWD_HPP
