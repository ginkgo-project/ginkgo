// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_ACCESSOR_UTILS_HPP_
#define GKO_CORE_TEST_ACCESSOR_UTILS_HPP_

#include <cstdint>

#include <gtest/gtest.h>


namespace gko {
namespace acc {
namespace test {


template <typename IndexType, typename SizeType = IndexType>
struct index_size_types {
    using index_type = IndexType;
    using size_type = SizeType;
};

using IndexSizeTypes =
    ::testing::Types<index_size_types<std::int64_t>,
                     index_size_types<std::int32_t>,
                     index_size_types<std::uint32_t>,
                     index_size_types<std::int64_t, std::int32_t>,
                     index_size_types<std::int32_t, std::int64_t>,
                     index_size_types<std::int64_t, std::uint32_t>,
                     index_size_types<std::uint32_t, std::int64_t>>;


}  // namespace test
}  // namespace acc
}  // namespace gko

#endif  // GKO_CORE_TEST_ACCESSOR_UTILS_HPP_
