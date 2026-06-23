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


using AltIndexTypes =
    ::testing::Types<std::int64_t, std::int32_t, std::uint32_t>;


}  // namespace test
}  // namespace acc
}  // namespace gko

#endif  // GKO_CORE_TEST_ACCESSOR_UTILS_HPP_
