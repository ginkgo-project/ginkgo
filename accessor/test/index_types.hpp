// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACCESSOR_TEST_INDEX_TYPES_HPP_
#define ACCESSOR_TEST_INDEX_TYPES_HPP_

#include <cstdint>

#include <gtest/gtest.h>


namespace acc {
namespace test {


using AltIndexTypes =
    ::testing::Types<std::int64_t, std::int32_t, std::uint32_t>;


}  // namespace test
}  // namespace acc

#endif  // ACCESSOR_TEST_INDEX_TYPES_HPP_
