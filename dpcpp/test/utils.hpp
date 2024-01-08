// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_TEST_UTILS_HPP_
#define GKO_DPCPP_TEST_UTILS_HPP_


#include <gtest/gtest.h>


namespace {


#if GINKGO_DPCPP_SINGLE_MODE
#define SKIP_IF_SINGLE_MODE GTEST_SKIP() << "Skip due to single mode"
#else
#define SKIP_IF_SINGLE_MODE                                                  \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


}  // namespace


#endif  // GKO_DPCPP_TEST_UTILS_HPP_
