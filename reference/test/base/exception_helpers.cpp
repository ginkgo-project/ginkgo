// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


TEST(AssertNoLapackErrors, ThrowsOnError)
{
    int info;
    ASSERT_THROW(GKO_ASSERT_NO_LAPACK_ERRORS(info = 1, info), gko::LapackError);
}


TEST(AssertNoLapackErrors, DoesNotThrowOnSuccess)
{
    int info;
    ASSERT_NO_THROW(GKO_ASSERT_NO_LAPACK_ERRORS(info = 0, info));
}


}  // namespace
