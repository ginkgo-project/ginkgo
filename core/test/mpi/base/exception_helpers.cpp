// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


TEST(AssertNoMpiErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_MPI_ERRORS(1), gko::MpiError);
}


TEST(AssertNoMpiErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_MPI_ERRORS(MPI_SUCCESS));
}


}  // namespace
