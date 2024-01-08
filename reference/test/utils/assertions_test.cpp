// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/assertions.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {

template <typename T>
class MatricesNear : public ::testing::Test {};

TYPED_TEST_SUITE(MatricesNear, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MatricesNear, CanPassAnyMatrixType)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<gko::matrix::Dense<TypeParam>>(
        {{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}, exec);

    auto csr_mtx = gko::matrix::Csr<TypeParam>::create(exec);
    csr_mtx->copy_from(mtx);

    GKO_EXPECT_MTX_NEAR(csr_mtx, mtx, 0.0);
    GKO_ASSERT_MTX_NEAR(csr_mtx, mtx, 0.0);
}


}  // namespace
