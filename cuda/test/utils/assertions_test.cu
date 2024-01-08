// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/assertions.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/test/utils.hpp"


namespace {


class MatricesNear : public CudaTestFixture {};


TEST_F(MatricesNear, CanPassCudaMatrix)
{
    auto mtx = gko::initialize<gko::matrix::Dense<>>(
        {{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}, ref);
    auto csr_ref = gko::matrix::Csr<>::create(ref);
    csr_ref->copy_from(mtx);
    auto csr_mtx = gko::matrix::Csr<>::create(exec);
    csr_mtx->move_from(csr_ref);

    GKO_EXPECT_MTX_NEAR(csr_mtx, mtx, 0.0);
    GKO_ASSERT_MTX_NEAR(csr_mtx, mtx, 0.0);
}


}  // namespace
