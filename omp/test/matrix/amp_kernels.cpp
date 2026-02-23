// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"


namespace {


class Amp : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::Ell<value_type, index_type>;

    Amp() : rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::default_random_engine rand_engine;
};


TEST_F(Amp, GenerateEllRownormsStorageIsEquivalentToRef)
{
    using T = value_type;
    using IndexType = index_type;
    using real_T = gko::remove_complex<T>;
    constexpr int q = gko::matrix::AMP<T, IndexType>::num_precisions;
    const float tol = 1e-10;
    auto mtx = gen_mtx(532, 231);
    auto dmtx = gko::clone(omp, mtx);
    gko::amp::precision_array<int, T> ref_max_nnz;
    gko::amp::precision_array<int, T> omp_max_nnz;
    gko::array<real_T> ref_rownorms(ref, mtx->get_size()[0]);
    gko::array<real_T> omp_rownorms(omp, dmtx->get_size()[0]);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        ref, mtx.get(), tol, ref_max_nnz, ref_rownorms);
    gko::kernels::omp::amp::generate_ell_rownorms_storage(
        omp, dmtx.get(), tol, omp_max_nnz, omp_rownorms);

    for (int k = 0; k < q; k++) {
        EXPECT_EQ(omp_max_nnz[k], ref_max_nnz[k]);
    }
    GKO_ASSERT_ARRAY_NEAR(omp_rownorms, ref_rownorms,
                          std::numeric_limits<real_T>::epsilon());
}


}  // namespace
