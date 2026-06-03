// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_builder.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class CsrBuilder : public CommonTestFixture {
protected:
    using Arr = gko::array<index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::Csr<value_type>;

    CsrBuilder()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 231),
#endif
          rand_engine(42)
    {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row, int max_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, max_nnz_row),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gen_mtx<MtxType>(num_rows, num_cols, min_nnz_row, num_cols);
    }

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;
};


TEST_F(CsrBuilder, SrowIsCorrectFromLoadBalance)
{
    auto mtx = Mtx::create(exec, gko::matrix::csr::spmv_strategy::load_balance);
    mtx->move_from(gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 1));
    int warp_size = 0;
    if (auto dexec = std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) {
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec)) {
        warp_size = 32;
    }
    const auto srow_size = mtx->get_num_srow_elements();
    auto srow_view = gko::make_array_view(exec, srow_size, mtx->get_srow());
    Arr original_srow(exec);
    // keep the original srow as answer
    original_srow = srow_view;
    std::vector<index_type> changed_srow(srow_size, -1);
    Arr changed_srow_view(ref, changed_srow.begin(), changed_srow.end());
    // modify srow to -1
    srow_view = changed_srow_view;

    ASSERT_NE(original_srow.get_data(), mtx->get_srow());
    GKO_ASSERT_ARRAY_EQ(srow_view, changed_srow_view);
    // after CsrBuilder destructor, it should rebuild srow to original one
    gko::matrix::CsrBuilder<value_type, index_type>{mtx};
    GKO_ASSERT_ARRAY_EQ(srow_view, original_srow);
}
