// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/hmis_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/hmis.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class HmisKernels : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Hmis<value_type, index_type>;

    HmisKernels() : exec(gko::ReferenceExecutor::create()) {}

    /**
     * Creates a 1D Laplacian (tridiagonal) of size n:
     *   2 -1  0  0  ...
     *  -1  2 -1  0  ...
     *   0 -1  2 -1  ...
     *   ...
     */
    std::shared_ptr<Mtx> create_1d_laplacian(gko::size_type n)
    {
        gko::matrix_data<value_type, index_type> data{gko::dim<2>{n, n}};
        for (gko::size_type i = 0; i < n; ++i) {
            data.nonzeros.emplace_back(i, i, value_type{2.0});
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, value_type{-1.0});
            }
            if (i + 1 < n) {
                data.nonzeros.emplace_back(i, i + 1, value_type{-1.0});
            }
        }
        data.sort_row_major();
        auto mtx = Mtx::create(exec);
        mtx->read(data);
        return mtx;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(HmisKernels, gko::test::RealValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(HmisKernels, ComputeStrengthSmallLaplacian)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // 4x4 1D Laplacian:
    //  2 -1  0  0
    // -1  2 -1  0
    //  0 -1  2 -1
    //  0  0 -1  2
    auto mtx = this->create_1d_laplacian(4);
    const index_type num_rows = 4;

    gko::array<index_type> strength_row_ptrs(this->exec, num_rows + 1);
    gko::array<index_type> strength_col_idxs(this->exec);
    gko::kernels::reference::hmis::compute_strength(
        this->exec, mtx.get(),
        static_cast<gko::remove_complex<value_type>>(0.25),
        strength_row_ptrs.get_data(), strength_col_idxs);

    // With theta=0.25 and max(-a_ik) = 1 for all rows,
    // threshold = 0.25, and -a_ij = 1 >= 0.25 for all off-diag entries.
    // So ALL off-diagonal connections are strong.
    auto rp = strength_row_ptrs.get_const_data();
    ASSERT_EQ(rp[0], 0);
    ASSERT_EQ(rp[1], 1);  // row 0: col 1
    ASSERT_EQ(rp[2], 3);  // row 1: col 0, 2
    ASSERT_EQ(rp[3], 5);  // row 2: col 1, 3
    ASSERT_EQ(rp[4], 6);  // row 3: col 2

    auto ci = strength_col_idxs.get_const_data();
    EXPECT_EQ(ci[0], 1);
    EXPECT_EQ(ci[1], 0);
    EXPECT_EQ(ci[2], 2);
    EXPECT_EQ(ci[3], 1);
    EXPECT_EQ(ci[4], 3);
    EXPECT_EQ(ci[5], 2);
}


TYPED_TEST(HmisKernels, RueStubenFirstPassSmallLaplacian)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = this->create_1d_laplacian(5);
    const index_type num_rows = 5;

    // First compute strength
    gko::array<index_type> s_row_ptrs(this->exec, num_rows + 1);
    gko::array<index_type> s_col_idxs(this->exec);
    gko::kernels::reference::hmis::compute_strength(
        this->exec, mtx.get(),
        static_cast<gko::remove_complex<value_type>>(0.25),
        s_row_ptrs.get_data(), s_col_idxs);

    // Build strength CSR
    auto s_nnz = s_col_idxs.get_size();
    gko::array<value_type> s_vals(this->exec, s_nnz);
    for (gko::size_type i = 0; i < s_nnz; ++i) {
        s_vals.get_data()[i] = gko::one<value_type>();
    }
    auto strength = Mtx::create(
        this->exec,
        gko::dim<2>{static_cast<gko::size_type>(num_rows),
                    static_cast<gko::size_type>(num_rows)},
        std::move(s_vals), std::move(s_col_idxs), std::move(s_row_ptrs));
    auto strength_t = gko::as<Mtx>(gko::share(strength->transpose()));

    // Run Ruge-Stueben first pass
    gko::array<index_type> cf_marker(this->exec, num_rows);
    gko::kernels::reference::hmis::ruge_stueben_first_pass(
        this->exec, strength.get(), strength_t.get(), cf_marker);

    // Verify: all points should be assigned (either C=1 or Z_PT=-2)
    auto marker = cf_marker.get_const_data();
    for (index_type i = 0; i < num_rows; ++i) {
        EXPECT_TRUE(marker[i] == 1 || marker[i] == -2)
            << "Point " << i << " has marker " << marker[i];
    }

    // At least one C-point should exist
    int c_count = 0;
    for (index_type i = 0; i < num_rows; ++i) {
        if (marker[i] == 1) c_count++;
    }
    EXPECT_GT(c_count, 0);

    // Not all should be C-points (coarsening should happen)
    EXPECT_LT(c_count, num_rows);
}


TYPED_TEST(HmisKernels, BuildCpointMap)
{
    using index_type = typename TestFixture::index_type;
    // cf_marker: C=1, F=0, C=1, F=0, C=1
    gko::array<index_type> cf_marker(this->exec, {1, 0, 1, 0, 1});
    gko::array<index_type> c_point_map(this->exec, 5);
    index_type num_c = 0;

    gko::kernels::reference::hmis::build_cpoint_map(this->exec, cf_marker,
                                                    &num_c, c_point_map);

    ASSERT_EQ(num_c, 3);
    auto map = c_point_map.get_const_data();
    EXPECT_EQ(map[0], 0);   // C-point -> coarse index 0
    EXPECT_EQ(map[1], -1);  // F-point
    EXPECT_EQ(map[2], 1);   // C-point -> coarse index 1
    EXPECT_EQ(map[3], -1);  // F-point
    EXPECT_EQ(map[4], 2);   // C-point -> coarse index 2
}


TYPED_TEST(HmisKernels, GenerateEndToEnd)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // Use a 7-point 1D Laplacian
    auto mtx = this->create_1d_laplacian(7);

    auto hmis_factory = MgLevel::build()
                            .with_strength_threshold(0.25)
                            .with_skip_sorting(true)
                            .on(this->exec);
    auto mg_level = hmis_factory->generate(gko::share(std::move(mtx)));

    // Verify basic properties
    auto prolong = mg_level->get_prolong_op();
    auto restrict = mg_level->get_restrict_op();
    auto coarse = mg_level->get_coarse_op();

    ASSERT_NE(prolong, nullptr);
    ASSERT_NE(restrict, nullptr);
    ASSERT_NE(coarse, nullptr);

    // Prolongation: fine_dim x coarse_dim
    EXPECT_EQ(prolong->get_size()[0], 7);
    EXPECT_GT(prolong->get_size()[1], 0);
    EXPECT_LT(prolong->get_size()[1], 7);

    // Restriction: coarse_dim x fine_dim
    EXPECT_EQ(restrict->get_size()[1], 7);
    EXPECT_EQ(restrict->get_size()[0], prolong->get_size()[1]);

    // Coarse matrix: coarse_dim x coarse_dim
    EXPECT_EQ(coarse->get_size()[0], prolong->get_size()[1]);
    EXPECT_EQ(coarse->get_size()[1], prolong->get_size()[1]);

    // CF marker should be set
    auto cf = mg_level->get_const_cf_marker();
    int c_count = 0;
    for (gko::size_type i = 0; i < 7; ++i) {
        EXPECT_TRUE(cf[i] == 0 || cf[i] == 1);
        if (cf[i] == 1) c_count++;
    }
    EXPECT_EQ(static_cast<gko::size_type>(c_count), prolong->get_size()[1]);
}


TYPED_TEST(HmisKernels, ProlongationCPointsAreInjection)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = this->create_1d_laplacian(7);

    auto hmis_factory = MgLevel::build()
                            .with_strength_threshold(0.25)
                            .with_skip_sorting(true)
                            .on(this->exec);
    auto mg_level = hmis_factory->generate(gko::share(std::move(mtx)));

    // Get P as CSR
    auto prolong_csr = gko::as<Mtx>(mg_level->get_prolong_op());
    auto cf = mg_level->get_const_cf_marker();
    auto p_rp = prolong_csr->get_const_row_ptrs();
    auto p_ci = prolong_csr->get_const_col_idxs();
    auto p_vals = prolong_csr->get_const_values();

    // For each C-point, P should have exactly one entry = 1.0
    for (gko::size_type i = 0; i < 7; ++i) {
        if (cf[i] == 1) {
            EXPECT_EQ(p_rp[i + 1] - p_rp[i], 1)
                << "C-point " << i << " should have exactly 1 entry in P";
            EXPECT_NEAR(static_cast<double>(gko::real(p_vals[p_rp[i]])), 1.0,
                        1e-14)
                << "C-point " << i << " injection should be 1.0";
        }
    }
}


}  // namespace
