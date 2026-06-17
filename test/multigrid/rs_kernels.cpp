// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/rs_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/rs.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class Rs : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Rs() : rand_engine(30) {}

    void initialize_data()
    {
#ifdef GINKGO_FAST_TESTS
        m = 129;
#else
        m = 597;
#endif
        theta = 0.25;

        // 1. Generate a valid M-matrix
        auto m_matrix_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                m, m, std::uniform_int_distribution<>(5, 15),
                std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                          1.0),
                rand_engine);

        using real_type = gko::remove_complex<value_type>;
        for (auto& el : m_matrix_data.nonzeros) {
            if (el.row == el.column) {
                el.value = value_type{std::abs(el.value) +
                                      real_type{1.0}};  // Positive diagonal
            } else {
                el.value = value_type{
                    -std::abs(el.value)};  // Non-positive off-diagonal
            }
        }
        gko::utils::make_diag_dominant(m_matrix_data);

        m_matrix = Csr::create(ref);
        m_matrix->read(m_matrix_data);
        d_m_matrix = gko::clone(exec, m_matrix);

        // 2. Generate an invalid M-matrix (has a positive off-diagonal element)
        auto non_m_matrix_data = m_matrix_data;
        for (auto& el : non_m_matrix_data.nonzeros) {
            if (el.row != el.column) {
                el.value = value_type{std::abs(el.value) + real_type{1.0}};
                break;
            }
        }
        non_m_matrix = Csr::create(ref);
        non_m_matrix->read(non_m_matrix_data);
        d_non_m_matrix = gko::clone(exec, non_m_matrix);
    }

    std::default_random_engine rand_engine;

    gko::size_type m;
    double theta;

    std::shared_ptr<Csr> m_matrix;
    std::shared_ptr<Csr> d_m_matrix;
    std::shared_ptr<Csr> non_m_matrix;
    std::shared_ptr<Csr> d_non_m_matrix;
};


TEST_F(Rs, CheckMMatrixIsEquivalentToRef)
{
    initialize_data();
    gko::array<bool> is_m_ref(ref, 1);
    gko::array<bool> is_m_exec(exec, 1);

    // Test on a valid M-matrix
    gko::kernels::reference::rs::check_m_matrix(ref, m_matrix.get(), is_m_ref);
    gko::kernels::GKO_DEVICE_NAMESPACE::rs::check_m_matrix(
        exec, d_m_matrix.get(), is_m_exec);
    GKO_ASSERT_ARRAY_EQ(is_m_ref, is_m_exec);
    EXPECT_TRUE(is_m_ref.get_const_data()[0]);

    // Test on an invalid M-matrix
    gko::kernels::reference::rs::check_m_matrix(ref, non_m_matrix.get(),
                                                is_m_ref);
    gko::kernels::GKO_DEVICE_NAMESPACE::rs::check_m_matrix(
        exec, d_non_m_matrix.get(), is_m_exec);
    GKO_ASSERT_ARRAY_EQ(is_m_ref, is_m_exec);
    EXPECT_FALSE(is_m_ref.get_const_data()[0]);
}


TEST_F(Rs, ComputeSocAndRunRsIsEquivalentToRef)
{
    initialize_data();
    auto num_rows = m_matrix->get_size()[0];
    auto nnz = m_matrix->get_num_stored_elements();

    gko::array<bool> is_strong_ref(ref, nnz);
    gko::array<index_type> lambda_ref(ref, num_rows);
    gko::array<index_type> cf_marker_ref(ref, num_rows);
    index_type coarse_size_ref = 0;

    gko::array<bool> is_strong_exec(exec, nnz);
    gko::array<index_type> lambda_exec(exec, num_rows);
    gko::array<index_type> cf_marker_exec(exec, num_rows);
    index_type coarse_size_exec = 0;

    gko::kernels::reference::rs::compute_soc_and_run_rs(
        ref, m_matrix.get(), theta, is_strong_ref, lambda_ref, cf_marker_ref,
        coarse_size_ref);

    gko::kernels::GKO_DEVICE_NAMESPACE::rs::compute_soc_and_run_rs(
        exec, d_m_matrix.get(), theta, is_strong_exec, lambda_exec,
        cf_marker_exec, coarse_size_exec);

    GKO_ASSERT_ARRAY_EQ(is_strong_ref, is_strong_exec);
    GKO_ASSERT_ARRAY_EQ(lambda_ref, lambda_exec);
    GKO_ASSERT_ARRAY_EQ(cf_marker_ref, cf_marker_exec);
    EXPECT_EQ(coarse_size_ref, coarse_size_exec);
}


TEST_F(Rs, FillCoarseAndComputeProlongRowPtrsIsEquivalentToRef)
{
    initialize_data();
    auto num_rows = m_matrix->get_size()[0];
    auto nnz = m_matrix->get_num_stored_elements();

    gko::array<bool> is_strong_ref(ref, nnz);
    gko::array<index_type> lambda_ref(ref, num_rows);
    gko::array<index_type> cf_marker_ref(ref, num_rows);
    index_type coarse_size = 0;

    gko::kernels::reference::rs::compute_soc_and_run_rs(
        ref, m_matrix.get(), theta, is_strong_ref, lambda_ref, cf_marker_ref,
        coarse_size);

    gko::array<bool> is_strong_exec(exec, is_strong_ref);
    gko::array<index_type> cf_marker_exec(exec, cf_marker_ref);

    gko::array<index_type> coarse_rows_ref(ref, coarse_size);
    gko::array<index_type> fine_to_coarse_ref(ref, num_rows);
    gko::array<index_type> row_ptrs_ref(ref, num_rows + 1);

    gko::array<index_type> coarse_rows_exec(exec, coarse_size);
    gko::array<index_type> fine_to_coarse_exec(exec, num_rows);
    gko::array<index_type> row_ptrs_exec(exec, num_rows + 1);

    gko::kernels::reference::rs::fill_coarse_and_compute_prolong_row_ptrs(
        ref, cf_marker_ref, coarse_rows_ref, fine_to_coarse_ref, m_matrix.get(),
        is_strong_ref, row_ptrs_ref);

    gko::kernels::GKO_DEVICE_NAMESPACE::rs::
        fill_coarse_and_compute_prolong_row_ptrs(
            exec, cf_marker_exec, coarse_rows_exec, fine_to_coarse_exec,
            d_m_matrix.get(), is_strong_exec, row_ptrs_exec);

    GKO_ASSERT_ARRAY_EQ(coarse_rows_ref, coarse_rows_exec);
    GKO_ASSERT_ARRAY_EQ(fine_to_coarse_ref, fine_to_coarse_exec);
    GKO_ASSERT_ARRAY_EQ(row_ptrs_ref, row_ptrs_exec);
}


TEST_F(Rs, ComputeInterpolationIsEquivalentToRef)
{
    initialize_data();
    auto num_rows = m_matrix->get_size()[0];
    auto nnz = m_matrix->get_num_stored_elements();

    gko::array<bool> is_strong_ref(ref, nnz);
    gko::array<index_type> lambda_ref(ref, num_rows);
    gko::array<index_type> cf_marker_ref(ref, num_rows);
    index_type coarse_size = 0;

    gko::kernels::reference::rs::compute_soc_and_run_rs(
        ref, m_matrix.get(), theta, is_strong_ref, lambda_ref, cf_marker_ref,
        coarse_size);

    gko::array<index_type> coarse_rows_ref(ref, coarse_size);
    gko::array<index_type> fine_to_coarse_ref(ref, num_rows);
    gko::array<index_type> row_ptrs_ref(ref, num_rows + 1);

    gko::kernels::reference::rs::fill_coarse_and_compute_prolong_row_ptrs(
        ref, cf_marker_ref, coarse_rows_ref, fine_to_coarse_ref, m_matrix.get(),
        is_strong_ref, row_ptrs_ref);

    index_type p_nnz = row_ptrs_ref.get_const_data()[num_rows];

    auto P_ref = Csr::create(ref, gko::dim<2>(num_rows, coarse_size), p_nnz);
    std::copy_n(row_ptrs_ref.get_const_data(), num_rows + 1,
                P_ref->get_row_ptrs());

    auto P_exec = Csr::create(exec, gko::dim<2>(num_rows, coarse_size), p_nnz);
    gko::array<index_type> p_row_ptrs_exec(exec, row_ptrs_ref);
    exec->copy(num_rows + 1, p_row_ptrs_exec.get_const_data(),
               P_exec->get_row_ptrs());

    gko::array<bool> is_strong_exec(exec, is_strong_ref);
    gko::array<index_type> cf_marker_exec(exec, cf_marker_ref);
    gko::array<index_type> fine_to_coarse_exec(exec, fine_to_coarse_ref);

    gko::kernels::reference::rs::compute_interpolation(
        ref, m_matrix.get(), is_strong_ref.get_const_data(), cf_marker_ref,
        fine_to_coarse_ref.get_const_data(), P_ref.get());

    gko::kernels::GKO_DEVICE_NAMESPACE::rs::compute_interpolation(
        exec, d_m_matrix.get(), is_strong_exec.get_const_data(), cf_marker_exec,
        fine_to_coarse_exec.get_const_data(), P_exec.get());

    GKO_ASSERT_MTX_NEAR(P_ref, P_exec, r<value_type>::value);
}
