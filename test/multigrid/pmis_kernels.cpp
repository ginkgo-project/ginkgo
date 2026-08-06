// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <fstream>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class Pmis : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using real_type = gko::remove_complex<value_type>;

    Pmis() : rand_engine(30) {}

    void initialize_data()
    {
#ifdef GINKGO_FAST_TESTS
        m = 129;
#else
        m = 597;
#endif
        auto system_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                m, m, std::uniform_int_distribution<>(10, m),
                std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::utils::make_diag_dominant(system_data);
        system_mtx = Csr::create(ref);
        system_mtx->read(system_data);

        d_system_mtx = gko::clone(exec, system_mtx);
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> system_mtx;
    std::shared_ptr<Csr> d_system_mtx;
    gko::size_type m;
};


TEST_F(Pmis, ComputeRowMaxAbsIsEquivalentToRef)
{
    initialize_data();
    gko::array<real_type> maxabs(ref, system_mtx->get_size()[0]);
    gko::array<real_type> d_maxabs(exec, d_system_mtx->get_size()[0]);

    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_row_maxabs(
        exec, d_system_mtx.get(), d_maxabs.get_data());

    GKO_ASSERT_ARRAY_NEAR(d_maxabs, maxabs, r<value_type>::value);
}


TEST_F(Pmis, ComputeStrongDepRowIsEquivalentToRef)
{
    initialize_data();
    gko::array<real_type> maxabs(ref, system_mtx->get_size()[0]);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::array<real_type> d_maxabs(exec, maxabs);
    gko::array<index_type> rows(ref, system_mtx->get_size()[0]);
    gko::array<index_type> d_rows(exec, d_system_mtx->get_size()[0]);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        rows.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep_row(
        exec, d_system_mtx.get(), d_maxabs.get_const_data(), real_type{0.25},
        d_rows.get_data());

    GKO_ASSERT_ARRAY_EQ(d_rows, rows);
}


TEST_F(Pmis, ComputeStrongDepIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> row_ptrs(ref, num + 1);
    gko::array<real_type> maxabs(ref, num);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, row_ptrs.get_data(), row_ptrs.get_size());
    gko::array<index_type> col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(col_idxs), std::move(row_ptrs));
    gko::array<index_type> d_rows(exec, d_system_mtx->get_size()[0]);
    gko::array<real_type> d_maxabs(exec, maxabs);
    auto d_strong_dep = gko::clone(exec, strong_dep);

    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep(
        exec, d_system_mtx.get(), d_maxabs.get_const_data(), real_type{0.25},
        d_strong_dep.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(d_strong_dep, strong_dep);
}


TEST_F(Pmis, CountIsEquivalentToRef)
{
    initialize_data();
    auto status = gko::test::generate_random_array<int>(
        m, std::uniform_int_distribution<>(-1, 1), rand_engine, ref);
    gko::array<int> d_status(exec, status);
    gko::size_type num;
    gko::size_type d_num;

    gko::kernels::reference::pmis::count(ref, m, status.get_const_data(), &num);
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::count(
        exec, m, d_status.get_const_data(), &d_num);

    ASSERT_EQ(d_num, num);
}


TEST_F(Pmis, InitializeWeightAndStatusIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> row_ptrs(ref, num + 1);
    gko::array<real_type> maxabs(ref, num);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, row_ptrs.get_data(), row_ptrs.get_size());
    gko::array<index_type> col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(col_idxs), std::move(row_ptrs));
    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    auto d_trans_strong_dep = gko::clone(exec, trans_strong_dep);

    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::array<real_type> d_weight(exec, num);
    gko::array<int> d_status(exec, num);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, trans_strong_dep.get(), weight.get_data(), status.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::initialize_weight_and_status(
        exec, d_trans_strong_dep.get(), d_weight.get_data(),
        d_status.get_data());

    GKO_ASSERT_ARRAY_EQ(d_status, status);
    for (int i = 0; i < m; i++) {
        ASSERT_EQ(
            std::floor(weight.get_const_data()[i]),
            std::floor(exec->copy_val_to_host(d_weight.get_const_data() + i)));
    }
}


TEST_F(Pmis, ClassifyIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> row_ptrs(ref, num + 1);
    gko::array<real_type> maxabs(ref, num);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, row_ptrs.get_data(), row_ptrs.get_size());
    gko::array<index_type> col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(col_idxs), std::move(row_ptrs));
    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, trans_strong_dep.get(), weight.get_data(), status.get_data());
    gko::array<real_type> d_weight(exec, weight);
    gko::array<int> d_status(exec, status);
    auto d_strong_dep = gko::clone(exec, strong_dep);
    auto d_trans_strong_dep = gko::clone(exec, trans_strong_dep);
    gko::array<int> new_status(ref, num);
    gko::array<int> d_new_status(exec, num);

    gko::kernels::reference::pmis::classify(
        ref, weight.get_data(), strong_dep.get(), trans_strong_dep.get(),
        status.get_const_data(), new_status.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::classify(
        exec, d_weight.get_data(), d_strong_dep.get(), d_trans_strong_dep.get(),
        d_status.get_const_data(), d_new_status.get_data());

    GKO_ASSERT_ARRAY_EQ(d_new_status, new_status);
}


TEST_F(Pmis, DirectInterpolationRowCountIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> row_ptrs(ref, num + 1);
    gko::array<real_type> maxabs(ref, num);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, row_ptrs.get_data(), row_ptrs.get_size());
    gko::array<index_type> col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(col_idxs), std::move(row_ptrs));
    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, trans_strong_dep.get(), weight.get_data(), status.get_data());
    gko::array<int> new_status(ref, num);
    auto status_ptr = status.get_data();
    auto new_status_ptr = new_status.get_data();
    gko::size_type num_not_assigned = 0;
    gko::kernels::reference::pmis::count(ref, num, status_ptr,
                                         &num_not_assigned);
    while (num_not_assigned != 0) {
        gko::kernels::reference::pmis::classify(
            ref, weight.get_data(), strong_dep.get(), trans_strong_dep.get(),
            status_ptr, new_status_ptr);
        gko::size_type new_num = 0;
        gko::kernels::reference::pmis::count(ref, num, new_status_ptr,
                                             &new_num);
        if (new_num == num_not_assigned) {
            // no progess -> throw error (maybe unneccessary)
            throw std::runtime_error("no progress in Pmis");
        }
        num_not_assigned = new_num;
        std::swap(new_status_ptr, status_ptr);
    }
    auto d_strong_dep = gko::clone(exec, strong_dep);
    gko::array<int> d_status(exec);
    if (status_ptr == status.get_data()) {
        d_status = status;
    } else {
        d_status = new_status;
    }
    gko::array<index_type> prolong_row_count(ref, num);
    gko::array<index_type> d_prolong_row_count(exec, num);

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        ref, strong_dep.get(), status_ptr, prolong_row_count.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::direct_interpolation_row_count(
        exec, d_strong_dep.get(), d_status.get_const_data(),
        d_prolong_row_count.get_data());

    GKO_ASSERT_ARRAY_EQ(d_prolong_row_count, prolong_row_count);
}


TEST_F(Pmis, DirectInterpolationFillIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> row_ptrs(ref, num + 1);
    gko::array<real_type> maxabs(ref, num);
    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      maxabs.get_data());
    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, row_ptrs.get_data(), row_ptrs.get_size());
    gko::array<index_type> col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(col_idxs), std::move(row_ptrs));
    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, trans_strong_dep.get(), weight.get_data(), status.get_data());
    gko::array<int> new_status(ref, num);
    auto status_ptr = status.get_data();
    auto new_status_ptr = new_status.get_data();
    gko::size_type num_not_assigned = 0;
    gko::kernels::reference::pmis::count(ref, num, status_ptr,
                                         &num_not_assigned);
    while (num_not_assigned != 0) {
        gko::kernels::reference::pmis::classify(
            ref, weight.get_data(), strong_dep.get(), trans_strong_dep.get(),
            status_ptr, new_status_ptr);
        gko::size_type new_num = 0;
        gko::kernels::reference::pmis::count(ref, num, new_status_ptr,
                                             &new_num);
        if (new_num == num_not_assigned) {
            // no progess -> throw error (maybe unneccessary)
            throw std::runtime_error("no progress in Pmis");
        }
        num_not_assigned = new_num;
        std::swap(new_status_ptr, status_ptr);
    }
    auto d_strong_dep = gko::clone(exec, strong_dep);
    gko::array<int> d_status(exec);
    if (status_ptr == status.get_data()) {
        d_status = status;
    } else {
        d_status = new_status;
    }
    gko::array<index_type> prolong_row_ptrs(ref, num + 1);
    gko::kernels::reference::pmis::direct_interpolation_row_count(
        ref, strong_dep.get(), status_ptr, prolong_row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, prolong_row_ptrs.get_data(), prolong_row_ptrs.get_size());
    gko::array<index_type> coarse_map(ref, num + 1);
    gko::kernels::reference::components::convert_precision(
        ref, num, status_ptr, coarse_map.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        ref, coarse_map.get_data(), coarse_map.get_size());
    auto prolong_nnz = prolong_row_ptrs.get_const_data()[num];
    gko::array<index_type> prolong_col_idxs(ref, prolong_nnz);
    gko::array<value_type> prolong_values(ref, prolong_nnz);
    gko::array<real_type> d_maxabs(exec, maxabs);
    gko::array<index_type> d_coarse_map(exec, coarse_map);
    gko::array<index_type> d_prolong_row_ptrs(exec, prolong_row_ptrs);
    gko::array<index_type> d_prolong_col_idxs(exec, prolong_nnz);
    gko::array<value_type> d_prolong_values(exec, prolong_nnz);

    gko::kernels::reference::pmis::direct_interpolation_fill(
        ref, system_mtx.get(), maxabs.get_const_data(), real_type{0.25},
        coarse_map.get_const_data(), prolong_row_ptrs.get_const_data(),
        prolong_col_idxs.get_data(), prolong_values.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::direct_interpolation_fill(
        exec, d_system_mtx.get(), d_maxabs.get_const_data(), real_type{0.25},
        d_coarse_map.get_const_data(), d_prolong_row_ptrs.get_const_data(),
        d_prolong_col_idxs.get_data(), d_prolong_values.get_data());

    GKO_ASSERT_ARRAY_EQ(d_prolong_col_idxs, prolong_col_idxs);
    GKO_ASSERT_ARRAY_NEAR(d_prolong_values, prolong_values,
                          r<value_type>::value);
}
