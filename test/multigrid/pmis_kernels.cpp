// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/multigrid/pmis_helpers.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class Pmis : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using real_type = gko::remove_complex<value_type>;

    Pmis()
        : rand_engine(30),
          row_ptrs(ref),
          maxabs(ref),
          col_idxs(ref),
          weight(ref),
          status(ref),
          new_status(ref),
          final_status(ref),
          prolong_row_ptrs(ref),
          coarse_map(ref)
    {}

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
        // the followings run almost whole pmis on reference.
        auto num = system_mtx->get_size()[0];
        row_ptrs.resize_and_reset(num + 1);
        maxabs.resize_and_reset(num);
        // compute_row_maxabs accumulates, so start from zero
        maxabs.fill(gko::zero<real_type>());
        gko::kernels::reference::pmis::compute_row_maxabs(
            ref, system_mtx.get(), true, maxabs.get_data());
        gko::kernels::reference::pmis::compute_strong_dep_row(
            ref, system_mtx.get(), true, maxabs.get_const_data(),
            real_type{0.25}, row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, row_ptrs.get_data(), row_ptrs.get_size());
        col_idxs.resize_and_reset(row_ptrs.get_const_data()[num]);
        strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
            ref, system_mtx->get_size(), col_idxs, row_ptrs);
        gko::kernels::reference::pmis::compute_strong_dep(
            ref, system_mtx.get(), true, maxabs.get_const_data(),
            real_type{0.25}, strong_dep.get());
        trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
        weight.resize_and_reset(num);
        status.resize_and_reset(num);
        gko::array<index_type> counts(ref, num);
        for (gko::size_type row = 0; row < num; row++) {
            counts.get_data()[row] =
                trans_strong_dep->get_const_row_ptrs()[row + 1] -
                trans_strong_dep->get_const_row_ptrs()[row];
        }
        gko::array<index_type> global_idx(ref, num);
        for (gko::size_type k = 0; k < num; k++) {
            global_idx.get_data()[k] = static_cast<index_type>(k);
        }
        gko::kernels::reference::pmis::initialize_weight_and_status(
            ref, num, counts.get_const_data(), global_idx.get_const_data(),
            weight.get_data(), status.get_data());
        new_status.resize_and_reset(num);
        auto status_ptr = status.get_data();
        auto new_status_ptr = new_status.get_data();
        gko::size_type num_not_assigned = 0;
        gko::kernels::reference::pmis::count(ref, num, status_ptr,
                                             &num_not_assigned);
        while (num_not_assigned != 0) {
            gko::multigrid::pmis::classify_round(
                ref, num,
                std::vector<sparsity_block>{
                    {strong_dep.get(), false, index_type{0}}},
                weight.get_const_data(), global_idx.get_const_data(),
                status_ptr, new_status_ptr, [] {});
            gko::size_type new_num = 0;
            gko::kernels::reference::pmis::count(ref, num, new_status_ptr,
                                                 &new_num);
            if (new_num == num_not_assigned) {
                // no progress -> throw error (maybe unnecessary)
                throw std::runtime_error("no progress in Pmis");
            }
            num_not_assigned = new_num;
            std::swap(new_status_ptr, status_ptr);
        }
        if (status_ptr == status.get_data()) {
            final_status = status;
        } else {
            final_status = new_status;
        }

        prolong_row_ptrs.resize_and_reset(num + 1);
        // the kernel accumulates and skips coarse rows, so seed them here
        for (gko::size_type row = 0; row < num; row++) {
            prolong_row_ptrs.get_data()[row] =
                status_ptr[row] == gko::kernels::pmis::coarse ? 1 : 0;
        }
        gko::kernels::reference::pmis::direct_interpolation_row_count(
            ref, index_type{0}, strong_dep.get(), status_ptr,
            prolong_row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, prolong_row_ptrs.get_data(), prolong_row_ptrs.get_size());
        coarse_map.resize_and_reset(num + 1);
        gko::kernels::reference::components::convert_precision(
            ref, num, status_ptr, coarse_map.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, coarse_map.get_data(), coarse_map.get_size());
    }


    using sparsity_block =
        gko::multigrid::pmis::column_block<SparsityCsr, index_type>;

    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> system_mtx;
    std::shared_ptr<Csr> d_system_mtx;
    gko::size_type m;
    gko::array<index_type> row_ptrs;
    gko::array<real_type> maxabs;
    gko::array<index_type> col_idxs;
    std::shared_ptr<SparsityCsr> strong_dep;
    std::shared_ptr<SparsityCsr> trans_strong_dep;
    gko::array<real_type> weight;
    gko::array<int> status;
    gko::array<int> new_status;
    gko::array<int> final_status;
    gko::array<index_type> prolong_row_ptrs;
    gko::array<index_type> coarse_map;
};


TEST_F(Pmis, ComputeRowMaxAbsIsEquivalentToRef)
{
    initialize_data();
    gko::array<real_type> maxabs(ref, system_mtx->get_size()[0]);
    gko::array<real_type> d_maxabs(exec, d_system_mtx->get_size()[0]);
    maxabs.fill(gko::zero<real_type>());
    d_maxabs.fill(gko::zero<real_type>());

    gko::kernels::reference::pmis::compute_row_maxabs(ref, system_mtx.get(),
                                                      true, maxabs.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_row_maxabs(
        exec, d_system_mtx.get(), true, d_maxabs.get_data());

    GKO_ASSERT_ARRAY_NEAR(d_maxabs, maxabs, r<value_type>::value);
}


TEST_F(Pmis, ComputeStrongDepRowIsEquivalentToRef)
{
    initialize_data();
    gko::array<real_type> d_maxabs(exec, maxabs);
    gko::array<index_type> rows(ref, system_mtx->get_size()[0]);
    gko::array<index_type> d_rows(exec, d_system_mtx->get_size()[0]);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), true, maxabs.get_const_data(), real_type{0.25},
        rows.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep_row(
        exec, d_system_mtx.get(), true, d_maxabs.get_const_data(),
        real_type{0.25}, d_rows.get_data());

    GKO_ASSERT_ARRAY_EQ(d_rows, rows);
}


TEST_F(Pmis, ComputeStrongDepIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<index_type> strong_col_idxs(ref, row_ptrs.get_const_data()[num]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        ref, system_mtx->get_size(), std::move(strong_col_idxs),
        std::move(row_ptrs));
    gko::array<real_type> d_maxabs(exec, maxabs);
    auto d_strong_dep = gko::clone(exec, strong_dep);

    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), true, maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep(
        exec, d_system_mtx.get(), true, d_maxabs.get_const_data(),
        real_type{0.25}, d_strong_dep.get());

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


TEST_F(Pmis, AddAtIndicesIsEquivalentToRef)
{
    // every backend implements this with its own atomic, and the kernel
    // exists for repeated indices: 10000 updates into 37 slots collide
    constexpr gko::size_type num = 10000;
    constexpr index_type range = 37;
    auto idxs = gko::test::generate_random_array<index_type>(
        num, std::uniform_int_distribution<index_type>(0, range - 1),
        rand_engine, ref);
    auto values = gko::test::generate_random_array<index_type>(
        num, std::uniform_int_distribution<index_type>(1, 100), rand_engine,
        ref);
    // pre-seeded, like the measure when the halo contributions arrive
    gko::array<index_type> out(ref, range);
    out.fill(index_type{3});
    gko::array<index_type> d_idxs(exec, idxs);
    gko::array<index_type> d_values(exec, values);
    gko::array<index_type> d_out(exec, out);

    gko::kernels::reference::pmis::add_at_indices(
        ref, num, idxs.get_const_data(), values.get_const_data(),
        out.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::add_at_indices(
        exec, num, d_idxs.get_const_data(), d_values.get_const_data(),
        d_out.get_data());

    GKO_ASSERT_ARRAY_EQ(d_out, out);
}


TEST_F(Pmis, AddAtIndicesWithNullValuesIsEquivalentToRef)
{
    // the null form, which adds one per index, is used by the column counts
    constexpr gko::size_type num = 10000;
    constexpr index_type range = 37;
    auto idxs = gko::test::generate_random_array<index_type>(
        num, std::uniform_int_distribution<index_type>(0, range - 1),
        rand_engine, ref);
    gko::array<index_type> out(ref, range);
    out.fill(index_type{3});
    gko::array<index_type> d_idxs(exec, idxs);
    gko::array<index_type> d_out(exec, out);
    const index_type* no_values = nullptr;

    gko::kernels::reference::pmis::add_at_indices(
        ref, num, idxs.get_const_data(), no_values, out.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::add_at_indices(
        exec, num, d_idxs.get_const_data(), no_values, d_out.get_data());

    GKO_ASSERT_ARRAY_EQ(d_out, out);
}


TEST_F(Pmis, CoarseGlobalIndexIsEquivalentToRef)
{
    initialize_data();
    const auto num = system_mtx->get_size()[0];
    gko::array<index_type> coarse_global(ref, num);
    gko::array<index_type> d_coarse_global(exec, num);
    gko::array<int> d_final_status(exec, final_status);
    gko::array<index_type> d_coarse_map(exec, coarse_map);

    gko::kernels::reference::pmis::coarse_global_index(
        ref, num, index_type{100}, final_status.get_const_data(),
        coarse_map.get_const_data(), coarse_global.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::coarse_global_index(
        exec, num, index_type{100}, d_final_status.get_const_data(),
        d_coarse_map.get_const_data(), d_coarse_global.get_data());

    GKO_ASSERT_ARRAY_EQ(d_coarse_global, coarse_global);
}


TEST_F(Pmis, InitializeWeightAndStatusIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    auto d_trans_strong_dep = gko::clone(exec, trans_strong_dep);
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::array<real_type> d_weight(exec, num);
    gko::array<int> d_status(exec, num);
    gko::array<index_type> counts(ref, num);
    for (gko::size_type row = 0; row < num; row++) {
        counts.get_data()[row] =
            trans_strong_dep->get_const_row_ptrs()[row + 1] -
            trans_strong_dep->get_const_row_ptrs()[row];
    }
    gko::array<index_type> global_idx(ref, num);
    for (gko::size_type k = 0; k < num; k++) {
        global_idx.get_data()[k] = static_cast<index_type>(k);
    }
    gko::array<index_type> d_counts(exec, counts);
    gko::array<index_type> d_global_idx(exec, global_idx);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, num, counts.get_const_data(), global_idx.get_const_data(),
        weight.get_data(), status.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::initialize_weight_and_status(
        exec, num, d_counts.get_const_data(), d_global_idx.get_const_data(),
        d_weight.get_data(), d_status.get_data());

    GKO_ASSERT_ARRAY_EQ(d_status, status);
    // the same hashed draw and count on both sides, so the results match
    // exactly
    GKO_ASSERT_ARRAY_EQ(d_weight, weight);
}


TEST_F(Pmis, ClassifyIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::array<index_type> counts(ref, num);
    for (gko::size_type row = 0; row < num; row++) {
        counts.get_data()[row] =
            trans_strong_dep->get_const_row_ptrs()[row + 1] -
            trans_strong_dep->get_const_row_ptrs()[row];
    }
    gko::array<index_type> global_idx(ref, num);
    for (gko::size_type k = 0; k < num; k++) {
        global_idx.get_data()[k] = static_cast<index_type>(k);
    }
    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, num, counts.get_const_data(), global_idx.get_const_data(),
        weight.get_data(), status.get_data());
    gko::array<real_type> d_weight(exec, weight);
    gko::array<int> d_status(exec, status);
    auto d_strong_dep = gko::clone(exec, strong_dep);
    gko::array<int> new_status(ref, num);
    gko::array<int> d_new_status(exec, num);
    gko::array<index_type> d_global_idx(exec, global_idx);

    gko::multigrid::pmis::classify_round(
        ref, num,
        std::vector<sparsity_block>{{strong_dep.get(), false, index_type{0}}},
        weight.get_const_data(), global_idx.get_const_data(),
        status.get_const_data(), new_status.get_data(), [] {});
    gko::multigrid::pmis::classify_round(
        exec, num,
        std::vector<sparsity_block>{{d_strong_dep.get(), false, index_type{0}}},
        d_weight.get_const_data(), d_global_idx.get_const_data(),
        d_status.get_const_data(), d_new_status.get_data(), [] {});

    GKO_ASSERT_ARRAY_EQ(d_new_status, new_status);
}


TEST_F(Pmis, WeightDrawIsInRangeAndMatchesRef)
{
    // with every in-degree zero the weight is the draw alone, which verifies
    // that it is a hash of the index rather than a per-backend generator
    constexpr gko::size_type num = 1000;
    gko::array<index_type> counts(ref, num);
    counts.fill(gko::zero<index_type>());
    gko::array<index_type> global_idx(ref, num);
    for (gko::size_type i = 0; i < num; i++) {
        global_idx.get_data()[i] = static_cast<index_type>(i);
    }
    gko::array<real_type> weight(ref, num);
    gko::array<int> status(ref, num);
    gko::array<index_type> d_counts(exec, counts);
    gko::array<index_type> d_global_idx(exec, global_idx);
    gko::array<real_type> d_weight(exec, num);
    gko::array<int> d_status(exec, num);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        ref, num, counts.get_const_data(), global_idx.get_const_data(),
        weight.get_data(), status.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::initialize_weight_and_status(
        exec, num, d_counts.get_const_data(), d_global_idx.get_const_data(),
        d_weight.get_data(), d_status.get_data());

    GKO_ASSERT_ARRAY_EQ(d_weight, weight);
    auto sum = 0.0;
    for (gko::size_type i = 0; i < num; i++) {
        const auto val = weight.get_const_data()[i];
        ASSERT_GE(val, gko::zero<real_type>());
        ASSERT_LT(val, gko::one<real_type>());
        sum += val;
    }
    // the mean is 0.495 with a standard error of about 0.009, so this only
    // rejects a degenerate generator
    ASSERT_NEAR(sum / num, 0.495, 0.1);
}


TEST_F(Pmis, DirectInterpolationFillSkipsRowWithoutStrongDependence)
{
    // row 1 stores an explicit zero off the diagonal, so its largest
    // off-diagonal magnitude is zero and it has no strong dependence, exactly
    // like the rows compute_strong_dep{,_row} skip.
    auto mtx =
        Csr::create(ref, gko::dim<2>{2, 2},
                    gko::array<value_type>(
                        ref, {value_type{1}, value_type{0}, value_type{1}}),
                    gko::array<index_type>(ref, {0, 0, 1}),
                    gko::array<index_type>(ref, {0, 1, 3}));
    auto d_mtx = gko::clone(exec, mtx);
    gko::array<real_type> d_row_maxabs(exec, {0, 0});
    // row 0 is coarse, row 1 is fine
    gko::array<index_type> d_coarse_map(exec, {0, 1, 1});
    // only the identity entry of the coarse row 0 is counted, row 1 gets none
    gko::array<index_type> d_prolong_row_ptrs(exec, {0, 1, 1});
    // the second slot is a canary: row 1 must not write anything, in
    // particular nothing past the end of its own (empty) range
    gko::array<index_type> d_prolong_col_idxs(exec, {0, -99});
    gko::array<value_type> d_prolong_values(exec,
                                            {value_type{0}, value_type{-99}});
    gko::array<index_type> expected_col_idxs(ref, {0, -99});
    gko::array<value_type> expected_values(ref,
                                           {value_type{1}, value_type{-99}});

    gko::array<int> d_status(
        exec, gko::array<int>(
                  ref, {gko::kernels::pmis::coarse, gko::kernels::pmis::fine}));
    gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
        exec, 2, {{d_mtx.get(), true, index_type{0}}},
        d_row_maxabs.get_const_data(), real_type{0.25},
        d_status.get_const_data(), d_prolong_row_ptrs.get_const_data(),
        d_prolong_col_idxs.get_data(), d_prolong_values.get_data());
    exec->run(gko::multigrid::pmis::make_gather(
        1, d_coarse_map.get_const_data(), d_prolong_col_idxs.get_const_data(),
        d_prolong_col_idxs.get_data()));

    GKO_ASSERT_ARRAY_EQ(d_prolong_col_idxs, expected_col_idxs);
    GKO_ASSERT_ARRAY_EQ(d_prolong_values, expected_values);
}


TEST_F(Pmis, DirectInterpolationRowCountIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    auto d_strong_dep = gko::clone(exec, strong_dep);
    gko::array<int> d_final_status(exec, final_status);
    gko::array<index_type> prolong_row_count(ref, num);
    gko::array<index_type> d_prolong_row_count(exec, num);
    // the kernel accumulates and skips coarse rows, so seed them here
    for (gko::size_type row = 0; row < num; row++) {
        prolong_row_count.get_data()[row] =
            final_status.get_const_data()[row] == gko::kernels::pmis::coarse
                ? 1
                : 0;
    }
    gko::array<index_type> d_seed(exec, prolong_row_count);
    d_prolong_row_count = d_seed;

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        ref, index_type{0}, strong_dep.get(), final_status.get_const_data(),
        prolong_row_count.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::direct_interpolation_row_count(
        exec, index_type{0}, d_strong_dep.get(),
        d_final_status.get_const_data(), d_prolong_row_count.get_data());

    GKO_ASSERT_ARRAY_EQ(d_prolong_row_count, prolong_row_count);
}


TEST_F(Pmis, DirectInterpolationFillIsEquivalentToRef)
{
    initialize_data();
    auto num = system_mtx->get_size()[0];
    gko::array<real_type> d_maxabs(exec, maxabs);
    gko::array<index_type> d_coarse_map(exec, coarse_map);
    gko::array<index_type> d_prolong_row_ptrs(exec, prolong_row_ptrs);
    auto prolong_nnz = prolong_row_ptrs.get_const_data()[num];
    gko::array<index_type> prolong_col_idxs(ref, prolong_nnz);
    gko::array<value_type> prolong_values(ref, prolong_nnz);
    gko::array<index_type> d_prolong_col_idxs(exec, prolong_nnz);
    gko::array<value_type> d_prolong_values(exec, prolong_nnz);

    gko::array<int> d_final_status(exec, final_status);
    const auto nnz = static_cast<gko::size_type>(prolong_nnz);

    gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
        ref, num, {{system_mtx.get(), true, index_type{0}}},
        maxabs.get_const_data(), real_type{0.25}, final_status.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data());
    ref->run(gko::multigrid::pmis::make_gather(
        nnz, coarse_map.get_const_data(), prolong_col_idxs.get_const_data(),
        prolong_col_idxs.get_data()));
    gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
        exec, num, {{d_system_mtx.get(), true, index_type{0}}},
        d_maxabs.get_const_data(), real_type{0.25},
        d_final_status.get_const_data(), d_prolong_row_ptrs.get_const_data(),
        d_prolong_col_idxs.get_data(), d_prolong_values.get_data());
    exec->run(gko::multigrid::pmis::make_gather(
        nnz, d_coarse_map.get_const_data(), d_prolong_col_idxs.get_const_data(),
        d_prolong_col_idxs.get_data()));

    GKO_ASSERT_ARRAY_EQ(d_prolong_col_idxs, prolong_col_idxs);
    GKO_ASSERT_ARRAY_NEAR(d_prolong_values, prolong_values,
                          r<value_type>::value);
}
