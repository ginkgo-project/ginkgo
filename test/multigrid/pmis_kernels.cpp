// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/array_access.hpp"
#include "core/components/gather_kernels.hpp"
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

    struct row_scales {
        gko::array<real_type> neg_scale;
        gko::array<real_type> row_sum;
        gko::array<real_type> diag_abs;
    };

    Pmis()
        : rand_engine(30),
          row_ptrs(ref),
          maxabs(ref),
          scales{gko::array<real_type>(ref), gko::array<real_type>(ref),
                 gko::array<real_type>(ref)},
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
        scales = compute_scales(ref, system_mtx.get());
        gko::kernels::reference::pmis::compute_strong_dep_row(
            ref, system_mtx.get(), true, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, row_ptrs.get_data(), row_ptrs.get_size());
        col_idxs.resize_and_reset(row_ptrs.get_const_data()[num]);
        ASSERT_GT(col_idxs.get_size(), 0);
        strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
            ref, system_mtx->get_size(), col_idxs, row_ptrs);
        gko::kernels::reference::pmis::compute_strong_dep(
            ref, system_mtx.get(), true, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, strong_dep.get());
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
        gko::kernels::reference::pmis::count_unassigned(ref, num, status_ptr,
                                                        &num_not_assigned);
        while (num_not_assigned != 0) {
            gko::multigrid::pmis::classify_round(
                ref, num,
                std::vector<sparsity_block>{
                    {strong_dep.get(), false, index_type{0}}},
                weight.get_const_data(), global_idx.get_const_data(),
                status_ptr, new_status_ptr, [] {});
            gko::size_type new_num = 0;
            gko::kernels::reference::pmis::count_unassigned(
                ref, num, new_status_ptr, &new_num);
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
        ASSERT_GT(coarse_map.get_const_data()[num], 0);
    }

    row_scales compute_scales(std::shared_ptr<const gko::Executor> e,
                              const Csr* mtx)
    {
        const auto num = mtx->get_size()[0];
        row_scales out{gko::array<real_type>(e, num),
                       gko::array<real_type>(e, num),
                       gko::array<real_type>(e, num)};
        out.neg_scale.fill(gko::zero<real_type>());
        out.row_sum.fill(gko::zero<real_type>());
        out.diag_abs.fill(gko::zero<real_type>());
        e->run(gko::multigrid::pmis::make_compute_row_scales(
            mtx, true, out.neg_scale.get_data(), out.row_sum.get_data(),
            out.diag_abs.get_data()));
        return out;
    }

    std::shared_ptr<SparsityCsr> strength_graph(
        std::shared_ptr<const gko::Executor> e, const Csr* mtx,
        real_type strength_threshold, real_type max_row_sum)
    {
        const auto num = mtx->get_size()[0];
        const auto s = compute_scales(e, mtx);
        gko::array<index_type> rows(e, num + 1);
        e->run(gko::multigrid::pmis::make_compute_strong_dep_row(
            mtx, true, s.neg_scale.get_const_data(), s.row_sum.get_const_data(),
            s.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            rows.get_data()));
        e->run(gko::multigrid::pmis::make_prefix_sum_nonnegative(
            rows.get_data(), rows.get_size()));
        const auto graph_nnz =
            static_cast<gko::size_type>(gko::get_element(rows, num));
        gko::array<index_type> cols(e, graph_nnz);
        cols.fill(index_type{-99});
        auto graph = SparsityCsr::create(e, mtx->get_size(), std::move(cols),
                                         std::move(rows));
        e->run(gko::multigrid::pmis::make_compute_strong_dep(
            mtx, true, s.neg_scale.get_const_data(), s.row_sum.get_const_data(),
            s.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            graph.get()));
        return graph;
    }

    void fill_prolongation(std::shared_ptr<const gko::Executor> e,
                           const Csr* mtx, real_type strength_threshold,
                           real_type max_row_sum, const int* status,
                           const index_type* coarse_map,
                           const index_type* prolong_row_ptrs,
                           gko::size_type nnz, index_type* prolong_col_idxs,
                           value_type* prolong_values)
    {
        const auto s = compute_scales(e, mtx);
        gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
            e, mtx->get_size()[0], {{mtx, true, index_type{0}}},
            s.neg_scale.get_const_data(), s.row_sum.get_const_data(),
            s.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            status, prolong_row_ptrs, prolong_col_idxs, prolong_values);
        e->run(gko::multigrid::pmis::make_gather(
            nnz, coarse_map, prolong_col_idxs, prolong_col_idxs));
    }

    std::shared_ptr<Csr> truncate_prolongation(
        std::shared_ptr<const gko::Executor> e, const Csr* mtx,
        gko::size_type p_max_elmts)
    {
        gko::array<index_type> t_row_ptrs(e);
        gko::array<index_type> t_col_idxs(e);
        gko::array<value_type> t_values(e);
        gko::multigrid::pmis::truncate_prolongation<value_type, index_type>(
            e, mtx->get_size()[0], p_max_elmts, mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), mtx->get_const_values(), t_row_ptrs,
            t_col_idxs, t_values);
        return Csr::create(e, mtx->get_size(), std::move(t_values),
                           std::move(t_col_idxs), std::move(t_row_ptrs));
    }


    using sparsity_block =
        gko::multigrid::pmis::column_block<SparsityCsr, index_type>;

    void check_classical_interpolation_matches_ref(
        const Csr* mtx, const gko::array<int>& split, real_type max_row_sum,
        std::unique_ptr<Csr>* out_ref_prolong = nullptr)
    {
        const auto num = mtx->get_size()[0];
        auto d_mtx = gko::clone(exec, mtx);

        gko::array<real_type> maxabs_ref(ref, num);
        maxabs_ref.fill(gko::zero<real_type>());
        gko::kernels::reference::pmis::compute_row_maxabs(
            ref, mtx, true, maxabs_ref.get_data());
        gko::array<real_type> d_maxabs(exec, num);
        d_maxabs.fill(gko::zero<real_type>());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_row_maxabs(
            exec, d_mtx.get(), true, d_maxabs.get_data());
        const auto s_ref = compute_scales(ref, mtx);
        const auto s_dev = compute_scales(exec, d_mtx.get());

        gko::array<index_type> map(ref, num + 1);
        gko::kernels::reference::components::convert_precision(
            ref, num, split.get_const_data(), map.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, map.get_data(), map.get_size());
        const auto num_coarse =
            static_cast<gko::size_type>(map.get_const_data()[num]);
        gko::array<int> d_split(exec, split);
        gko::array<index_type> d_map(exec, map);

        gko::array<index_type> p_row_ptrs(ref, num + 1);
        gko::array<index_type> d_p_row_ptrs(exec, num + 1);
        gko::kernels::reference::pmis::classical_interpolation_row_count(
            ref, mtx, s_ref.neg_scale.get_const_data(),
            s_ref.row_sum.get_const_data(), s_ref.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, split.get_const_data(),
            p_row_ptrs.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::
            classical_interpolation_row_count(
                exec, d_mtx.get(), s_dev.neg_scale.get_const_data(),
                s_dev.row_sum.get_const_data(), s_dev.diag_abs.get_const_data(),
                strength_threshold, max_row_sum, d_split.get_const_data(),
                d_p_row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, p_row_ptrs.get_data(), p_row_ptrs.get_size());
        gko::kernels::GKO_DEVICE_NAMESPACE::components::prefix_sum_nonnegative(
            exec, d_p_row_ptrs.get_data(), d_p_row_ptrs.get_size());
        GKO_ASSERT_ARRAY_EQ(d_p_row_ptrs, p_row_ptrs);

        const auto nnz =
            static_cast<gko::size_type>(p_row_ptrs.get_const_data()[num]);
        gko::array<index_type> p_col_idxs(ref, nnz);
        gko::array<value_type> p_values(ref, nnz);
        gko::array<index_type> d_p_col_idxs(exec, nnz);
        gko::array<value_type> d_p_values(exec, nnz);

        gko::kernels::reference::pmis::classical_interpolation_fill(
            ref, mtx, maxabs_ref.get_const_data(),
            s_ref.neg_scale.get_const_data(), s_ref.row_sum.get_const_data(),
            s_ref.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            split.get_const_data(), map.get_const_data(),
            p_row_ptrs.get_const_data(), p_col_idxs.get_data(),
            p_values.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::classical_interpolation_fill(
            exec, d_mtx.get(), d_maxabs.get_const_data(),
            s_dev.neg_scale.get_const_data(), s_dev.row_sum.get_const_data(),
            s_dev.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            d_split.get_const_data(), d_map.get_const_data(),
            d_p_row_ptrs.get_const_data(), d_p_col_idxs.get_data(),
            d_p_values.get_data());

        auto P =
            Csr::create(ref, gko::dim<2>{num, num_coarse}, std::move(p_values),
                        std::move(p_col_idxs), std::move(p_row_ptrs));
        auto d_P = Csr::create(exec, gko::dim<2>{num, num_coarse},
                               std::move(d_p_values), std::move(d_p_col_idxs),
                               std::move(d_p_row_ptrs));

        GKO_ASSERT_MTX_NEAR(d_P, P, r<value_type>::value * 100);
        gko::size_type interpolating_fine_rows = 0;
        const auto p_rows = P->get_const_row_ptrs();
        for (gko::size_type row = 0; row < num; row++) {
            if (split.get_const_data()[row] != gko::kernels::pmis::coarse &&
                p_rows[row + 1] > p_rows[row]) {
                interpolating_fine_rows++;
            }
        }
        ASSERT_GT(interpolating_fine_rows, 0);
        if (out_ref_prolong) {
            *out_ref_prolong = std::move(P);
        }
    }

    void check_ext_i_interpolation_matches_ref(
        const Csr* mtx, const gko::array<int>& split, real_type max_row_sum,
        std::unique_ptr<Csr>* out_dev_prolong = nullptr)
    {
        const auto num = mtx->get_size()[0];
        auto d_mtx = gko::clone(exec, mtx);

        gko::array<real_type> maxabs_ref(ref, num);
        maxabs_ref.fill(gko::zero<real_type>());
        gko::kernels::reference::pmis::compute_row_maxabs(
            ref, mtx, true, maxabs_ref.get_data());
        gko::array<real_type> d_maxabs(exec, num);
        d_maxabs.fill(gko::zero<real_type>());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_row_maxabs(
            exec, d_mtx.get(), true, d_maxabs.get_data());
        const auto s_ref = compute_scales(ref, mtx);
        const auto s_dev = compute_scales(exec, d_mtx.get());

        gko::array<index_type> map(ref, num + 1);
        gko::kernels::reference::components::convert_precision(
            ref, num, split.get_const_data(), map.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, map.get_data(), map.get_size());
        const auto num_coarse =
            static_cast<gko::size_type>(map.get_const_data()[num]);
        gko::array<int> d_split(exec, split);
        gko::array<index_type> d_map(exec, map);

        gko::array<index_type> p_row_ptrs(ref, num + 1);
        gko::array<index_type> d_p_row_ptrs(exec, num + 1);
        p_row_ptrs.fill(gko::zero<index_type>());
        d_p_row_ptrs.fill(gko::zero<index_type>());
        gko::kernels::reference::pmis::ext_i_interpolation_row_count(
            ref, mtx, s_ref.neg_scale.get_const_data(),
            s_ref.row_sum.get_const_data(), s_ref.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, split.get_const_data(),
            p_row_ptrs.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::ext_i_interpolation_row_count(
            exec, d_mtx.get(), s_dev.neg_scale.get_const_data(),
            s_dev.row_sum.get_const_data(), s_dev.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, d_split.get_const_data(),
            d_p_row_ptrs.get_data());
        GKO_ASSERT_ARRAY_EQ(d_p_row_ptrs, p_row_ptrs);
        gko::kernels::reference::components::prefix_sum_nonnegative(
            ref, p_row_ptrs.get_data(), p_row_ptrs.get_size());
        gko::kernels::GKO_DEVICE_NAMESPACE::components::prefix_sum_nonnegative(
            exec, d_p_row_ptrs.get_data(), d_p_row_ptrs.get_size());
        GKO_ASSERT_ARRAY_EQ(d_p_row_ptrs, p_row_ptrs);

        const auto nnz =
            static_cast<gko::size_type>(p_row_ptrs.get_const_data()[num]);
        gko::array<index_type> p_col_idxs(ref, nnz);
        gko::array<value_type> p_values(ref, nnz);
        gko::array<index_type> d_p_col_idxs(exec, nnz);
        gko::array<value_type> d_p_values(exec, nnz);
        p_col_idxs.fill(index_type{-99});
        d_p_col_idxs.fill(index_type{-99});

        gko::kernels::reference::pmis::ext_i_interpolation_fill(
            ref, mtx, maxabs_ref.get_const_data(),
            s_ref.neg_scale.get_const_data(), s_ref.row_sum.get_const_data(),
            s_ref.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            split.get_const_data(), map.get_const_data(),
            p_row_ptrs.get_const_data(), p_col_idxs.get_data(),
            p_values.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::ext_i_interpolation_fill(
            exec, d_mtx.get(), d_maxabs.get_const_data(),
            s_dev.neg_scale.get_const_data(), s_dev.row_sum.get_const_data(),
            s_dev.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            d_split.get_const_data(), d_map.get_const_data(),
            d_p_row_ptrs.get_const_data(), d_p_col_idxs.get_data(),
            d_p_values.get_data());

        GKO_ASSERT_ARRAY_EQ(d_p_col_idxs, p_col_idxs);
        GKO_ASSERT_ARRAY_NEAR(d_p_values, p_values, r<value_type>::value * 100);
        for (gko::size_type idx = 0; idx < nnz; idx++) {
            SCOPED_TRACE(idx);
            ASSERT_NE(p_col_idxs.get_const_data()[idx], index_type{-99});
        }
        gko::size_type interpolating_fine_rows = 0;
        for (gko::size_type row = 0; row < num; row++) {
            if (split.get_const_data()[row] != gko::kernels::pmis::coarse &&
                p_row_ptrs.get_const_data()[row + 1] >
                    p_row_ptrs.get_const_data()[row]) {
                interpolating_fine_rows++;
            }
        }
        ASSERT_GT(interpolating_fine_rows, 0);

        if (out_dev_prolong) {
            auto d_P = Csr::create(
                exec, gko::dim<2>{num, num_coarse}, std::move(d_p_values),
                std::move(d_p_col_idxs), std::move(d_p_row_ptrs));
            *out_dev_prolong = gko::clone(ref, d_P);
        }
    }

    struct device_row_counts {
        gko::array<index_type> ext_i;
        gko::array<index_type> classical;
    };

    device_row_counts device_row_counts_for(const Csr* mtx,
                                            const gko::array<int>& split,
                                            real_type max_row_sum)
    {
        const auto num = mtx->get_size()[0];
        auto d_mtx = gko::clone(exec, mtx);
        const auto s_dev = compute_scales(exec, d_mtx.get());
        gko::array<int> d_split(exec, split);
        gko::array<index_type> d_ext_i(exec, num);
        gko::array<index_type> d_classical(exec, num);
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::ext_i_interpolation_row_count(
            exec, d_mtx.get(), s_dev.neg_scale.get_const_data(),
            s_dev.row_sum.get_const_data(), s_dev.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, d_split.get_const_data(),
            d_ext_i.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::pmis::
            classical_interpolation_row_count(
                exec, d_mtx.get(), s_dev.neg_scale.get_const_data(),
                s_dev.row_sum.get_const_data(), s_dev.diag_abs.get_const_data(),
                strength_threshold, max_row_sum, d_split.get_const_data(),
                d_classical.get_data());
        return {gko::array<index_type>(ref, d_ext_i),
                gko::array<index_type>(ref, d_classical)};
    }

    value_type matrix_entry(const Csr* mtx, index_type row,
                            index_type col) const
    {
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto col_idxs = mtx->get_const_col_idxs();
        const auto values = mtx->get_const_values();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (col_idxs[idx] == col) {
                return values[idx];
            }
        }
        return gko::zero<value_type>();
    }

    std::shared_ptr<Csr> ext_i_path_mtx()
    {
        auto mtx = Csr::create(ref);
        mtx->read({{4, 4},
                   {{0, 0, value_type{1}},
                    {0, 1, value_type{-1}},
                    {1, 0, value_type{-1}},
                    {1, 1, value_type{2}},
                    {1, 2, value_type{-1}},
                    {2, 1, value_type{-1}},
                    {2, 2, value_type{2}},
                    {2, 3, value_type{-1}},
                    {3, 2, value_type{-1}},
                    {3, 3, value_type{1}}}});
        return mtx;
    }

    std::shared_ptr<Csr> sign_rule_mtx(gko::size_type num,
                                       gko::size_type stripe)
    {
        gko::matrix_data<value_type, index_type> data{gko::dim<2>{num, num}};
        for (gko::size_type i = 0; i < num; i++) {
            const auto row = static_cast<index_type>(i);
            if (i > 0) {
                data.nonzeros.emplace_back(row, row - 1, value_type{-2});
            }
            data.nonzeros.emplace_back(row, row, value_type{4});
            if (i + 1 < num) {
                data.nonzeros.emplace_back(row, row + 1,
                                           is_striped_row(i, num, stripe)
                                               ? value_type{2.5}
                                               : value_type{1});
            }
        }
        data.sort_row_major();
        auto mtx = Csr::create(ref);
        mtx->read(data);
        return mtx;
    }

    static bool is_striped_row(gko::size_type i, gko::size_type num,
                               gko::size_type stripe)
    {
        return i > 0 && i + 1 < num && i % stripe == 2;
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> system_mtx;
    std::shared_ptr<Csr> d_system_mtx;
    gko::size_type m;
    gko::array<index_type> row_ptrs;
    gko::array<real_type> maxabs;
    row_scales scales;
    gko::array<index_type> col_idxs;
    std::shared_ptr<SparsityCsr> strong_dep;
    std::shared_ptr<SparsityCsr> trans_strong_dep;
    gko::array<real_type> weight;
    gko::array<int> status;
    gko::array<int> new_status;
    gko::array<int> final_status;
    gko::array<index_type> prolong_row_ptrs;
    gko::array<index_type> coarse_map;
    const real_type strength_threshold{0.25};
    const real_type max_row_sum{0.9};
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
    const auto d_scales = compute_scales(exec, d_system_mtx.get());
    gko::array<index_type> rows(ref, system_mtx->get_size()[0]);
    gko::array<index_type> d_rows(exec, d_system_mtx->get_size()[0]);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        ref, system_mtx.get(), true, scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        strength_threshold, max_row_sum, rows.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep_row(
        exec, d_system_mtx.get(), true, d_scales.neg_scale.get_const_data(),
        d_scales.row_sum.get_const_data(), d_scales.diag_abs.get_const_data(),
        strength_threshold, max_row_sum, d_rows.get_data());

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
    const auto d_scales = compute_scales(exec, d_system_mtx.get());
    auto d_strong_dep = gko::clone(exec, strong_dep);

    gko::kernels::reference::pmis::compute_strong_dep(
        ref, system_mtx.get(), true, scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        strength_threshold, max_row_sum, strong_dep.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::compute_strong_dep(
        exec, d_system_mtx.get(), true, d_scales.neg_scale.get_const_data(),
        d_scales.row_sum.get_const_data(), d_scales.diag_abs.get_const_data(),
        strength_threshold, max_row_sum, d_strong_dep.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(d_strong_dep, strong_dep);
}


TEST_F(Pmis, CountUnassignedIsEquivalentToRef)
{
    initialize_data();
    auto status = gko::test::generate_random_array<int>(
        m, std::uniform_int_distribution<>(-1, 1), rand_engine, ref);
    gko::array<int> d_status(exec, status);
    gko::size_type num;
    gko::size_type d_num;

    gko::kernels::reference::pmis::count_unassigned(
        ref, m, status.get_const_data(), &num);
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::count_unassigned(
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
    // the scale is an exact power of two, so the backends agree bit for bit
    // whether or not they fuse the multiply-add
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
    // the draw is uniform on [0, 1) scaled by 0.5, so the mean is 0.25; this
    // only rejects a degenerate generator
    ASSERT_NEAR(sum / num, 0.25, 0.1);
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
    fill_prolongation(exec, d_mtx.get(), real_type{0.25}, real_type{1.0},
                      d_status.get_const_data(), d_coarse_map.get_const_data(),
                      d_prolong_row_ptrs.get_const_data(), 1,
                      d_prolong_col_idxs.get_data(),
                      d_prolong_values.get_data());

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
    gko::array<index_type> d_coarse_map(exec, coarse_map);
    gko::array<index_type> d_prolong_row_ptrs(exec, prolong_row_ptrs);
    auto prolong_nnz = prolong_row_ptrs.get_const_data()[num];
    gko::array<index_type> prolong_col_idxs(ref, prolong_nnz);
    gko::array<value_type> prolong_values(ref, prolong_nnz);
    gko::array<index_type> d_prolong_col_idxs(exec, prolong_nnz);
    gko::array<value_type> d_prolong_values(exec, prolong_nnz);

    gko::array<int> d_final_status(exec, final_status);
    const auto nnz = static_cast<gko::size_type>(prolong_nnz);
    fill_prolongation(ref, system_mtx.get(), strength_threshold, max_row_sum,
                      final_status.get_const_data(),
                      coarse_map.get_const_data(),
                      prolong_row_ptrs.get_const_data(), nnz,
                      prolong_col_idxs.get_data(), prolong_values.get_data());
    fill_prolongation(
        exec, d_system_mtx.get(), strength_threshold, max_row_sum,
        d_final_status.get_const_data(), d_coarse_map.get_const_data(),
        d_prolong_row_ptrs.get_const_data(), nnz, d_prolong_col_idxs.get_data(),
        d_prolong_values.get_data());

    GKO_ASSERT_ARRAY_EQ(d_prolong_col_idxs, prolong_col_idxs);
    GKO_ASSERT_ARRAY_NEAR(d_prolong_values, prolong_values,
                          r<value_type>::value);
}


TEST_F(Pmis, ClassicalInterpolationIsEquivalentToRef)
{
    constexpr gko::size_type num = 60;
    auto system_data =
        gko::test::generate_random_matrix_data<value_type, index_type>(
            num, num, std::uniform_int_distribution<>(3, 10),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
    gko::utils::make_diag_dominant(system_data);
    auto mtx = Csr::create(ref);
    mtx->read(system_data);

    auto split = gko::test::generate_random_array<int>(
        num,
        std::uniform_int_distribution<>(gko::kernels::pmis::fine,
                                        gko::kernels::pmis::coarse),
        rand_engine, ref);
    bool has_coarse = false;
    for (gko::size_type i = 0; i < num; i++) {
        has_coarse |= split.get_const_data()[i] == gko::kernels::pmis::coarse;
    }
    if (!has_coarse) {
        split.get_data()[0] = gko::kernels::pmis::coarse;
    }

    check_classical_interpolation_matches_ref(mtx.get(), split, max_row_sum);
}


TEST_F(Pmis, ClassicalInterpolationWithGatedRowIsEquivalentToRef)
{
    constexpr gko::size_type num = 60;
    constexpr index_type gated_row = 7;
    auto system_data =
        gko::test::generate_random_matrix_data<value_type, index_type>(
            num, num, std::uniform_int_distribution<>(3, 10),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
    gko::utils::make_diag_dominant(system_data);
    system_data.nonzeros.erase(
        std::remove_if(
            system_data.nonzeros.begin(), system_data.nonzeros.end(),
            [gated_row](const auto& entry) { return entry.row == gated_row; }),
        system_data.nonzeros.end());
    system_data.nonzeros.emplace_back(gated_row, gated_row, value_type{1e-4});
    system_data.nonzeros.emplace_back(gated_row, index_type{0}, value_type{-1});
    system_data.nonzeros.emplace_back(gated_row, index_type{1}, value_type{-1});
    system_data.nonzeros.emplace_back(gated_row, index_type{3}, value_type{-1});
    system_data.nonzeros.emplace_back(gated_row, index_type{4},
                                      value_type{0.5});
    for (index_type col :
         {index_type{5}, index_type{6}, index_type{8}, index_type{9}}) {
        system_data.nonzeros.emplace_back(gated_row, col, value_type{-0.125});
    }
    system_data.sort_row_major();
    auto mtx = Csr::create(ref);
    mtx->read(system_data);

    auto split = gko::test::generate_random_array<int>(
        num,
        std::uniform_int_distribution<>(gko::kernels::pmis::fine,
                                        gko::kernels::pmis::coarse),
        rand_engine, ref);
    split.get_data()[0] = gko::kernels::pmis::coarse;
    split.get_data()[1] = gko::kernels::pmis::coarse;
    split.get_data()[4] = gko::kernels::pmis::coarse;
    split.get_data()[3] = gko::kernels::pmis::fine;
    split.get_data()[gated_row] = gko::kernels::pmis::fine;

    std::unique_ptr<Csr> P;
    check_classical_interpolation_matches_ref(mtx.get(), split, real_type{1.0},
                                              &P);

    const auto p_row_ptrs = P->get_const_row_ptrs();
    const auto p_col_idxs = P->get_const_col_idxs();
    const auto p_values = P->get_const_values();
    ASSERT_EQ(p_row_ptrs[gated_row + 1] - p_row_ptrs[gated_row], 2);
    for (auto idx = p_row_ptrs[gated_row]; idx < p_row_ptrs[gated_row + 1];
         idx++) {
        SCOPED_TRACE(idx);
        EXPECT_GE(p_col_idxs[idx], index_type{0});
        EXPECT_LT(p_col_idxs[idx], static_cast<index_type>(P->get_size()[1]));
        EXPECT_EQ(p_values[idx], gko::zero<value_type>());
    }
}


TEST_F(Pmis, ClassicalInterpolationWithSameSignCoarseNeighbourIsEquivalentToRef)
{
    gko::matrix_data<value_type, index_type> data{gko::dim<2>{4, 4}};
    data.nonzeros.emplace_back(0, 0, value_type{4});
    data.nonzeros.emplace_back(1, 0, value_type{-1});
    data.nonzeros.emplace_back(1, 1, value_type{-4});
    data.nonzeros.emplace_back(1, 2, value_type{2});
    data.nonzeros.emplace_back(1, 3, value_type{2.5});
    data.nonzeros.emplace_back(2, 2, value_type{4});
    data.nonzeros.emplace_back(3, 3, value_type{4});
    data.sort_row_major();
    auto mtx = Csr::create(ref);
    mtx->read(data);
    gko::array<int> split(ref,
                          {gko::kernels::pmis::coarse, gko::kernels::pmis::fine,
                           gko::kernels::pmis::fine, gko::kernels::pmis::fine});

    std::unique_ptr<Csr> P;
    check_classical_interpolation_matches_ref(mtx.get(), split, max_row_sum,
                                              &P);

    const auto p_row_ptrs = P->get_const_row_ptrs();
    const auto p_col_idxs = P->get_const_col_idxs();
    const auto p_values = P->get_const_values();
    ASSERT_EQ(p_row_ptrs[1] - p_row_ptrs[0], index_type{1});
    EXPECT_EQ(p_col_idxs[p_row_ptrs[0]], index_type{0});
    EXPECT_EQ(p_values[p_row_ptrs[0]], gko::one<value_type>());
    ASSERT_EQ(p_row_ptrs[2] - p_row_ptrs[1], index_type{1});
    EXPECT_EQ(p_col_idxs[p_row_ptrs[1]], index_type{0});
    EXPECT_EQ(p_values[p_row_ptrs[1]], gko::zero<value_type>());
}


TEST_F(Pmis, ComputeRowScalesIsEquivalentToRef)
{
    initialize_data();

    const auto d_scales = compute_scales(exec, d_system_mtx.get());

    GKO_ASSERT_ARRAY_EQ(d_scales.neg_scale, scales.neg_scale);
    GKO_ASSERT_ARRAY_EQ(d_scales.diag_abs, scales.diag_abs);
    GKO_ASSERT_ARRAY_NEAR(d_scales.row_sum, scales.row_sum,
                          r<value_type>::value * 100);
}


TEST_F(Pmis, PositiveOffDiagonalIsNeverStrongIsEquivalentToRef)
{
    constexpr gko::size_type num = 137;
    constexpr gko::size_type stripe = 5;
    auto mtx = sign_rule_mtx(num, stripe);
    auto d_mtx = gko::clone(exec, mtx);

    auto graph =
        strength_graph(ref, mtx.get(), strength_threshold, real_type{1.0});
    auto d_graph =
        strength_graph(exec, d_mtx.get(), strength_threshold, real_type{1.0});

    GKO_ASSERT_MTX_EQ_SPARSITY(d_graph, graph);
    ASSERT_EQ(graph->get_num_nonzeros(), num - 1);
    const auto row_ptrs = graph->get_const_row_ptrs();
    const auto col_idxs = graph->get_const_col_idxs();
    EXPECT_EQ(row_ptrs[1] - row_ptrs[0], index_type{0});
    for (gko::size_type row = 1; row < num; row++) {
        SCOPED_TRACE(row);
        ASSERT_EQ(row_ptrs[row + 1] - row_ptrs[row], index_type{1});
        EXPECT_EQ(col_idxs[row_ptrs[row]], static_cast<index_type>(row) - 1);
    }
}


TEST_F(Pmis, MaxRowSumFilterIsEquivalentToRef)
{
    constexpr gko::size_type num = 137;
    constexpr gko::size_type stripe = 5;
    auto mtx = sign_rule_mtx(num, stripe);
    auto d_mtx = gko::clone(exec, mtx);

    auto filtered =
        strength_graph(ref, mtx.get(), strength_threshold, real_type{0.9});
    auto d_filtered =
        strength_graph(exec, d_mtx.get(), strength_threshold, real_type{0.9});
    auto unfiltered =
        strength_graph(ref, mtx.get(), strength_threshold, real_type{1.0});

    GKO_ASSERT_MTX_EQ_SPARSITY(d_filtered, filtered);
    gko::size_type striped = 0;
    for (gko::size_type row = 0; row < num; row++) {
        striped += is_striped_row(row, num, stripe) ? 1 : 0;
    }
    ASSERT_GT(striped, 0);
    ASSERT_EQ(unfiltered->get_num_nonzeros(), num - 1);
    ASSERT_EQ(filtered->get_num_nonzeros(), num - 1 - striped);
    ASSERT_GT(filtered->get_num_nonzeros(), 0);

    const auto f_rows = filtered->get_const_row_ptrs();
    const auto u_rows = unfiltered->get_const_row_ptrs();
    for (gko::size_type row = 0; row < num; row++) {
        SCOPED_TRACE(row);
        const index_type expected =
            (row == 0 || is_striped_row(row, num, stripe)) ? 0 : 1;
        EXPECT_EQ(f_rows[row + 1] - f_rows[row], expected);
        if (is_striped_row(row, num, stripe)) {
            EXPECT_EQ(u_rows[row + 1] - u_rows[row], index_type{1});
        }
    }
}


TEST_F(Pmis, ExtIExtendedSetIsStrictlyLargerThanStrongC)
{
    auto mtx = ext_i_path_mtx();
    gko::array<int> split(
        ref, {gko::kernels::pmis::coarse, gko::kernels::pmis::fine,
              gko::kernels::pmis::fine, gko::kernels::pmis::coarse});

    const auto counts = device_row_counts_for(mtx.get(), split, max_row_sum);

    const auto ext_i = counts.ext_i.get_const_data();
    const auto classical = counts.classical.get_const_data();
    EXPECT_EQ(classical[0], index_type{1});
    EXPECT_EQ(ext_i[0], index_type{1});
    EXPECT_EQ(classical[1], index_type{1});
    EXPECT_EQ(ext_i[1], index_type{2});
    EXPECT_EQ(classical[2], index_type{1});
    EXPECT_EQ(ext_i[2], index_type{2});
    EXPECT_EQ(classical[3], index_type{1});
    EXPECT_EQ(ext_i[3], index_type{1});
    EXPECT_EQ(matrix_entry(mtx.get(), index_type{1}, index_type{3}),
              gko::zero<value_type>());

    std::unique_ptr<Csr> d_P;
    check_ext_i_interpolation_matches_ref(mtx.get(), split, max_row_sum, &d_P);
    const auto row_ptrs = d_P->get_const_row_ptrs();
    const auto col_idxs = d_P->get_const_col_idxs();
    const auto values = d_P->get_const_values();
    const auto tol = static_cast<double>(r<value_type>::value) * 50;
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], index_type{1});
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1]])), 2.0 / 3.0,
                tol);
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1] + 1])),
                1.0 / 3.0, tol);
}


TEST_F(Pmis, ExtIInterpolationIsEquivalentToRef)
{
    constexpr gko::size_type num = 60;
    auto system_data =
        gko::test::generate_random_matrix_data<value_type, index_type>(
            num, num, std::uniform_int_distribution<>(3, 10),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
    gko::utils::make_diag_dominant(system_data);
    auto mtx = Csr::create(ref);
    mtx->read(system_data);

    auto split = gko::test::generate_random_array<int>(
        num,
        std::uniform_int_distribution<>(gko::kernels::pmis::fine,
                                        gko::kernels::pmis::coarse),
        rand_engine, ref);
    bool has_coarse = false;
    for (gko::size_type i = 0; i < num; i++) {
        has_coarse |= split.get_const_data()[i] == gko::kernels::pmis::coarse;
    }
    if (!has_coarse) {
        split.get_data()[0] = gko::kernels::pmis::coarse;
    }

    const auto counts = device_row_counts_for(mtx.get(), split, max_row_sum);
    gko::size_type extended_rows = 0;
    for (gko::size_type row = 0; row < num; row++) {
        SCOPED_TRACE(row);
        const auto diff = counts.ext_i.get_const_data()[row] -
                          counts.classical.get_const_data()[row];
        ASSERT_GE(diff, index_type{0});
        if (diff > index_type{0}) {
            extended_rows++;
        }
    }
    ASSERT_GT(extended_rows, 0);

    check_ext_i_interpolation_matches_ref(mtx.get(), split, max_row_sum);
}


TEST_F(Pmis, ExtIInterpolationWithZeroSigmaKIsEquivalentToRef)
{
    gko::matrix_data<value_type, index_type> data{gko::dim<2>{5, 5}};
    data.nonzeros.emplace_back(0, 0, value_type{1});
    data.nonzeros.emplace_back(0, 1, value_type{-1});
    data.nonzeros.emplace_back(1, 0, value_type{-1});
    data.nonzeros.emplace_back(1, 1, value_type{2});
    data.nonzeros.emplace_back(1, 2, value_type{-1});
    data.nonzeros.emplace_back(2, 1, value_type{1});
    data.nonzeros.emplace_back(2, 2, value_type{2});
    data.nonzeros.emplace_back(2, 4, value_type{-3});
    data.nonzeros.emplace_back(3, 3, value_type{1});
    data.nonzeros.emplace_back(3, 4, value_type{-1});
    data.nonzeros.emplace_back(4, 2, value_type{-1});
    data.nonzeros.emplace_back(4, 3, value_type{-1});
    data.nonzeros.emplace_back(4, 4, value_type{2});
    data.sort_row_major();
    auto mtx = Csr::create(ref);
    mtx->read(data);
    gko::array<int> split(ref,
                          {gko::kernels::pmis::coarse, gko::kernels::pmis::fine,
                           gko::kernels::pmis::fine, gko::kernels::pmis::coarse,
                           gko::kernels::pmis::fine});

    std::unique_ptr<Csr> d_P;
    check_ext_i_interpolation_matches_ref(mtx.get(), split, max_row_sum, &d_P);

    const auto row_ptrs = d_P->get_const_row_ptrs();
    const auto col_idxs = d_P->get_const_col_idxs();
    const auto values = d_P->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], index_type{1});
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{0});
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1]])), 1.0,
                static_cast<double>(r<value_type>::value) * 50);
}


TEST_F(Pmis, ExtIInterpolationWithCollapsingDivisorIsEquivalentToRef)
{
    gko::matrix_data<value_type, index_type> data{gko::dim<2>{4, 4}};
    data.nonzeros.emplace_back(0, 0, value_type{1});
    data.nonzeros.emplace_back(0, 1, value_type{-1});
    data.nonzeros.emplace_back(1, 0, value_type{-1});
    data.nonzeros.emplace_back(1, 1, value_type{1});
    data.nonzeros.emplace_back(1, 2, value_type{-1});
    data.nonzeros.emplace_back(2, 1, value_type{-1});
    data.nonzeros.emplace_back(2, 2, value_type{2});
    data.nonzeros.emplace_back(2, 3, value_type{-1});
    data.nonzeros.emplace_back(3, 2, value_type{-1});
    data.nonzeros.emplace_back(3, 3, value_type{1});
    data.sort_row_major();
    auto mtx = Csr::create(ref);
    mtx->read(data);
    gko::array<int> split(ref,
                          {gko::kernels::pmis::coarse, gko::kernels::pmis::fine,
                           gko::kernels::pmis::fine, gko::kernels::pmis::fine});

    std::unique_ptr<Csr> d_P;
    check_ext_i_interpolation_matches_ref(mtx.get(), split, real_type{1.0},
                                          &d_P);

    const auto row_ptrs = d_P->get_const_row_ptrs();
    const auto col_idxs = d_P->get_const_col_idxs();
    const auto values = d_P->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], index_type{1});
    const auto idx = row_ptrs[1];
    EXPECT_EQ(col_idxs[idx], index_type{0});
    EXPECT_TRUE(gko::is_finite(values[idx])) << "entry " << idx;
    EXPECT_EQ(values[idx], gko::zero<value_type>());
}


TEST_F(Pmis, ExtIInterpolationWithNegativeNeighbourDiagonalIsEquivalentToRef)
{
    gko::matrix_data<value_type, index_type> data{gko::dim<2>{4, 4}};
    data.nonzeros.emplace_back(0, 0, value_type{1});
    data.nonzeros.emplace_back(0, 1, value_type{-1});
    data.nonzeros.emplace_back(1, 0, value_type{-1});
    data.nonzeros.emplace_back(1, 1, value_type{3});
    data.nonzeros.emplace_back(1, 2, value_type{-1});
    data.nonzeros.emplace_back(1, 3, value_type{-1});
    data.nonzeros.emplace_back(2, 0, value_type{2});
    data.nonzeros.emplace_back(2, 1, value_type{1});
    data.nonzeros.emplace_back(2, 2, value_type{-4});
    data.nonzeros.emplace_back(2, 3, value_type{1});
    data.nonzeros.emplace_back(3, 1, value_type{-1});
    data.nonzeros.emplace_back(3, 3, value_type{1});
    data.sort_row_major();
    auto mtx = Csr::create(ref);
    mtx->read(data);
    gko::array<int> split(
        ref, {gko::kernels::pmis::coarse, gko::kernels::pmis::fine,
              gko::kernels::pmis::fine, gko::kernels::pmis::coarse});

    std::unique_ptr<Csr> d_P;
    check_ext_i_interpolation_matches_ref(mtx.get(), split, max_row_sum, &d_P);

    const auto row_ptrs = d_P->get_const_row_ptrs();
    const auto col_idxs = d_P->get_const_col_idxs();
    const auto values = d_P->get_const_values();
    const auto tol = static_cast<double>(r<value_type>::value) * 50;
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], index_type{1});
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1]])), 6.0 / 11.0,
                tol);
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1] + 1])),
                5.0 / 11.0, tol);
}


class PmisTruncateInterpolation : public Pmis {
protected:
    std::shared_ptr<Csr> wide_mtx()
    {
        auto mtx = Csr::create(ref);
        mtx->read({{3, 6},
                   {{0, 0, value_type{4}},
                    {0, 5, value_type{-2}},
                    {1, 1, value_type{3}},
                    {1, 2, value_type{-7}},
                    {1, 3, value_type{1}},
                    {2, 0, value_type{5}},
                    {2, 1, value_type{-1}},
                    {2, 2, value_type{4}},
                    {2, 3, value_type{-8}},
                    {2, 4, value_type{2}}}});
        return mtx;
    }
};


TEST_F(PmisTruncateInterpolation, RowCountIsEquivalentToRefAndExact)
{
    auto mtx = wide_mtx();
    auto d_mtx = gko::clone(exec, mtx);
    const gko::size_type p_max_elmts = 3;
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> row_counts(ref, num_rows + 1);
    gko::array<index_type> d_row_counts(exec, num_rows + 1);
    row_counts.fill(gko::zero<index_type>());
    d_row_counts.fill(gko::zero<index_type>());

    gko::kernels::reference::pmis::truncate_interpolation_row_count(
        ref, num_rows, p_max_elmts, mtx->get_const_row_ptrs(),
        row_counts.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::pmis::truncate_interpolation_row_count(
        exec, num_rows, p_max_elmts, d_mtx->get_const_row_ptrs(),
        d_row_counts.get_data());

    GKO_ASSERT_ARRAY_EQ(d_row_counts, row_counts);
    gko::array<index_type> expected(ref, {2, 3, 3});
    for (gko::size_type row = 0; row < num_rows; row++) {
        SCOPED_TRACE(row);
        ASSERT_EQ(row_counts.get_const_data()[row],
                  expected.get_const_data()[row]);
    }
}


TEST_F(PmisTruncateInterpolation, FillLeavesNarrowAndBoundaryRowsUntouched)
{
    auto mtx = wide_mtx();
    auto d_mtx = gko::clone(exec, mtx);
    const gko::size_type p_max_elmts = 3;

    auto out = truncate_prolongation(ref, mtx.get(), p_max_elmts);
    auto d_out = truncate_prolongation(exec, d_mtx.get(), p_max_elmts);
    auto d_out_on_ref = gko::clone(ref, d_out);

    GKO_ASSERT_MTX_NEAR(d_out_on_ref, out, 0);

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{0});
    EXPECT_EQ(values[row_ptrs[0]], value_type{4});
    EXPECT_EQ(col_idxs[row_ptrs[0] + 1], index_type{5});
    EXPECT_EQ(values[row_ptrs[0] + 1], value_type{-2});
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], index_type{3});
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{1});
    EXPECT_EQ(values[row_ptrs[1]], value_type{3});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], index_type{2});
    EXPECT_EQ(values[row_ptrs[1] + 1], value_type{-7});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 2], index_type{3});
    EXPECT_EQ(values[row_ptrs[1] + 2], value_type{1});
}


TEST_F(PmisTruncateInterpolation, FillKeepsLargestMagnitudeEntriesAndRescales)
{
    auto mtx = wide_mtx();
    auto d_mtx = gko::clone(exec, mtx);
    const gko::size_type p_max_elmts = 3;

    auto out = truncate_prolongation(ref, mtx.get(), p_max_elmts);
    auto d_out = truncate_prolongation(exec, d_mtx.get(), p_max_elmts);
    auto d_out_on_ref = gko::clone(ref, d_out);

    GKO_ASSERT_MTX_NEAR(d_out_on_ref, out, 0);

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[3] - row_ptrs[2], index_type{3});
    EXPECT_EQ(col_idxs[row_ptrs[2]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[2] + 1], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[2] + 2], index_type{3});
    EXPECT_EQ(values[row_ptrs[2]], value_type{10});
    EXPECT_EQ(values[row_ptrs[2] + 1], value_type{8});
    EXPECT_EQ(values[row_ptrs[2] + 2], value_type{-16});
    auto rescaled_sum =
        values[row_ptrs[2]] + values[row_ptrs[2] + 1] + values[row_ptrs[2] + 2];
    EXPECT_EQ(rescaled_sum, value_type{2});
}


TEST_F(PmisTruncateInterpolation, FillLeavesZeroSumKeptEntriesUnscaled)
{
    auto mtx = Csr::create(ref);
    mtx->read({{1, 3},
               {{0, 0, value_type{5}},
                {0, 1, value_type{-5}},
                {0, 2, value_type{2}}}});
    auto d_mtx = gko::clone(exec, mtx);
    const gko::size_type p_max_elmts = 2;

    auto out = truncate_prolongation(ref, mtx.get(), p_max_elmts);
    auto d_out = truncate_prolongation(exec, d_mtx.get(), p_max_elmts);
    auto d_out_on_ref = gko::clone(ref, d_out);

    GKO_ASSERT_MTX_NEAR(d_out_on_ref, out, 0);

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[0] + 1], index_type{1});
    EXPECT_EQ(values[row_ptrs[0]], value_type{5});
    EXPECT_EQ(values[row_ptrs[0] + 1], value_type{-5});
    EXPECT_TRUE(gko::is_finite(values[row_ptrs[0]]));
    EXPECT_TRUE(gko::is_finite(values[row_ptrs[0] + 1]));

    const auto d_values = d_out_on_ref->get_const_values();
    EXPECT_TRUE(gko::is_finite(d_values[0]));
    EXPECT_TRUE(gko::is_finite(d_values[1]));
}
