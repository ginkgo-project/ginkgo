// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <array>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/stop/combined.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/multigrid/pmis_helpers.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"

constexpr auto c = gko::kernels::pmis::coarse;
constexpr auto f = gko::kernels::pmis::fine;
constexpr auto u = gko::kernels::pmis::unassigned;


template <typename ValueType, typename IndexType>
struct pmis_row_scales {
    gko::array<gko::remove_complex<ValueType>> neg_scale;
    gko::array<gko::remove_complex<ValueType>> row_sum;
    gko::array<gko::remove_complex<ValueType>> diag_abs;
};


template <typename ValueType, typename IndexType>
pmis_row_scales<ValueType, IndexType> compute_pmis_row_scales(
    std::shared_ptr<const gko::ReferenceExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* mtx)
{
    using real_type = gko::remove_complex<ValueType>;
    const auto num_rows = mtx->get_size()[0];
    pmis_row_scales<ValueType, IndexType> scales{
        gko::array<real_type>(exec, num_rows),
        gko::array<real_type>(exec, num_rows),
        gko::array<real_type>(exec, num_rows)};
    scales.neg_scale.fill(gko::zero<real_type>());
    scales.row_sum.fill(gko::zero<real_type>());
    scales.diag_abs.fill(gko::zero<real_type>());
    gko::kernels::reference::pmis::compute_row_scales(
        exec, mtx, true, scales.neg_scale.get_data(), scales.row_sum.get_data(),
        scales.diag_abs.get_data());
    return scales;
}

template <typename ValueIndexType>
class Pmis : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;

    Pmis()
        : exec(gko::ReferenceExecutor::create()),
          mtx{Mtx::create(this->exec), Mtx::create(this->exec)},
          row_maxabs{gko::array<real_type>(this->exec, {4, 0, 8, 6}),
                     gko::array<real_type>(this->exec, {1, 1, 5, 1, 0})},
          dep_row_ptrs{gko::array<index_type>(this->exec, {0, 3, 3, 4, 5}),
                       gko::array<index_type>(this->exec, {0, 2, 3, 4, 6, 6})},
          dep_col_idxs{gko::array<index_type>(this->exec, {1, 2, 3, 1, 2}),
                       gko::array<index_type>(this->exec, {2, 4, 4, 1, 2, 4})},
          expected_status{gko::array<int>(this->exec, {f, u, u, u}),
                          gko::array<int>(this->exec, {f, u, u, f, u})},
          floor_weight{gko::array<real_type>(this->exec, {0, 2, 2, 1}),
                       gko::array<real_type>(this->exec, {0, 1, 2, 0, 3})},
          prolong_op1(Mtx::create(this->exec)),
          coarse_op1(Mtx::create(this->exec))
    {
        /**
         * 4  -1 4 2
         *     3
         * -1 -8 1
         * 1     6 3
         */
        mtx.at(0)->read({{4, 4},
                         {{0, 0, value_type{4}},
                          {0, 1, value_type{-1}},
                          {0, 2, value_type{4}},
                          {0, 3, value_type{2}},
                          {1, 1, value_type{3}},
                          {2, 0, value_type{-1}},
                          {2, 1, value_type{-8}},
                          {2, 2, value_type{1}},
                          {3, 0, value_type{1}},
                          {3, 2, value_type{6}},
                          {3, 3, value_type{3}}}});
        /**
         * 4    1   1
         *   4      1
         *   5  3   1
         *     -1 2 1
         *          1
         */
        mtx.at(1)->read({{5, 5},
                         {{0, 0, value_type{4}},
                          {0, 2, value_type{1}},
                          {0, 4, value_type{1}},
                          {1, 1, value_type{4}},
                          {1, 4, value_type{1}},
                          {2, 1, value_type{5}},
                          {2, 2, value_type{3}},
                          {2, 4, value_type{1}},
                          {3, 2, value_type{-1}},
                          {3, 3, value_type{2}},
                          {3, 4, value_type{1}},
                          {4, 4, value_type{1}}}});
        // we only have the following for mtx.at(1).
        // For mtx.at(0), we have same weight before randomization, so there is
        // no deterministic result
        prolong_op1->read(
            {{5, 1}, {{2, 0, value_type{1}}, {3, 0, value_type{0.5}}}});
        coarse_op1->read({{1, 1}, {{0, 0, value_type{3}}}});
    }

    // one selection round over a single block, as core/multigrid/pmis.cpp
    // drives it
    void classify(const real_type* weight, const index_type* global_idx,
                  const SparsityCsr* strong_dep, const int* status,
                  int* new_status)
    {
        gko::multigrid::pmis::classify_round(
            exec, strong_dep->get_size()[0],
            std::vector<
                gko::multigrid::pmis::column_block<SparsityCsr, index_type>>{
                {strong_dep, false, index_type{0}}},
            weight, global_idx, status, new_status, [] {});
    }

    void fill_prolongation(const Mtx* mtx, real_type strength_threshold,
                           real_type max_row_sum, const int* status,
                           const index_type* coarse_map,
                           const index_type* prolong_row_ptrs,
                           index_type* prolong_col_idxs,
                           value_type* prolong_values)
    {
        const auto num_rows = mtx->get_size()[0];
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
            exec, num_rows, {{mtx, true, index_type{0}}},
            scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
            scales.diag_abs.get_const_data(), strength_threshold, max_row_sum,
            status, prolong_row_ptrs, prolong_col_idxs, prolong_values);
        const auto nnz =
            static_cast<gko::size_type>(prolong_row_ptrs[num_rows]);
        exec->run(gko::multigrid::pmis::make_gather(
            nnz, coarse_map, prolong_col_idxs, prolong_col_idxs));
    }

    std::shared_ptr<Mtx> sign_rule_mtx()
    {
        auto mtx = Mtx::create(exec);
        mtx->read({{4, 4},
                   {{0, 0, value_type{4}},
                    {0, 1, value_type{3}},
                    {0, 2, value_type{-4}},
                    {1, 1, value_type{4}},
                    {1, 2, value_type{3}},
                    {1, 3, value_type{2}},
                    {2, 2, value_type{4}},
                    {2, 3, value_type{-0.1}},
                    {3, 0, value_type{-2}},
                    {3, 1, value_type{1.5}},
                    {3, 2, value_type{-0.1}},
                    {3, 3, value_type{4}}}});
        return mtx;
    }

    std::shared_ptr<SparsityCsr> strength_graph(const Mtx* mtx,
                                                real_type strength_threshold,
                                                real_type max_row_sum)
    {
        const auto num_rows = mtx->get_size()[0];
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::array<index_type> rows(exec, num_rows + 1);
        gko::kernels::reference::pmis::compute_strong_dep_row(
            exec, mtx, true, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, rows.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            exec, rows.get_data(), rows.get_size());
        const auto nnz =
            static_cast<gko::size_type>(rows.get_const_data()[num_rows]);
        gko::array<index_type> cols(exec, nnz);
        cols.fill(index_type{-99});
        auto graph = SparsityCsr::create(exec, mtx->get_size(), std::move(cols),
                                         std::move(rows));
        gko::kernels::reference::pmis::compute_strong_dep(
            exec, mtx, true, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            strength_threshold, max_row_sum, graph.get());
        return graph;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::array<std::shared_ptr<Mtx>, 2> mtx;
    std::array<gko::array<real_type>, 2> row_maxabs;
    std::array<gko::array<index_type>, 2> dep_row_ptrs;
    std::array<gko::array<index_type>, 2> dep_col_idxs;
    std::array<gko::array<int>, 2> expected_status;
    std::array<gko::array<real_type>, 2> floor_weight;
    std::shared_ptr<Mtx> prolong_op1;
    std::shared_ptr<Mtx> coarse_op1;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Pmis, ComputeRowMaxAbs)
{
    using real_type = typename TestFixture::real_type;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        gko::array<real_type> maxabs(this->exec,
                                     this->mtx.at(i)->get_size()[0]);
        maxabs.fill(gko::zero<real_type>());

        gko::kernels::reference::pmis::compute_row_maxabs(
            this->exec, this->mtx.at(i).get(), true, maxabs.get_data());

        GKO_ASSERT_ARRAY_EQ(maxabs, this->row_maxabs.at(i));
    }
}


TYPED_TEST(Pmis, ComputeRowMaxAbsWithoutDiagonalKeepsMatchingIndex)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    // an off-diagonal block: column index 0 in row 0 is a remote node, not a
    // diagonal entry, so it must NOT be skipped
    auto off_diag = Mtx::create(
        this->exec, gko::dim<2>{2, 1},
        gko::array<value_type>(this->exec, {value_type{7}, value_type{3}}),
        gko::array<index_type>(this->exec, {0, 0}),
        gko::array<index_type>(this->exec, {0, 1, 2}));
    gko::array<real_type> maxabs(this->exec, {0, 0});
    gko::array<real_type> expected(this->exec, {7, 3});

    gko::kernels::reference::pmis::compute_row_maxabs(
        this->exec, off_diag.get(), false, maxabs.get_data());

    GKO_ASSERT_ARRAY_EQ(maxabs, expected);
}


TYPED_TEST(Pmis, ComputeStrongDepRow)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    std::array<gko::array<index_type>, 2> expected{
        gko::array<index_type>(this->exec, {0, 1, 1, 2, 2}),
        gko::array<index_type>(this->exec, {0, 0, 0, 0, 1, 1})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        const auto num_rows = this->mtx.at(i)->get_size()[0];
        const auto scales =
            compute_pmis_row_scales(this->exec, this->mtx.at(i).get());
        gko::array<index_type> sparsity_rows(this->exec, num_rows + 1);

        gko::kernels::reference::pmis::compute_strong_dep_row(
            this->exec, this->mtx.at(i).get(), true,
            scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
            scales.diag_abs.get_const_data(), real_type{0.25}, real_type{1.0},
            sparsity_rows.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

        GKO_ASSERT_ARRAY_EQ(sparsity_rows, expected.at(i));
    }
}


TYPED_TEST(Pmis, ComputeStrongDep)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    std::array<gko::array<index_type>, 2> expected_rows{
        gko::array<index_type>(this->exec, {0, 1, 1, 2, 2}),
        gko::array<index_type>(this->exec, {0, 0, 0, 0, 1, 1})};
    std::array<gko::array<index_type>, 2> expected_cols{
        gko::array<index_type>(this->exec, {1, 1}),
        gko::array<index_type>(this->exec, {2})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep_ans =
            gko::matrix::SparsityCsr<value_type, index_type>::create(
                this->exec, this->mtx.at(i)->get_size(),
                std::move(expected_cols.at(i)), std::move(expected_rows.at(i)));

        auto strong_dep = this->strength_graph(this->mtx.at(i).get(),
                                               real_type{0.25}, real_type{1.0});

        GKO_ASSERT_MTX_EQ_SPARSITY(strong_dep, strong_dep_ans);
    }
}


TYPED_TEST(Pmis, PositiveOffDiagonalIsNeverStrong)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto mtx = this->sign_rule_mtx();

    auto graph =
        this->strength_graph(mtx.get(), real_type{0.25}, real_type{0.9});

    const auto row_ptrs = graph->get_const_row_ptrs();
    const auto col_idxs = graph->get_const_col_idxs();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], index_type{1});
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{2});
}


TYPED_TEST(Pmis, RowWithoutNegativeOffDiagonalHasNoStrongDep)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto mtx = this->sign_rule_mtx();

    auto graph =
        this->strength_graph(mtx.get(), real_type{0.25}, real_type{1.0});

    const auto row_ptrs = graph->get_const_row_ptrs();
    EXPECT_EQ(row_ptrs[2] - row_ptrs[1], index_type{0});
}


TYPED_TEST(Pmis, MaxRowSumStripsNearlyFullySummedRow)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto mtx = this->sign_rule_mtx();

    auto filtered =
        this->strength_graph(mtx.get(), real_type{0.25}, real_type{0.9});
    auto unfiltered =
        this->strength_graph(mtx.get(), real_type{0.25}, real_type{1.0});

    const auto filtered_rows = filtered->get_const_row_ptrs();
    EXPECT_EQ(filtered_rows[3] - filtered_rows[2], index_type{0});
    const auto unfiltered_rows = unfiltered->get_const_row_ptrs();
    const auto unfiltered_cols = unfiltered->get_const_col_idxs();
    ASSERT_EQ(unfiltered_rows[3] - unfiltered_rows[2], index_type{1});
    EXPECT_EQ(unfiltered_cols[unfiltered_rows[2]], index_type{3});
}


TYPED_TEST(Pmis, KeepsStrongDepsWhenNeitherFilterFires)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto mtx = this->sign_rule_mtx();

    auto graph =
        this->strength_graph(mtx.get(), real_type{0.25}, real_type{0.9});

    const auto row_ptrs = graph->get_const_row_ptrs();
    const auto col_idxs = graph->get_const_col_idxs();
    ASSERT_EQ(row_ptrs[4] - row_ptrs[3], index_type{1});
    EXPECT_EQ(col_idxs[row_ptrs[3]], index_type{0});
}


TYPED_TEST(Pmis, InitializeWeightAndStatus)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
        auto num_row = this->mtx.at(i)->get_size()[0];
        gko::array<real_type> weight(this->exec, num_row);
        gko::array<int> status(this->exec, num_row);
        gko::array<index_type> counts(this->exec, num_row);
        for (gko::size_type row = 0; row < num_row; row++) {
            counts.get_data()[row] =
                trans_strong_dep->get_const_row_ptrs()[row + 1] -
                trans_strong_dep->get_const_row_ptrs()[row];
        }
        gko::array<index_type> global_idx(this->exec, num_row);
        for (gko::size_type row = 0; row < num_row; row++) {
            global_idx.get_data()[row] = static_cast<index_type>(row);
        }

        gko::kernels::reference::pmis::initialize_weight_and_status(
            this->exec, num_row, counts.get_const_data(),
            global_idx.get_const_data(), weight.get_data(), status.get_data());

        GKO_ASSERT_ARRAY_EQ(status, this->expected_status.at(i));
        for (int row = 0; row < num_row; row++) {
            auto val = weight.get_const_data()[row];
            auto ans = this->floor_weight.at(i).get_const_data()[row];
            ASSERT_GE(val, ans);
            ASSERT_LE(val, ans + 1);
        }
    }
}


TYPED_TEST(Pmis, Classify)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    std::array<gko::array<real_type>, 2> weight{
        gko::array<real_type>(this->exec, {0.1, 2.2, 2.1, 1.2}),
        gko::array<real_type>(this->exec, {0.0, 1.0, 2.0, 0.0, 3.0})};
    std::array<gko::array<int>, 3> status_ans{
        gko::array<int>(this->exec, {f, c, f, u}),
        gko::array<int>(this->exec, {f, c, f, c}),
        gko::array<int>(this->exec, {f, f, c, f, c})};
    using index_type = typename TestFixture::index_type;
    std::array<int, 2> required_step{2, 1};
    int status_idx = 0;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        auto new_status = this->expected_status.at(i);
        for (int step = 0; step < required_step.at(i); step++) {
            SCOPED_TRACE(step);
            auto status = new_status;
            gko::array<index_type> global_idx(this->exec,
                                              new_status.get_size());
            for (gko::size_type k = 0; k < global_idx.get_size(); k++) {
                global_idx.get_data()[k] = static_cast<index_type>(k);
            }
            this->classify(weight.at(i).get_const_data(),
                           global_idx.get_const_data(), strong_dep.get(),
                           status.get_const_data(), new_status.get_data());

            GKO_ASSERT_ARRAY_EQ(new_status, status_ans.at(status_idx));
            status_idx++;
        }
    }
}


TYPED_TEST(Pmis, ClassifySelectWithColOffsetReadsHaloNeighbours)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    // one local row, one halo neighbour at column 0 of the off-diagonal
    // strength block. The halo node has the higher weight, so the local row
    // has to be downgraded.
    gko::array<index_type> row_ptrs(this->exec, {0, 1});
    gko::array<index_type> col_idxs(this->exec, {0});
    auto s_offd = SparsityCsr::create(this->exec, gko::dim<2>{1, 1},
                                      std::move(col_idxs), std::move(row_ptrs));
    // extended arrays: index 0 is the local node, index 1 the halo node
    gko::array<real_type> weight(this->exec, {1, 2});
    gko::array<index_type> global_idx(this->exec, {0, 1});
    gko::array<int> status(this->exec, {u, u});
    gko::array<int> new_status(this->exec, {c, u});
    gko::array<int> expected(this->exec, {u, u});

    gko::kernels::reference::pmis::classify_select(
        this->exec, weight.get_const_data(), global_idx.get_const_data(),
        index_type{1}, s_offd.get(), status.get_const_data(),
        new_status.get_data());

    GKO_ASSERT_ARRAY_EQ(new_status, expected);
}


TYPED_TEST(Pmis, ClassifyOnSameWeight)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    using index_type = typename TestFixture::index_type;
    // when weight + rand are still the same, we use index to compare
    gko::array<real_type> weight(this->exec, {0.1, 2.0, 2.0, 1.2});
    auto strong_dep =
        SparsityCsr::create(this->exec, this->mtx.at(0)->get_size(),
                            std::move(this->dep_col_idxs.at(0)),
                            std::move(this->dep_row_ptrs.at(0)));
    gko::array<int> status_ans(this->exec, {f, c, c, f});
    auto new_status = this->expected_status.at(0);
    auto status = new_status;

    gko::array<index_type> global_idx(this->exec, new_status.get_size());
    for (gko::size_type k = 0; k < global_idx.get_size(); k++) {
        global_idx.get_data()[k] = static_cast<index_type>(k);
    }
    this->classify(weight.get_const_data(), global_idx.get_const_data(),
                   strong_dep.get(), status.get_const_data(),
                   new_status.get_data());

    GKO_ASSERT_ARRAY_EQ(new_status, status_ans);
}


TYPED_TEST(Pmis, CountUnassigned)
{
    gko::array<int> arr(this->exec, 5);
    auto data = arr.get_data();
    data[0] = u;
    data[1] = f;
    data[2] = u;
    data[3] = c;
    data[4] = u;

    gko::size_type num = 0;
    gko::kernels::reference::pmis::count_unassigned(this->exec, 5,
                                                    arr.get_const_data(), &num);

    EXPECT_EQ(num, 3);
}


TYPED_TEST(Pmis, AddAtIndicesAccumulatesRepeatedIndices)
{
    using index_type = typename TestFixture::index_type;
    // index 1 appears three times and index 3 twice: the kernel has to add,
    // not assign
    gko::array<index_type> idxs(this->exec, {1, 3, 1, 0, 3, 1});
    gko::array<index_type> values(this->exec, {10, 20, 30, 40, 50, 60});
    // pre-seeded, like the measure when the halo contributions arrive
    gko::array<index_type> out(this->exec, {7, 0, 5, 0});
    gko::array<index_type> expected(this->exec, {47, 100, 5, 70});

    gko::kernels::reference::pmis::add_at_indices(
        this->exec, idxs.get_size(), idxs.get_const_data(),
        values.get_const_data(), out.get_data());

    GKO_ASSERT_ARRAY_EQ(out, expected);
}


TYPED_TEST(Pmis, AddAtIndicesCountsOccurrencesWhenValuesAreNull)
{
    using index_type = typename TestFixture::index_type;
    // the null form adds one per index, a histogram of idxs onto out
    gko::array<index_type> idxs(this->exec, {1, 3, 1, 0, 3, 1});
    gko::array<index_type> out(this->exec, {7, 0, 5, 0});
    gko::array<index_type> expected(this->exec, {8, 3, 5, 2});
    const index_type* no_values = nullptr;

    gko::kernels::reference::pmis::add_at_indices(this->exec, idxs.get_size(),
                                                  idxs.get_const_data(),
                                                  no_values, out.get_data());

    GKO_ASSERT_ARRAY_EQ(out, expected);
}


TYPED_TEST(Pmis, CoarseGlobalIndexMarksFinePointsInvalid)
{
    using index_type = typename TestFixture::index_type;
    // coarse_map is the exclusive prefix sum of "is coarse", so nodes 1 and 2
    // are at coarse indices 0 and 1
    gko::array<int> status(this->exec, {f, c, c, f});
    gko::array<index_type> coarse_map(this->exec, {0, 0, 1, 2, 2});
    gko::array<index_type> coarse_global(this->exec, 4);
    // a fine node gets a sentinel that nothing may read back
    gko::array<index_type> expected(
        this->exec, {gko::invalid_index<index_type>(), index_type{100},
                     index_type{101}, gko::invalid_index<index_type>()});

    gko::kernels::reference::pmis::coarse_global_index(
        this->exec, 4, index_type{100}, status.get_const_data(),
        coarse_map.get_const_data(), coarse_global.get_data());

    GKO_ASSERT_ARRAY_EQ(coarse_global, expected);
}


TYPED_TEST(Pmis, DirectInterpolationRowCount)
{
    using index_type = typename TestFixture::index_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    std::array<gko::array<int>, 2> status{
        gko::array<int>(this->exec, {f, c, f, c}),
        gko::array<int>(this->exec, {f, f, c, f, c})};
    std::array<gko::array<index_type>, 2> row_count_ans{
        gko::array<index_type>(this->exec, {2, 1, 1, 1}),
        gko::array<index_type>(this->exec, {2, 1, 1, 2, 1})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        gko::array<index_type> prolong_row_count(
            this->exec, this->mtx.at(i)->get_size()[0]);
        // the kernel accumulates and skips coarse rows, so seed them here
        for (gko::size_type row = 0; row < prolong_row_count.get_size();
             row++) {
            prolong_row_count.get_data()[row] =
                status.at(i).get_const_data()[row] == c ? 1 : 0;
        }

        gko::kernels::reference::pmis::direct_interpolation_row_count(
            this->exec, index_type{0}, strong_dep.get(),
            status.at(i).get_const_data(), prolong_row_count.get_data());

        GKO_ASSERT_ARRAY_EQ(prolong_row_count, row_count_ans.at(i));
    }
}


TYPED_TEST(Pmis, DirectInterpolationRowCountAccumulatesAcrossBlocks)
{
    using index_type = typename TestFixture::index_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    // local row 0 has one strong local C-neighbour and one strong halo
    // C-neighbour; the two calls must total 2
    gko::array<index_type> d_ptrs(this->exec, {0, 1});
    gko::array<index_type> d_cols(this->exec, {1});
    auto s_diag = SparsityCsr::create(this->exec, gko::dim<2>{1, 2},
                                      std::move(d_cols), std::move(d_ptrs));
    gko::array<index_type> o_ptrs(this->exec, {0, 1});
    gko::array<index_type> o_cols(this->exec, {0});
    auto s_offd = SparsityCsr::create(this->exec, gko::dim<2>{1, 1},
                                      std::move(o_cols), std::move(o_ptrs));
    // extended status: node 0 fine (the row), node 1 coarse (local),
    // node 2 coarse (halo)
    gko::array<int> status(this->exec, {f, c, c});
    gko::array<index_type> count(this->exec, {0});
    gko::array<index_type> expected(this->exec, {2});

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        this->exec, index_type{0}, s_diag.get(), status.get_const_data(),
        count.get_data());
    gko::kernels::reference::pmis::direct_interpolation_row_count(
        this->exec, index_type{2}, s_offd.get(), status.get_const_data(),
        count.get_data());

    GKO_ASSERT_ARRAY_EQ(count, expected);
}


TYPED_TEST(Pmis, DirectInterpolationFill)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    // status matches coarse_map: coarse_map[row] != coarse_map[row + 1]
    // marks a coarse row
    std::array<gko::array<int>, 2> status{
        gko::array<int>(this->exec, {f, c, f, c}),
        gko::array<int>(this->exec, {f, f, c, f, c})};
    std::array<gko::array<index_type>, 2> coarse_map{
        gko::array<index_type>(this->exec, {0, 0, 1, 1, 2}),
        gko::array<index_type>(this->exec, {0, 0, 0, 1, 1, 2})};
    std::array<gko::array<index_type>, 2> prolong_row_ptrs{
        gko::array<index_type>(this->exec, {0, 1, 2, 3, 4}),
        gko::array<index_type>(this->exec, {0, 0, 0, 1, 2, 3})};
    std::array<gko::array<index_type>, 2> expected_col_idxs{
        gko::array<index_type>(this->exec, {0, 0, 0, 1}),
        gko::array<index_type>(this->exec, {0, 0, 1})};
    std::array<gko::array<value_type>, 2> expected_values{
        gko::array<value_type>(this->exec, {value_type(0.25), value_type(1),
                                            value_type(9), value_type(1)}),
        gko::array<value_type>(
            this->exec, {value_type{1}, value_type{0.5}, value_type{1}})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto prolong_nnz =
            prolong_row_ptrs.at(i)
                .get_const_data()[this->mtx.at(i)->get_size()[0]];
        gko::array<index_type> prolong_col_idxs(this->exec, prolong_nnz);
        gko::array<value_type> prolong_values(this->exec, prolong_nnz);

        this->fill_prolongation(
            this->mtx.at(i).get(), real_type{0.25}, real_type{1.0},
            status.at(i).get_const_data(), coarse_map.at(i).get_const_data(),
            prolong_row_ptrs.at(i).get_const_data(),
            prolong_col_idxs.get_data(), prolong_values.get_data());

        GKO_ASSERT_ARRAY_EQ(prolong_col_idxs, expected_col_idxs.at(i));
        GKO_ASSERT_ARRAY_EQ(prolong_values, expected_values.at(i));
    }
}


TYPED_TEST(Pmis, DirectInterpolationFillSkipsRowWithoutStrongDependence)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    // row 1 stores an explicit zero off the diagonal, so its largest
    // off-diagonal magnitude is zero and it has no strong dependence
    auto mtx = Mtx::create(
        this->exec, gko::dim<2>{2, 2},
        gko::array<value_type>(this->exec,
                               {value_type{1}, value_type{0}, value_type{1}}),
        gko::array<index_type>(this->exec, {0, 0, 1}),
        gko::array<index_type>(this->exec, {0, 1, 3}));
    // row 0 is coarse, row 1 is fine
    gko::array<int> status(this->exec, {c, f});
    gko::array<index_type> coarse_map(this->exec, {0, 1, 1});
    // only the identity entry of the coarse row 0 is counted, row 1 gets none
    gko::array<index_type> prolong_row_ptrs(this->exec, {0, 1, 1});
    // a guard: row 1 must not write past its own empty range
    gko::array<index_type> prolong_col_idxs(this->exec, {0, -99});
    gko::array<value_type> prolong_values(this->exec,
                                          {value_type{0}, value_type{-99}});
    gko::array<index_type> expected_col_idxs(this->exec, {0, -99});
    gko::array<value_type> expected_values(this->exec,
                                           {value_type{1}, value_type{-99}});

    this->fill_prolongation(
        mtx.get(), real_type{0.25}, real_type{1.0}, status.get_const_data(),
        coarse_map.get_const_data(), prolong_row_ptrs.get_const_data(),
        prolong_col_idxs.get_data(), prolong_values.get_data());

    GKO_ASSERT_ARRAY_EQ(prolong_col_idxs, expected_col_idxs);
    GKO_ASSERT_ARRAY_EQ(prolong_values, expected_values);
}


// the draw itself: a hash of the global index, inlined by every backend
TEST(PmisRandomWeight, IsInRangeAndReproducible)
{
    constexpr gko::size_type num = 1000;
    std::vector<float> values(num);
    std::vector<float> repeated(num);

    for (gko::size_type i = 0; i < num; i++) {
        values[i] = gko::kernels::pmis::random_weight_from_index(i);
        repeated[i] = gko::kernels::pmis::random_weight_from_index(i);
    }

    auto sum = 0.0;
    for (gko::size_type i = 0; i < num; i++) {
        SCOPED_TRACE(i);
        // the same index always gives the same value
        ASSERT_EQ(values[i], repeated[i]);
        ASSERT_GE(values[i], 0.0f);
        ASSERT_LT(values[i], 1.0f);
        sum += values[i];
    }
    // uniform on [0, 1] has mean 0.5 with a standard error of about 0.009 for
    // this sample size, so this only rejects a degenerate generator
    ASSERT_NEAR(sum / num, 0.5, 0.1);
}


TYPED_TEST(Pmis, AccumulateAndEmitAcrossBlocksMatchesWholeRow)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    // one fine row [4, -1, -2] with two strong coarse neighbours, split
    // across a "diagonal" and an "off-diagonal" block
    auto whole = Mtx::create(
        this->exec, gko::dim<2>{1, 3},
        gko::array<value_type>(this->exec,
                               {value_type{4}, value_type{-1}, value_type{-2}}),
        gko::array<index_type>(this->exec, {0, 1, 2}),
        gko::array<index_type>(this->exec, {0, 3}));
    auto block_a = Mtx::create(
        this->exec, gko::dim<2>{1, 2},
        gko::array<value_type>(this->exec, {value_type{4}, value_type{-1}}),
        gko::array<index_type>(this->exec, {0, 1}),
        gko::array<index_type>(this->exec, {0, 2}));
    auto block_b =
        Mtx::create(this->exec, gko::dim<2>{1, 1},
                    gko::array<value_type>(this->exec, {value_type{-2}}),
                    gko::array<index_type>(this->exec, {0}),
                    gko::array<index_type>(this->exec, {0, 1}));
    // node 0 is the fine row, nodes 1 and 2 are coarse
    gko::array<int> status(this->exec, {f, c, c});

    auto run = [&](std::vector<std::pair<Mtx*, index_type>> blocks) {
        gko::array<real_type> neg_scale(this->exec, {gko::zero<real_type>()});
        gko::array<real_type> row_sum(this->exec, {gko::zero<real_type>()});
        gko::array<real_type> diag_abs(this->exec, {gko::zero<real_type>()});
        for (auto& b : blocks) {
            gko::kernels::reference::pmis::compute_row_scales(
                this->exec, b.first, b.second == 0, neg_scale.get_data(),
                row_sum.get_data(), diag_abs.get_data());
        }
        gko::array<value_type> pos(this->exec, {gko::zero<value_type>()});
        gko::array<value_type> pos_div(this->exec, {gko::zero<value_type>()});
        gko::array<value_type> neg(this->exec, {gko::zero<value_type>()});
        gko::array<value_type> neg_div(this->exec, {gko::zero<value_type>()});
        gko::array<value_type> diag(this->exec, {gko::zero<value_type>()});
        gko::array<int> en_pos(this->exec, {0});
        gko::array<int> en_neg(this->exec, {0});
        gko::array<index_type> cursor(this->exec, {0});
        gko::array<index_type> cols(this->exec, {0, 0});
        gko::array<value_type> vals(
            this->exec, {gko::zero<value_type>(), gko::zero<value_type>()});
        const gko::kernels::pmis::interpolation_workspace<value_type,
                                                          index_type>
            ws{pos.get_data(),     pos_div.get_data(), neg.get_data(),
               neg_div.get_data(), diag.get_data(),    en_pos.get_data(),
               en_neg.get_data(),  cursor.get_data()};
        for (auto& b : blocks) {
            gko::kernels::reference::pmis::direct_interpolation_accumulate(
                this->exec, b.first, b.second == 0, b.second,
                neg_scale.get_const_data(), row_sum.get_const_data(),
                diag_abs.get_const_data(), real_type{0.25}, real_type{0.9},
                status.get_const_data(), ws);
        }
        for (auto& b : blocks) {
            gko::kernels::reference::pmis::direct_interpolation_emit(
                this->exec, b.first, b.second == 0, b.second,
                neg_scale.get_const_data(), row_sum.get_const_data(),
                diag_abs.get_const_data(), real_type{0.25}, real_type{0.9},
                status.get_const_data(), ws, cols.get_data(), vals.get_data());
        }
        return vals;
    };

    auto single = run({{whole.get(), index_type{0}}});
    auto split =
        run({{block_a.get(), index_type{0}}, {block_b.get(), index_type{2}}});

    GKO_ASSERT_ARRAY_EQ(split, single);
}


TYPED_TEST(Pmis, GenerateMgLevel)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    using real_type = typename TestFixture::real_type;
    auto factory = MgLevel::build()
                       .with_skip_sorting(true)
                       .with_max_row_sum(real_type{1.0})
                       .on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op1->transpose());

    auto mg = factory->generate(this->mtx.at(1));

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx.at(1), 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}


TYPED_TEST(Pmis, GenerateMgLevelStopsCoarseningWhenEveryRowIsFullySummed)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto factory = MgLevel::build().with_skip_sorting(true).on(this->exec);
    const auto num_rows = this->mtx.at(1)->get_size()[0];
    gko::matrix_data<value_type, index_type> identity_data{
        gko::dim<2>(num_rows, num_rows), {}};
    for (gko::size_type i = 0; i < num_rows; ++i) {
        identity_data.nonzeros.emplace_back(static_cast<index_type>(i),
                                            static_cast<index_type>(i),
                                            gko::one<value_type>());
    }
    auto identity = Mtx::create(this->exec);
    identity->read(identity_data);

    auto mg = factory->generate(this->mtx.at(1));

    ASSERT_EQ(mg->get_coarse_op()->get_size(), gko::dim<2>(num_rows, num_rows));
    ASSERT_EQ(mg->get_prolong_op()->get_size(),
              gko::dim<2>(num_rows, num_rows));
    ASSERT_EQ(mg->get_restrict_op()->get_size(),
              gko::dim<2>(num_rows, num_rows));
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), identity, 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), identity, 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->mtx.at(1),
                        0.0);
}


TYPED_TEST(Pmis, GenerateMgLevelOnUnsortedMtx)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    using real_type = typename TestFixture::real_type;
    auto factory =
        MgLevel::build().with_max_row_sum(real_type{1.0}).on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op1->transpose());
    std::default_random_engine rng{793643};
    gko::test::unsort_matrix(this->mtx.at(1), rng);

    auto mg = factory->generate(this->mtx.at(1));

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx.at(1), 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}


template <typename ValueIndexType>
class ClassicalInterpolation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    ClassicalInterpolation()
        : exec(gko::ReferenceExecutor::create()),
          path(Mtx::create(exec)),
          clique(Mtx::create(exec)),
          status(exec, {c, f, f, c}),
          coarse_map(exec, {0, -1, -1, 1}),
          row_maxabs(exec, {1, 1, 1, 1}),
          num_coarse(2)
    {
        path->read({{4, 4},
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
        clique->read({{4, 4},
                      {{0, 0, value_type{3}},
                       {0, 1, value_type{-1}},
                       {0, 2, value_type{-1}},
                       {0, 3, value_type{-1}},
                       {1, 0, value_type{-1}},
                       {1, 1, value_type{3}},
                       {1, 2, value_type{-1}},
                       {1, 3, value_type{-1}},
                       {2, 0, value_type{-1}},
                       {2, 1, value_type{-1}},
                       {2, 2, value_type{3}},
                       {2, 3, value_type{-1}},
                       {3, 0, value_type{-1}},
                       {3, 1, value_type{-1}},
                       {3, 2, value_type{-1}},
                       {3, 3, value_type{3}}}});
    }

    std::shared_ptr<Mtx> interpolate(const Mtx* mtx, real_type threshold)
    {
        const auto num_rows = mtx->get_size()[0];
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::array<index_type> row_ptrs(exec, num_rows + 1);
        gko::kernels::reference::pmis::classical_interpolation_row_count(
            exec, mtx, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            threshold, max_row_sum, status.get_const_data(),
            row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            exec, row_ptrs.get_data(), row_ptrs.get_size());
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        gko::array<index_type> col_idxs(exec, nnz);
        gko::array<value_type> values(exec, nnz);

        gko::kernels::reference::pmis::classical_interpolation_fill(
            exec, mtx, row_maxabs.get_const_data(),
            scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
            scales.diag_abs.get_const_data(), threshold, max_row_sum,
            status.get_const_data(), coarse_map.get_const_data(),
            row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

        return Mtx::create(exec, gko::dim<2>{num_rows, num_coarse},
                           std::move(values), std::move(col_idxs),
                           std::move(row_ptrs));
    }

    value_type entry(const Mtx* mtx, index_type row, index_type col) const
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

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> path;
    std::shared_ptr<Mtx> clique;
    gko::array<int> status;
    gko::array<index_type> coarse_map;
    gko::array<real_type> row_maxabs;
    real_type max_row_sum{0.9};
    gko::size_type num_coarse;
};

TYPED_TEST_SUITE(ClassicalInterpolation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ClassicalInterpolation, IsIdentityOnCoarseRows)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    auto prolong = this->interpolate(this->path.get(), real_type{0.25});
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();

    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], 1);
    EXPECT_EQ(col_idxs[row_ptrs[0]], this->coarse_map.get_const_data()[0]);
    EXPECT_EQ(values[row_ptrs[0]], value_type{1});
    ASSERT_EQ(row_ptrs[4] - row_ptrs[3], 1);
    EXPECT_EQ(col_idxs[row_ptrs[3]], this->coarse_map.get_const_data()[3]);
    EXPECT_EQ(values[row_ptrs[3]], value_type{1});
}


TYPED_TEST(ClassicalInterpolation, ReproducesConstants)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using Vec = typename TestFixture::Vec;
    auto prolong = this->interpolate(this->clique.get(), real_type{0.25});
    auto ones_coarse =
        Vec::create(this->exec, gko::dim<2>{this->num_coarse, 1});
    ones_coarse->fill(gko::one<value_type>());
    auto ones_fine = Vec::create(this->exec, gko::dim<2>{4, 1});
    ones_fine->fill(gko::one<value_type>());
    auto result = Vec::create(this->exec, gko::dim<2>{4, 1});

    prolong->apply(ones_coarse, result);

    GKO_ASSERT_MTX_NEAR(result, ones_fine, r<value_type>::value * 50);
}


TYPED_TEST(ClassicalInterpolation, ReproducesConstantsWithDegenerateFNeighbour)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using Vec = typename TestFixture::Vec;
    auto prolong = this->interpolate(this->path.get(), real_type{0.25});
    auto ones_coarse =
        Vec::create(this->exec, gko::dim<2>{this->num_coarse, 1});
    ones_coarse->fill(gko::one<value_type>());
    auto ones_fine = Vec::create(this->exec, gko::dim<2>{4, 1});
    ones_fine->fill(gko::one<value_type>());
    auto result = Vec::create(this->exec, gko::dim<2>{4, 1});

    prolong->apply(ones_coarse, result);

    GKO_ASSERT_MTX_NEAR(result, ones_fine, r<value_type>::value * 50);
}


TYPED_TEST(ClassicalInterpolation, ReproducesConstantsWithPositiveOffDiagonal)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    auto mixed_sign = Mtx::create(this->exec);
    mixed_sign->read({{4, 4},
                      {{0, 0, value_type{5}},
                       {0, 1, value_type{1}},
                       {0, 2, value_type{-1}},
                       {1, 0, value_type{1}},
                       {1, 1, value_type{3}},
                       {1, 2, value_type{-3}},
                       {1, 3, value_type{-1}},
                       {2, 0, value_type{-1}},
                       {2, 1, value_type{-3}},
                       {2, 2, value_type{5}},
                       {2, 3, value_type{-1}},
                       {3, 1, value_type{-1}},
                       {3, 2, value_type{-1}},
                       {3, 3, value_type{5}}}});
    this->row_maxabs = gko::array<real_type>(this->exec, {1, 3, 3, 1});

    auto prolong = this->interpolate(mixed_sign.get(), real_type{0.25});
    auto ones_coarse =
        Vec::create(this->exec, gko::dim<2>{this->num_coarse, 1});
    ones_coarse->fill(gko::one<value_type>());
    auto ones_fine = Vec::create(this->exec, gko::dim<2>{4, 1});
    ones_fine->fill(gko::one<value_type>());
    auto result = Vec::create(this->exec, gko::dim<2>{4, 1});

    prolong->apply(ones_coarse, result);

    GKO_ASSERT_MTX_NEAR(result, ones_fine, r<value_type>::value * 50);
}


TYPED_TEST(ClassicalInterpolation, BoundedWeightsWithCancellingSK)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto cancelling = Mtx::create(this->exec);
    cancelling->read({{5, 5},
                      {{0, 0, value_type{5}},
                       {0, 1, value_type{2}},
                       {1, 0, value_type{2}},
                       {1, 1, value_type{10}},
                       {1, 2, value_type{-4}},
                       {1, 3, value_type{-3}},
                       {1, 4, value_type{-3}},
                       {2, 1, value_type{-4}},
                       {2, 2, value_type{20}},
                       {2, 3, value_type{5}},
                       {2, 4, value_type{-5.0000001}},
                       {3, 1, value_type{-3}},
                       {3, 2, value_type{5}},
                       {3, 3, value_type{5}},
                       {4, 1, value_type{-3}},
                       {4, 2, value_type{-5.0000001}},
                       {4, 4, value_type{5}}}});
    this->status = gko::array<int>(this->exec, {c, f, f, c, c});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, -1, 1, 2});
    this->row_maxabs = gko::array<real_type>(
        this->exec, {real_type{2}, real_type{4}, real_type{5.0000001},
                     real_type{5}, real_type{5.0000001}});
    this->num_coarse = 3;

    auto prolong = this->interpolate(cancelling.get(), real_type{0.25});

    const auto values = prolong->get_const_values();
    for (gko::size_type i = 0; i < prolong->get_num_stored_elements(); i++) {
        EXPECT_LE(gko::abs(values[i]), real_type{10})
            << "entry " << i << " = " << values[i];
    }
}


TYPED_TEST(ClassicalInterpolation,
           BoundedWeightsWithCancellingFallbackDenominator)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto cancelling = Mtx::create(this->exec);
    cancelling->read({{3, 3},
                      {{0, 0, value_type{1}},
                       {1, 0, value_type{-1}},
                       {1, 1, value_type{0.05}},
                       {1, 2, value_type{-0.0499999}},
                       {2, 2, value_type{1}}}});
    this->status = gko::array<int>(this->exec, {c, f, c});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, 1});
    this->row_maxabs = gko::array<real_type>(
        this->exec, {real_type{0}, real_type{1}, real_type{0}});
    this->num_coarse = 2;
    this->max_row_sum = real_type{1.0};

    auto prolong = this->interpolate(cancelling.get(), real_type{0.25});

    const auto values = prolong->get_const_values();
    for (gko::size_type i = 0; i < prolong->get_num_stored_elements(); i++) {
        EXPECT_LE(gko::abs(values[i]), real_type{100})
            << "entry " << i << " = " << values[i];
    }
}


TYPED_TEST(ClassicalInterpolation, ReproducesConstantsWhenStabilityGuardFires)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto guard_fires = Mtx::create(this->exec);
    guard_fires->read({{4, 4},
                       {{0, 0, value_type{1}},
                        {1, 1, value_type{1}},
                        {1, 2, value_type{-0.94}},
                        {1, 3, value_type{-0.06}},
                        {2, 1, value_type{-0.94}},
                        {2, 2, value_type{0.94}},
                        {3, 1, value_type{-0.06}},
                        {3, 3, value_type{0.06}}}});
    this->status = gko::array<int>(this->exec, {c, f, f, c});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, -1, 1});
    this->row_maxabs = gko::array<real_type>(
        this->exec,
        {real_type{0}, real_type{0.94}, real_type{0.94}, real_type{0}});
    this->num_coarse = 2;

    auto prolong = this->interpolate(guard_fires.get(), real_type{0.05});
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 1)
        << "row 1 should have exactly one entry (its only strong "
           "C-neighbour, node 3)";
    value_type row1_sum = gko::zero<value_type>();
    for (auto idx = row_ptrs[1]; idx < row_ptrs[2]; idx++) {
        row1_sum += values[idx];
    }
    EXPECT_NEAR(static_cast<double>(gko::real(row1_sum)), 1.0,
                static_cast<double>(r<value_type>::value) * 50)
        << "row 1 sum = " << row1_sum;
}


TYPED_TEST(ClassicalInterpolation, EmitsZeroWeightsWhenDivisorIsTinyForRow)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto tiny_divisor = Mtx::create(this->exec);
    tiny_divisor->read({{3, 3},
                        {{0, 0, value_type{1}},
                         {1, 0, value_type{-1}},
                         {1, 1, value_type{1e-4}},
                         {1, 2, value_type{-1}},
                         {2, 2, value_type{1}}}});
    this->status = gko::array<int>(this->exec, {c, f, c});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, 1});
    this->row_maxabs = gko::array<real_type>(
        this->exec, {real_type{0}, real_type{1}, real_type{0}});
    this->num_coarse = 2;
    const real_type threshold{0.25};
    const real_type max_row_sum{1.0};
    const auto num_rows = tiny_divisor->get_size()[0];
    const auto scales = compute_pmis_row_scales(this->exec, tiny_divisor.get());
    gko::array<index_type> row_ptrs(this->exec, num_rows + 1);
    gko::kernels::reference::pmis::classical_interpolation_row_count(
        this->exec, tiny_divisor.get(), scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        threshold, max_row_sum, this->status.get_const_data(),
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, row_ptrs.get_data(), row_ptrs.get_size());
    ASSERT_EQ(row_ptrs.get_const_data()[2] - row_ptrs.get_const_data()[1], 2);
    const auto nnz =
        static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
    gko::array<index_type> col_idxs(this->exec, nnz);
    gko::array<value_type> values(this->exec, nnz);
    col_idxs.fill(index_type{-99});
    values.fill(value_type{-12345});

    gko::kernels::reference::pmis::classical_interpolation_fill(
        this->exec, tiny_divisor.get(), this->row_maxabs.get_const_data(),
        scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
        scales.diag_abs.get_const_data(), threshold, max_row_sum,
        this->status.get_const_data(), this->coarse_map.get_const_data(),
        row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

    const auto row1_begin = row_ptrs.get_const_data()[1];
    EXPECT_EQ(col_idxs.get_const_data()[row1_begin], index_type{0});
    EXPECT_EQ(col_idxs.get_const_data()[row1_begin + 1], index_type{1});
    for (auto idx = row1_begin; idx < row_ptrs.get_const_data()[2]; idx++) {
        SCOPED_TRACE(idx);
        const auto val = values.get_const_data()[idx];
        EXPECT_TRUE(gko::is_finite(val)) << "entry " << idx << " = " << val;
        EXPECT_EQ(val, gko::zero<value_type>())
            << "entry " << idx << " = " << val;
    }
}


TYPED_TEST(ClassicalInterpolation, OnlyUsesStrongCoarseNeighbours)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    const real_type threshold = 0.25;
    const auto scales = compute_pmis_row_scales(this->exec, this->path.get());
    auto prolong = this->interpolate(this->path.get(), threshold);
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    std::vector<index_type> node_of_coarse_col(this->num_coarse);
    for (index_type node = 0; node < 4; node++) {
        if (this->status.get_const_data()[node] == c) {
            node_of_coarse_col[this->coarse_map.get_const_data()[node]] = node;
        }
    }

    for (index_type row = 0; row < 4; row++) {
        if (this->status.get_const_data()[row] != f) {
            continue;
        }
        SCOPED_TRACE(row);
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto node = node_of_coarse_col[col_idxs[idx]];
            const auto val = this->entry(this->path.get(), row, node);
            EXPECT_GT(-gko::real(val),
                      threshold * scales.neg_scale.get_const_data()[row]);
        }
    }
}


TYPED_TEST(ClassicalInterpolation, RowCountMatchesFilledRows)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    const real_type threshold = 0.25;
    const auto num_rows = this->path->get_size()[0];
    const auto scales = compute_pmis_row_scales(this->exec, this->path.get());
    gko::array<index_type> counts(this->exec, num_rows);
    gko::kernels::reference::pmis::classical_interpolation_row_count(
        this->exec, this->path.get(), scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        threshold, this->max_row_sum, this->status.get_const_data(),
        counts.get_data());
    gko::array<index_type> row_ptrs(this->exec, num_rows + 1);
    this->exec->copy_from(this->exec, num_rows, counts.get_const_data(),
                          row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, row_ptrs.get_data(), row_ptrs.get_size());
    const auto nnz =
        static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
    gko::array<index_type> col_idxs(this->exec, nnz);
    gko::array<value_type> values(this->exec, nnz);
    col_idxs.fill(index_type{-99});

    gko::kernels::reference::pmis::classical_interpolation_fill(
        this->exec, this->path.get(), this->row_maxabs.get_const_data(),
        scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
        scales.diag_abs.get_const_data(), threshold, this->max_row_sum,
        this->status.get_const_data(), this->coarse_map.get_const_data(),
        row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

    for (index_type row = 0; row < static_cast<index_type>(num_rows); row++) {
        SCOPED_TRACE(row);
        gko::size_type written = 0;
        for (auto idx = row_ptrs.get_const_data()[row];
             idx < row_ptrs.get_const_data()[row + 1]; idx++) {
            if (col_idxs.get_const_data()[idx] != index_type{-99}) {
                written++;
            }
        }
        EXPECT_EQ(written,
                  static_cast<gko::size_type>(counts.get_const_data()[row]));
    }
}


TEST(ClassicalInterpolationDegenerate,
     FallsBackToDirectWhenDenominatorCancelsExactly)
{
    using value_type = double;
    using index_type = int;
    using real_type = double;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<value_type, index_type>::create(exec);
    mtx->read({{3, 3},
               {{0, 0, value_type{1}},
                {1, 0, value_type{-1}},
                {1, 1, value_type{0.05}},
                {1, 2, value_type{-0.05}},
                {2, 2, value_type{1}}}});
    gko::array<real_type> row_maxabs(exec, {0, 1, 0});
    gko::array<int> status(exec, {c, f, c});
    gko::array<index_type> coarse_map(exec, {0, -1, 1});
    const real_type max_row_sum{1.0};
    const auto scales = compute_pmis_row_scales(exec, mtx.get());
    gko::array<index_type> row_ptrs(exec, 4);
    gko::kernels::reference::pmis::classical_interpolation_row_count(
        exec, mtx.get(), scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        real_type{0.25}, max_row_sum, status.get_const_data(),
        row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        exec, row_ptrs.get_data(), row_ptrs.get_size());
    ASSERT_EQ(row_ptrs.get_const_data()[3], 3);
    gko::array<index_type> col_idxs(exec, 3);
    gko::array<value_type> values(exec, 3);

    gko::kernels::reference::pmis::classical_interpolation_fill(
        exec, mtx.get(), row_maxabs.get_const_data(),
        scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
        scales.diag_abs.get_const_data(), real_type{0.25}, max_row_sum,
        status.get_const_data(), coarse_map.get_const_data(),
        row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

    const auto row1_begin = row_ptrs.get_const_data()[1];
    EXPECT_EQ(col_idxs.get_const_data()[row1_begin], 0);
    EXPECT_NEAR(values.get_const_data()[row1_begin], 21.0, 1e-9);
}


template <typename ValueIndexType>
class ExtIInterpolation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    ExtIInterpolation()
        : exec(gko::ReferenceExecutor::create()),
          path(Mtx::create(exec)),
          status(exec, {c, f, f, c}),
          coarse_map(exec, {0, -1, -1, 1}),
          row_maxabs(exec, {1, 1, 1, 1}),
          max_row_sum{0.9},
          num_coarse(2)
    {
        path->read({{4, 4},
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
    }

    gko::array<index_type> ext_i_row_count(const Mtx* mtx, real_type threshold)
    {
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::array<index_type> counts(exec, mtx->get_size()[0]);
        gko::kernels::reference::pmis::ext_i_interpolation_row_count(
            exec, mtx, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            threshold, max_row_sum, status.get_const_data(), counts.get_data());
        return counts;
    }

    gko::array<index_type> classical_row_count(const Mtx* mtx,
                                               real_type threshold)
    {
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::array<index_type> counts(exec, mtx->get_size()[0]);
        gko::kernels::reference::pmis::classical_interpolation_row_count(
            exec, mtx, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            threshold, max_row_sum, status.get_const_data(), counts.get_data());
        return counts;
    }

    std::shared_ptr<Mtx> interpolate(const Mtx* mtx, real_type threshold)
    {
        const auto num_rows = mtx->get_size()[0];
        const auto scales = compute_pmis_row_scales(exec, mtx);
        gko::array<index_type> row_ptrs(exec, num_rows + 1);
        gko::kernels::reference::pmis::ext_i_interpolation_row_count(
            exec, mtx, scales.neg_scale.get_const_data(),
            scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
            threshold, max_row_sum, status.get_const_data(),
            row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            exec, row_ptrs.get_data(), row_ptrs.get_size());
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        gko::array<index_type> col_idxs(exec, nnz);
        gko::array<value_type> values(exec, nnz);

        gko::kernels::reference::pmis::ext_i_interpolation_fill(
            exec, mtx, row_maxabs.get_const_data(),
            scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
            scales.diag_abs.get_const_data(), threshold, max_row_sum,
            status.get_const_data(), coarse_map.get_const_data(),
            row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

        return Mtx::create(exec, gko::dim<2>{num_rows, num_coarse},
                           std::move(values), std::move(col_idxs),
                           std::move(row_ptrs));
    }

    value_type entry(const Mtx* mtx, index_type row, index_type col) const
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

    std::shared_ptr<Mtx> sigma_zero_mtx()
    {
        auto mtx = Mtx::create(exec);
        mtx->read({{5, 5},
                   {{0, 0, value_type{1}},
                    {0, 1, value_type{-1}},
                    {1, 0, value_type{-1}},
                    {1, 1, value_type{2}},
                    {1, 2, value_type{-1}},
                    {2, 1, value_type{1}},
                    {2, 2, value_type{2}},
                    {2, 4, value_type{-3}},
                    {3, 3, value_type{1}},
                    {3, 4, value_type{-1}},
                    {4, 2, value_type{-1}},
                    {4, 3, value_type{-1}},
                    {4, 4, value_type{2}}}});
        return mtx;
    }

    std::shared_ptr<Mtx> collapsing_mtx()
    {
        auto mtx = Mtx::create(exec);
        mtx->read({{4, 4},
                   {{0, 0, value_type{1}},
                    {0, 1, value_type{-1}},
                    {1, 0, value_type{-1}},
                    {1, 1, value_type{1}},
                    {1, 2, value_type{-1}},
                    {2, 1, value_type{-1}},
                    {2, 2, value_type{2}},
                    {2, 3, value_type{-1}},
                    {3, 2, value_type{-1}},
                    {3, 3, value_type{1}}}});
        return mtx;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> path;
    gko::array<int> status;
    gko::array<index_type> coarse_map;
    gko::array<real_type> row_maxabs;
    real_type max_row_sum;
    gko::size_type num_coarse;
};

TYPED_TEST_SUITE(ExtIInterpolation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ExtIInterpolation, IsIdentityOnCoarseRows)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    auto prolong = this->interpolate(this->path.get(), real_type{0.25});
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();

    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], 1);
    EXPECT_EQ(col_idxs[row_ptrs[0]], this->coarse_map.get_const_data()[0]);
    EXPECT_EQ(values[row_ptrs[0]], value_type{1});
    ASSERT_EQ(row_ptrs[4] - row_ptrs[3], 1);
    EXPECT_EQ(col_idxs[row_ptrs[3]], this->coarse_map.get_const_data()[3]);
    EXPECT_EQ(values[row_ptrs[3]], value_type{1});
}


TYPED_TEST(ExtIInterpolation, ExtendedSetIsStrictlyLargerThanStrongC)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto ext_i = this->ext_i_row_count(this->path.get(), real_type{0.25});
    auto classical =
        this->classical_row_count(this->path.get(), real_type{0.25});

    EXPECT_EQ(ext_i.get_const_data()[0], index_type{1});
    EXPECT_EQ(classical.get_const_data()[0], index_type{1});
    EXPECT_EQ(classical.get_const_data()[1], index_type{1});
    EXPECT_EQ(ext_i.get_const_data()[1], index_type{2});
    EXPECT_EQ(classical.get_const_data()[2], index_type{1});
    EXPECT_EQ(ext_i.get_const_data()[2], index_type{2});
    EXPECT_EQ(this->entry(this->path.get(), index_type{1}, index_type{3}),
              gko::zero<value_type>());

    auto prolong = this->interpolate(this->path.get(), real_type{0.25});
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 2);
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], index_type{1});
}


TYPED_TEST(ExtIInterpolation, ReproducesConstants)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using Vec = typename TestFixture::Vec;
    auto prolong = this->interpolate(this->path.get(), real_type{0.25});
    auto ones_coarse =
        Vec::create(this->exec, gko::dim<2>{this->num_coarse, 1});
    ones_coarse->fill(gko::one<value_type>());
    auto ones_fine = Vec::create(this->exec, gko::dim<2>{4, 1});
    ones_fine->fill(gko::one<value_type>());
    auto result = Vec::create(this->exec, gko::dim<2>{4, 1});

    prolong->apply(ones_coarse, result);

    GKO_ASSERT_MTX_NEAR(result, ones_fine, r<value_type>::value * 50);
}


TYPED_TEST(ExtIInterpolation, LumpsFNeighbourWithZeroSigmaK)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Vec = typename TestFixture::Vec;
    auto mtx = this->sigma_zero_mtx();
    this->status = gko::array<int>(this->exec, {c, f, f, c, f});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, -1, 1, -1});
    this->row_maxabs = gko::array<real_type>(this->exec, {1, 1, 3, 1, 1});
    this->num_coarse = 2;

    auto prolong = this->interpolate(mtx.get(), real_type{0.25});

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 1);
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{0});
    EXPECT_NEAR(static_cast<double>(gko::real(values[row_ptrs[1]])), 1.0,
                static_cast<double>(r<value_type>::value) * 50);
    auto ones_coarse =
        Vec::create(this->exec, gko::dim<2>{this->num_coarse, 1});
    ones_coarse->fill(gko::one<value_type>());
    auto ones_fine = Vec::create(this->exec, gko::dim<2>{5, 1});
    ones_fine->fill(gko::one<value_type>());
    auto result = Vec::create(this->exec, gko::dim<2>{5, 1});

    prolong->apply(ones_coarse, result);

    GKO_ASSERT_MTX_NEAR(result, ones_fine, r<value_type>::value * 50);
}


TYPED_TEST(ExtIInterpolation, EmitsZeroWeightsWhenDivisorCollapses)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto mtx = this->collapsing_mtx();
    this->status = gko::array<int>(this->exec, {c, f, f, f});
    this->coarse_map = gko::array<index_type>(this->exec, {0, -1, -1, -1});
    this->row_maxabs = gko::array<real_type>(this->exec, {1, 1, 1, 1});
    this->num_coarse = 1;
    this->max_row_sum = real_type{1.0};

    auto prolong = this->interpolate(mtx.get(), real_type{0.25});

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 1);
    const auto idx = row_ptrs[1];
    EXPECT_EQ(col_idxs[idx], index_type{0});
    EXPECT_TRUE(gko::is_finite(values[idx]))
        << "entry " << idx << " = " << values[idx];
    EXPECT_EQ(values[idx], gko::zero<value_type>())
        << "entry " << idx << " = " << values[idx];
}


TYPED_TEST(ExtIInterpolation, RowCountMatchesFilledRows)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    const real_type threshold{0.25};
    const auto num_rows = this->path->get_size()[0];
    const auto scales = compute_pmis_row_scales(this->exec, this->path.get());
    auto counts = this->ext_i_row_count(this->path.get(), threshold);
    gko::array<index_type> row_ptrs(this->exec, num_rows + 1);
    this->exec->copy_from(this->exec, num_rows, counts.get_const_data(),
                          row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, row_ptrs.get_data(), row_ptrs.get_size());
    const auto nnz =
        static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
    gko::array<index_type> col_idxs(this->exec, nnz);
    gko::array<value_type> values(this->exec, nnz);
    col_idxs.fill(index_type{-99});

    gko::kernels::reference::pmis::ext_i_interpolation_fill(
        this->exec, this->path.get(), this->row_maxabs.get_const_data(),
        scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
        scales.diag_abs.get_const_data(), threshold, this->max_row_sum,
        this->status.get_const_data(), this->coarse_map.get_const_data(),
        row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

    for (index_type row = 0; row < static_cast<index_type>(num_rows); row++) {
        SCOPED_TRACE(row);
        gko::size_type written = 0;
        for (auto idx = row_ptrs.get_const_data()[row];
             idx < row_ptrs.get_const_data()[row + 1]; idx++) {
            if (col_idxs.get_const_data()[idx] != index_type{-99}) {
                written++;
            }
        }
        EXPECT_EQ(written,
                  static_cast<gko::size_type>(counts.get_const_data()[row]));
    }
}


namespace {


std::shared_ptr<gko::matrix::Csr<double, int>> ext_i_interpolate_double(
    std::shared_ptr<const gko::ReferenceExecutor> exec,
    const gko::matrix::Csr<double, int>* mtx,
    const gko::array<double>& row_maxabs, const gko::array<int>& status,
    const gko::array<int>& coarse_map, double threshold, double max_row_sum,
    gko::size_type num_coarse)
{
    const auto num_rows = mtx->get_size()[0];
    const auto scales = compute_pmis_row_scales(exec, mtx);
    gko::array<int> row_ptrs(exec, num_rows + 1);
    gko::kernels::reference::pmis::ext_i_interpolation_row_count(
        exec, mtx, scales.neg_scale.get_const_data(),
        scales.row_sum.get_const_data(), scales.diag_abs.get_const_data(),
        threshold, max_row_sum, status.get_const_data(), row_ptrs.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        exec, row_ptrs.get_data(), row_ptrs.get_size());
    const auto nnz =
        static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
    gko::array<int> col_idxs(exec, nnz);
    gko::array<double> values(exec, nnz);

    gko::kernels::reference::pmis::ext_i_interpolation_fill(
        exec, mtx, row_maxabs.get_const_data(),
        scales.neg_scale.get_const_data(), scales.row_sum.get_const_data(),
        scales.diag_abs.get_const_data(), threshold, max_row_sum,
        status.get_const_data(), coarse_map.get_const_data(),
        row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());

    return gko::matrix::Csr<double, int>::create(
        exec, gko::dim<2>{num_rows, num_coarse}, std::move(values),
        std::move(col_idxs), std::move(row_ptrs));
}


}  // namespace


TEST(ExtIInterpolationExact, PathWeightsMatchHandComputation)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<double, int>::create(exec);
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, -1.0},
                {1, 0, -1.0},
                {1, 1, 2.0},
                {1, 2, -1.0},
                {2, 1, -1.0},
                {2, 2, 2.0},
                {2, 3, -1.0},
                {3, 2, -1.0},
                {3, 3, 1.0}}});
    gko::array<double> row_maxabs(exec, {1, 1, 1, 1});
    gko::array<int> status(exec, {c, f, f, c});
    gko::array<int> coarse_map(exec, {0, -1, -1, 1});

    auto prolong = ext_i_interpolate_double(exec, mtx.get(), row_maxabs, status,
                                            coarse_map, 0.25, 0.9, 2);

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 2);
    EXPECT_NEAR(values[row_ptrs[1]], 2.0 / 3.0, 1e-9);
    EXPECT_NEAR(values[row_ptrs[1] + 1], 1.0 / 3.0, 1e-9);
    ASSERT_EQ(row_ptrs[3] - row_ptrs[2], 2);
    EXPECT_NEAR(values[row_ptrs[2]], 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(values[row_ptrs[2] + 1], 2.0 / 3.0, 1e-9);
}


TEST(ExtIInterpolationExact, WeaklyConnectedCoarsePointInExtendedSetIsNumerator)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<double, int>::create(exec);
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, -1.0},
                {1, 0, -1.0},
                {1, 1, 2.25},
                {1, 2, -1.0},
                {1, 3, -0.25},
                {2, 1, -1.0},
                {2, 2, 2.0},
                {2, 3, -1.0},
                {3, 1, -0.25},
                {3, 2, -1.0},
                {3, 3, 1.25}}});
    gko::array<double> row_maxabs(exec, {1, 1, 1, 1});
    gko::array<int> status(exec, {c, f, f, c});
    gko::array<int> coarse_map(exec, {0, -1, -1, 1});

    auto prolong = ext_i_interpolate_double(exec, mtx.get(), row_maxabs, status,
                                            coarse_map, 0.25, 0.9, 2);

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 2);
    EXPECT_EQ(col_idxs[row_ptrs[1]], 0);
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], 1);
    EXPECT_NEAR(values[row_ptrs[1]], 1.0 / 1.75, 1e-9);
    EXPECT_NEAR(values[row_ptrs[1] + 1], 0.75 / 1.75, 1e-9);
    EXPECT_NEAR(values[row_ptrs[1]] + values[row_ptrs[1] + 1], 1.0, 1e-9);
}


TEST(ExtIInterpolationExact, EmitsZeroWeightsWhenDivisorNearlyCollapses)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<double, int>::create(exec);
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, -1.0},
                {1, 0, -1.0},
                {1, 1, 1.0 + 1e-7},
                {1, 2, -1.0},
                {2, 1, -1.0},
                {2, 2, 2.0},
                {2, 3, -1.0},
                {3, 2, -1.0},
                {3, 3, 1.0}}});
    gko::array<double> row_maxabs(exec, {1, 1, 1, 1});
    gko::array<int> status(exec, {c, f, f, f});
    gko::array<int> coarse_map(exec, {0, -1, -1, -1});
    const double max_row_sum = 1.0;

    auto prolong = ext_i_interpolate_double(exec, mtx.get(), row_maxabs, status,
                                            coarse_map, 0.25, max_row_sum, 1);

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 1);
    EXPECT_EQ(col_idxs[row_ptrs[1]], 0);
    EXPECT_TRUE(gko::is_finite(values[row_ptrs[1]]));
    EXPECT_EQ(values[row_ptrs[1]], 0.0);
}


TEST(ExtIInterpolationExact, NegativeNeighbourDiagonalFlipsTheSignFilter)
{
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::matrix::Csr<double, int>::create(exec);
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, -1.0},
                {1, 0, -1.0},
                {1, 1, 3.0},
                {1, 2, -1.0},
                {1, 3, -1.0},
                {2, 0, 2.0},
                {2, 1, 1.0},
                {2, 2, -4.0},
                {2, 3, 1.0},
                {3, 1, -1.0},
                {3, 3, 1.0}}});
    gko::array<double> row_maxabs(exec, {1, 1, 2, 1});
    gko::array<int> status(exec, {c, f, f, c});
    gko::array<int> coarse_map(exec, {0, -1, -1, 1});

    auto prolong = ext_i_interpolate_double(exec, mtx.get(), row_maxabs, status,
                                            coarse_map, 0.25, 0.9, 2);

    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 2);
    EXPECT_EQ(col_idxs[row_ptrs[1]], 0);
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], 1);
    EXPECT_NEAR(values[row_ptrs[1]], 6.0 / 11.0, 1e-9);
    EXPECT_NEAR(values[row_ptrs[1] + 1], 5.0 / 11.0, 1e-9);
    EXPECT_NEAR(values[row_ptrs[1]] + values[row_ptrs[1] + 1], 1.0, 1e-9);
}


template <typename ValueIndexType>
class TruncateInterpolation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    TruncateInterpolation()
        : exec(gko::ReferenceExecutor::create()), wide(Mtx::create(exec))
    {
        wide->read({{3, 6},
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
    }

    std::shared_ptr<Mtx> truncate(const Mtx* mtx, gko::size_type p_max_elmts)
    {
        const auto num_rows = mtx->get_size()[0];
        gko::array<index_type> row_ptrs(exec, num_rows + 1);
        gko::kernels::reference::pmis::truncate_interpolation_row_count(
            exec, num_rows, p_max_elmts, mtx->get_const_row_ptrs(),
            row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            exec, row_ptrs.get_data(), row_ptrs.get_size());
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        gko::array<index_type> col_idxs(exec, nnz);
        gko::array<value_type> values(exec, nnz);
        gko::kernels::reference::pmis::truncate_interpolation_fill(
            exec, num_rows, p_max_elmts, mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), mtx->get_const_values(),
            row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());
        return Mtx::create(exec, mtx->get_size(), std::move(values),
                           std::move(col_idxs), std::move(row_ptrs));
    }

    value_type row_sum(const Mtx* mtx, index_type row) const
    {
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto values = mtx->get_const_values();
        auto sum = gko::zero<value_type>();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            sum += values[idx];
        }
        return sum;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> wide;
};

TYPED_TEST_SUITE(TruncateInterpolation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(TruncateInterpolation, LeavesRowWithFewerEntriesUntouched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto out = this->truncate(this->wide.get(), gko::size_type{3});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], 2);
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{0});
    EXPECT_EQ(values[row_ptrs[0]], value_type{4});
    EXPECT_EQ(col_idxs[row_ptrs[0] + 1], index_type{5});
    EXPECT_EQ(values[row_ptrs[0] + 1], value_type{-2});
}


TYPED_TEST(TruncateInterpolation, LeavesRowWithExactlyPMaxElmtsUntouched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto out = this->truncate(this->wide.get(), gko::size_type{3});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[2] - row_ptrs[1], 3);
    EXPECT_EQ(col_idxs[row_ptrs[1]], index_type{1});
    EXPECT_EQ(values[row_ptrs[1]], value_type{3});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 1], index_type{2});
    EXPECT_EQ(values[row_ptrs[1] + 1], value_type{-7});
    EXPECT_EQ(col_idxs[row_ptrs[1] + 2], index_type{3});
    EXPECT_EQ(values[row_ptrs[1] + 2], value_type{1});
}


TYPED_TEST(TruncateInterpolation, KeepsLargestMagnitudeEntries)
{
    using index_type = typename TestFixture::index_type;
    auto out = this->truncate(this->wide.get(), gko::size_type{3});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    ASSERT_EQ(row_ptrs[3] - row_ptrs[2], 3);
    EXPECT_EQ(col_idxs[row_ptrs[2]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[2] + 1], index_type{2});
    EXPECT_EQ(col_idxs[row_ptrs[2] + 2], index_type{3});
}


TYPED_TEST(TruncateInterpolation, ReproducesOriginalRowSum)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    const auto expected = this->row_sum(this->wide.get(), 2);
    const auto tol = std::numeric_limits<real_type>::epsilon() * real_type{16};

    auto out = this->truncate(this->wide.get(), gko::size_type{3});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto values = out->get_const_values();
    auto actual = gko::zero<value_type>();
    for (auto idx = row_ptrs[2]; idx < row_ptrs[3]; idx++) {
        actual += values[idx];
    }
    EXPECT_LE(static_cast<double>(gko::abs(actual - expected)),
              static_cast<double>(tol * gko::abs(expected)));
}


TYPED_TEST(TruncateInterpolation, KeptEntriesSummingToZeroAreLeftUnscaled)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);
    mtx->read({{1, 3},
               {{0, 0, value_type{5}},
                {0, 1, value_type{-5}},
                {0, 2, value_type{2}}}});

    auto out = this->truncate(mtx.get(), gko::size_type{2});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], 2);
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{0});
    EXPECT_EQ(col_idxs[row_ptrs[0] + 1], index_type{1});
    EXPECT_EQ(values[row_ptrs[0]], value_type{5});
    EXPECT_EQ(values[row_ptrs[0] + 1], value_type{-5});
    EXPECT_TRUE(gko::is_finite(values[row_ptrs[0]]));
    EXPECT_TRUE(gko::is_finite(values[row_ptrs[0] + 1]));
}


template <typename ValueIndexType>
class TruncateInterpolationComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    TruncateInterpolationComplex() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<Mtx> truncate(const Mtx* mtx, gko::size_type p_max_elmts)
    {
        const auto num_rows = mtx->get_size()[0];
        gko::array<index_type> row_ptrs(exec, num_rows + 1);
        gko::kernels::reference::pmis::truncate_interpolation_row_count(
            exec, num_rows, p_max_elmts, mtx->get_const_row_ptrs(),
            row_ptrs.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            exec, row_ptrs.get_data(), row_ptrs.get_size());
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        gko::array<index_type> col_idxs(exec, nnz);
        gko::array<value_type> values(exec, nnz);
        gko::kernels::reference::pmis::truncate_interpolation_fill(
            exec, num_rows, p_max_elmts, mtx->get_const_row_ptrs(),
            mtx->get_const_col_idxs(), mtx->get_const_values(),
            row_ptrs.get_const_data(), col_idxs.get_data(), values.get_data());
        return Mtx::create(exec, mtx->get_size(), std::move(values),
                           std::move(col_idxs), std::move(row_ptrs));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(TruncateInterpolationComplex,
                 gko::test::ComplexValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(TruncateInterpolationComplex, SelectionUsesMagnitudeNotRealPart)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);
    mtx->read({{1, 2}, {{0, 0, value_type{3, 0}}, {0, 1, value_type{1, 4}}}});

    auto out = this->truncate(mtx.get(), gko::size_type{1});

    const auto row_ptrs = out->get_const_row_ptrs();
    const auto col_idxs = out->get_const_col_idxs();
    const auto values = out->get_const_values();
    ASSERT_EQ(row_ptrs[1] - row_ptrs[0], 1);
    EXPECT_EQ(col_idxs[row_ptrs[0]], index_type{1});
    const auto mass = gko::abs(value_type{3, 0}) + gko::abs(value_type{1, 4});
    const auto tol = std::numeric_limits<real_type>::epsilon() * real_type{16};
    EXPECT_LE(
        static_cast<double>(gko::abs(values[row_ptrs[0]] - value_type{4, 4})),
        static_cast<double>(tol * mass));
}
