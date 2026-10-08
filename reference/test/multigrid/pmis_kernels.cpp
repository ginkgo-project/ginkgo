// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
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
        prolong_op1->read({{5, 2},
                           {{0, 0, value_type{-0.25}},
                            {0, 1, value_type{-0.25}},
                            {1, 1, value_type{-0.25}},
                            {2, 0, value_type{1}},
                            {3, 0, value_type{0.5}},
                            {3, 1, value_type{-0.5}},
                            {4, 1, value_type{1}}}});
        coarse_op1->read({{2, 2},
                          {{0, 0, value_type{3}},
                           {0, 1, value_type{-0.25}},
                           {1, 1, value_type{1}}}});
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
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        gko::array<index_type> sparsity_rows(
            this->exec, this->mtx.at(i)->get_size()[0] + 1);

        gko::kernels::reference::pmis::compute_strong_dep_row(
            this->exec, this->mtx.at(i).get(), true,
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            sparsity_rows.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

        GKO_ASSERT_ARRAY_EQ(sparsity_rows, this->dep_row_ptrs.at(i));
    }
}


TYPED_TEST(Pmis, ComputeStrongDep)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto num_rows = this->mtx.at(i)->get_size()[0];
        auto sparsity_rows = this->dep_row_ptrs.at(i);
        auto strong_dep_ans =
            gko::matrix::SparsityCsr<value_type, index_type>::create(
                this->exec, this->mtx.at(i)->get_size(),
                std::move(this->dep_col_idxs.at(i)),
                std::move(this->dep_row_ptrs.at(i)));
        gko::array<index_type> sparsity_cols(
            this->exec, sparsity_rows.get_const_data()[num_rows]);
        auto strong_dep =
            gko::matrix::SparsityCsr<value_type, index_type>::create(
                this->exec, this->mtx.at(i)->get_size(),
                std::move(sparsity_cols), std::move(sparsity_rows));

        gko::kernels::reference::pmis::compute_strong_dep(
            this->exec, this->mtx.at(i).get(), true,
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            strong_dep.get());

        GKO_ASSERT_MTX_EQ_SPARSITY(strong_dep, strong_dep_ans);
    }
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


TYPED_TEST(Pmis, Count)
{
    gko::array<int> arr(this->exec, 5);
    auto data = arr.get_data();
    data[0] = u;
    data[1] = f;
    data[2] = u;
    data[3] = c;
    data[4] = u;

    gko::size_type num = 0;
    gko::kernels::reference::pmis::count(this->exec, 5, arr.get_const_data(),
                                         &num);

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
        gko::array<index_type>(this->exec, {0, 2, 3, 4, 5}),
        gko::array<index_type>(this->exec, {0, 2, 3, 4, 6, 7})};
    std::array<gko::array<index_type>, 2> expected_col_idxs{
        gko::array<index_type>(this->exec, {0, 1, 0, 0, 1}),
        gko::array<index_type>(this->exec, {0, 1, 1, 0, 0, 1, 1})};
    std::array<gko::array<value_type>, 2> expected_values{
        gko::array<value_type>(
            this->exec, {value_type(0.25), value_type(-1.5), value_type(1),
                         value_type(9), value_type(1)}),
        gko::array<value_type>(
            this->exec,
            {value_type{-0.25}, value_type{-0.25}, value_type{-0.25},
             value_type{1}, value_type{0.5}, value_type{-0.5}, value_type{1}})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto prolong_nnz =
            prolong_row_ptrs.at(i)
                .get_const_data()[this->mtx.at(i)->get_size()[0]];
        gko::array<index_type> prolong_col_idxs(this->exec, prolong_nnz);
        gko::array<value_type> prolong_values(this->exec, prolong_nnz);

        gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
            this->exec, this->mtx.at(i)->get_size()[0],
            {{this->mtx.at(i).get(), true, index_type{0}}},
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            status.at(i).get_const_data(),
            prolong_row_ptrs.at(i).get_const_data(),
            prolong_col_idxs.get_data(), prolong_values.get_data());
        this->exec->run(gko::multigrid::pmis::make_gather(
            static_cast<gko::size_type>(prolong_nnz),
            coarse_map.at(i).get_const_data(),
            prolong_col_idxs.get_const_data(), prolong_col_idxs.get_data()));

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
    gko::array<real_type> row_maxabs(this->exec, {0, 0});
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

    gko::multigrid::pmis::fill_prolongation<value_type, index_type>(
        this->exec, mtx->get_size()[0], {{mtx.get(), true, index_type{0}}},
        row_maxabs.get_const_data(), real_type{0.25}, status.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data());
    this->exec->run(gko::multigrid::pmis::make_gather(
        gko::size_type{1}, coarse_map.get_const_data(),
        prolong_col_idxs.get_const_data(), prolong_col_idxs.get_data()));

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
    gko::array<real_type> row_maxabs(this->exec, {2});

    auto run = [&](std::vector<std::pair<Mtx*, index_type>> blocks) {
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
                row_maxabs.get_const_data(), real_type{0.25},
                status.get_const_data(), ws);
        }
        for (auto& b : blocks) {
            gko::kernels::reference::pmis::direct_interpolation_emit(
                this->exec, b.first, b.second == 0, b.second,
                row_maxabs.get_const_data(), real_type{0.25},
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
    auto factory = MgLevel::build().with_skip_sorting(true).on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op1->transpose());

    auto mg = factory->generate(this->mtx.at(1));

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx.at(1), 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}


TYPED_TEST(Pmis, GenerateMgLevelOnUnsortedMtx)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto factory = MgLevel::build().on(this->exec);
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
