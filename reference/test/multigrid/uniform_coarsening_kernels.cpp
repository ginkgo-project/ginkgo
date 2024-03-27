// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/uniform_coarsening.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/multigrid/uniform_coarsening_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class UniformCoarsening : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::UniformCoarsening<value_type, index_type>;
    using RowGatherer = gko::matrix::RowGatherer<index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    UniformCoarsening()
        : exec(gko::ReferenceExecutor::create()),
          uniform_coarsening_factory(
              MgLevel::build().with_coarse_skip(2u).with_skip_sorting(true).on(
                  exec)),
          fine_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({-1.0, 2.0}), I<VT>({0.0, -1.0}),
               I<VT>({3.0, -2.0}), I<VT>({-2.0, 1.0})},
              exec)),
          coarse_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({3.0, 1.0}), I<VT>({0.0, -1.0})},
              exec)),
          restrict_ans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0}), I<VT>({-2.0, 1.0})},
              exec)),
          prolong_ans(gko::initialize<Vec>(
              {I<VT>({0.0, -2.0}), I<VT>({1.0, -1.0}), I<VT>({2.0, 0.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, 1.0})},
              exec)),
          prolong_applyans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, 0.0}), I<VT>({3.0, 1.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, -1.0})},
              exec)),
          fine_x(gko::initialize<Vec>(
              {I<VT>({-2.0, -1.0}), I<VT>({1.0, -1.0}), I<VT>({-1.0, -1.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, 2.0})},
              exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          coarse(Mtx::create(exec, gko::dim<2>(3, 3), 7,
                             std::make_shared<typename Mtx::classical>())),
          coarse_rows(exec, 5)
    {
        this->create_mtx(mtx.get(), &coarse_rows, coarse.get());
        mg_level = uniform_coarsening_factory->generate(mtx);
    }

    void create_mtx(Mtx* fine, gko::array<index_type>* coarse_rows, Mtx* coarse)
    {
        auto coarse_rows_val = coarse_rows->get_data();
        coarse_rows_val[0] = 0;
        coarse_rows_val[1] = -1;
        coarse_rows_val[2] = 1;
        coarse_rows_val[3] = -1;
        coarse_rows_val[4] = 2;

        /* this matrix is stored:
         *  5 -3 -3  0  0
         * -3  5  0 -2 -1
         * -3  0  5  0 -1
         *  0 -3  0  5  0
         *  0 -2 -2  0  5
         */
        fine->read({{5, 5},
                    {{0, 0, 5},
                     {0, 1, -3},
                     {0, 2, -3},
                     {1, 0, -3},
                     {1, 1, 5},
                     {1, 3, -2},
                     {1, 4, -1},
                     {2, 0, -3},
                     {2, 2, 5},
                     {2, 4, -1},
                     {3, 1, -3},
                     {3, 3, 5},
                     {4, 1, -2},
                     {4, 2, -2},
                     {4, 4, 5}}});


        /* this coarse is stored:
         *  5 -3  0
         * -3  5 -1
         *  0 -2  5
         */
        coarse->read({{3, 3},
                      {{0, 0, 5},
                       {0, 1, -3},
                       {1, 0, -3},
                       {1, 1, 5},
                       {1, 2, -1},
                       {2, 1, -2},
                       {2, 2, 5}}});
    }

    static void assert_same_matrices(const Mtx* m1, const Mtx* m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        ASSERT_EQ(m1->get_num_stored_elements(), m2->get_num_stored_elements());
        for (gko::size_type i = 0; i < m1->get_size()[0] + 1; i++) {
            ASSERT_EQ(m1->get_const_row_ptrs()[i], m2->get_const_row_ptrs()[i]);
        }
        for (gko::size_type i = 0; i < m1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m1->get_const_values()[i], m2->get_const_values()[i]);
            EXPECT_EQ(m1->get_const_col_idxs()[i], m2->get_const_col_idxs()[i]);
        }
    }

    static void assert_same_coarse_rows(const index_type* m1,
                                        const index_type* m2,
                                        gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    gko::array<index_type> coarse_rows;
    std::shared_ptr<Vec> coarse_b;
    std::shared_ptr<Vec> fine_b;
    std::shared_ptr<Vec> restrict_ans;
    std::shared_ptr<Vec> prolong_ans;
    std::shared_ptr<Vec> prolong_applyans;
    std::shared_ptr<Vec> fine_x;
    std::unique_ptr<typename MgLevel::Factory> uniform_coarsening_factory;
    std::unique_ptr<MgLevel> mg_level;
};

TYPED_TEST_SUITE(UniformCoarsening, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(UniformCoarsening, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy =
        this->uniform_coarsening_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->mg_level.get());
    auto copy_mtx = copy->get_system_matrix();
    auto copy_coarse_rows = copy->get_const_coarse_rows();
    auto copy_coarse = copy->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_coarse_rows(copy_coarse_rows,
                                  this->coarse_rows.get_data(),
                                  this->coarse_rows.get_size());
    this->assert_same_matrices(static_cast<const Mtx*>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(UniformCoarsening, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy =
        this->uniform_coarsening_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->mg_level);
    auto copy_mtx = copy->get_system_matrix();
    auto copy_coarse_rows = copy->get_const_coarse_rows();
    auto copy_coarse = copy->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_coarse_rows(copy_coarse_rows,
                                  this->coarse_rows.get_data(),
                                  this->coarse_rows.get_size());
    this->assert_same_matrices(static_cast<const Mtx*>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(UniformCoarsening, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto clone = this->mg_level->clone();
    auto clone_mtx = clone->get_system_matrix();
    auto clone_coarse_rows = clone->get_const_coarse_rows();
    auto clone_coarse = clone->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx*>(clone_mtx.get()),
                               this->mtx.get());
    this->assert_same_coarse_rows(clone_coarse_rows,
                                  this->coarse_rows.get_data(),
                                  this->coarse_rows.get_size());
    this->assert_same_matrices(static_cast<const Mtx*>(clone_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(UniformCoarsening, CanBeCleared)
{
    using MgLevel = typename TestFixture::MgLevel;

    this->mg_level->clear();
    auto mtx = this->mg_level->get_system_matrix();
    auto coarse = this->mg_level->get_coarse_op();
    auto sel = this->mg_level->get_coarse_rows();

    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
    ASSERT_EQ(sel, nullptr);
}


TYPED_TEST(UniformCoarsening, Generate)
{
    auto coarse_fine = this->uniform_coarsening_factory->generate(this->mtx);

    auto sel_result = coarse_fine->get_const_coarse_rows();

    ASSERT_EQ(sel_result[0], 0);
    ASSERT_EQ(sel_result[1], -1);
    ASSERT_EQ(sel_result[2], 1);
    ASSERT_EQ(sel_result[3], -1);
    ASSERT_EQ(sel_result[4], 2);
}


TYPED_TEST(UniformCoarsening, FillIncrementalIndicesWorks)
{
    using index_type = typename TestFixture::index_type;
    auto c2_rows =
        gko::array<index_type>(this->exec, {0, -1, 1, -1, 2, -1, 3, -1, 4, -1});
    auto c3_rows = gko::array<index_type>(this->exec,
                                          {0, -1, -1, 1, -1, -1, 2, -1, -1, 3});
    auto c4_rows = gko::array<index_type>(
        this->exec, {0, -1, -1, -1, 1, -1, -1, -1, 2, -1});
    auto c5_rows = gko::array<index_type>(
        this->exec, {0, -1, -1, -1, -1, 1, -1, -1, -1, -1});
    auto c_rows = gko::array<index_type>(this->exec, 10);
    c_rows.fill(-gko::one<index_type>());

    gko::kernels::reference::uniform_coarsening::fill_incremental_indices(
        this->exec, 2, &c_rows);
    GKO_ASSERT_ARRAY_EQ(c_rows, c2_rows);

    c_rows.fill(-gko::one<index_type>());
    gko::kernels::reference::uniform_coarsening::fill_incremental_indices(
        this->exec, 3, &c_rows);
    GKO_ASSERT_ARRAY_EQ(c_rows, c3_rows);

    c_rows.fill(-gko::one<index_type>());
    gko::kernels::reference::uniform_coarsening::fill_incremental_indices(
        this->exec, 4, &c_rows);
    GKO_ASSERT_ARRAY_EQ(c_rows, c4_rows);

    c_rows.fill(-gko::one<index_type>());
    gko::kernels::reference::uniform_coarsening::fill_incremental_indices(
        this->exec, 5, &c_rows);
    GKO_ASSERT_ARRAY_EQ(c_rows, c5_rows);
}


TYPED_TEST(UniformCoarsening, CoarseFineRestrictApply)
{
    auto uniform_coarsening =
        this->uniform_coarsening_factory->generate(this->mtx);
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(this->coarse_b);

    uniform_coarsening->get_restrict_op()->apply(this->fine_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans, r<value_type>::value);
}


TYPED_TEST(UniformCoarsening, CoarseFineProlongApplyadd)
{
    using value_type = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto uniform_coarsening =
        this->uniform_coarsening_factory->generate(this->mtx);
    auto one = gko::initialize<Vec>({value_type{1.0}}, this->exec);
    auto x = gko::clone(this->fine_x);

    uniform_coarsening->get_prolong_op()->apply(one.get(), this->coarse_b.get(),
                                                one.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->prolong_ans, r<value_type>::value);
}


TYPED_TEST(UniformCoarsening, CoarseFineProlongApply)
{
    using value_type = typename TestFixture::value_type;
    auto uniform_coarsening =
        this->uniform_coarsening_factory->generate(this->mtx);
    auto x = gko::clone(this->fine_x);

    uniform_coarsening->get_prolong_op()->apply(this->coarse_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->prolong_applyans, r<value_type>::value);
}


TYPED_TEST(UniformCoarsening, Apply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto uniform_coarsening =
        this->uniform_coarsening_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = uniform_coarsening->get_executor();
    auto answer = gko::initialize<Vec>(
        {I<VT>({-7.0, -2.0}), I<VT>({0.0, 0.0}), I<VT>({1.0, -4.0}),
         I<VT>({0.0, 0.0}), I<VT>({2.0, 12.0})},
        exec);

    uniform_coarsening->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(UniformCoarsening, AdvancedApply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto uniform_coarsening =
        this->uniform_coarsening_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = uniform_coarsening->get_executor();
    auto alpha = gko::initialize<Vec>({1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto answer = gko::initialize<Vec>(
        {I<VT>({-11.0, -4.0}), I<VT>({2.0, -2.0}), I<VT>({-1.0, -6.0}),
         I<VT>({0.0, 0.0}), I<VT>({2.0, 16.0})},
        exec);

    uniform_coarsening->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(UniformCoarsening, GenerateMgLevel)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 3}, 0));
    // 0-2-4
    prolong_op->read({{5, 3}, {{0, 0, 1}, {2, 1, 1}, {4, 2, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = this->uniform_coarsening_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_prolong_op()), prolong_op,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
}


TYPED_TEST(UniformCoarsening, GenerateMgLevelOnUnsortedMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto mglevel_sort = MgLevel::build().with_coarse_skip(2u).on(this->exec);
    /* this unsorted matrix is stored as this->fine:
     *  5 -3 -3  0  0
     * -3  5  0 -2 -1
     * -3  0  5  0 -1
     *  0 -3  0  5  0
     *  0 -2 -2  0  5
     */
    auto mtx_values = {-3, -3, 5, -3, -2, -1, 5, -3, -1, 5, 5, -3, -2, -2, 5};
    auto mtx_col_idxs = {1, 2, 0, 0, 3, 4, 1, 0, 4, 2, 1, 3, 1, 2, 4};
    auto mtx_row_ptrs = {0, 3, 7, 10, 12, 15};
    auto matrix = gko::share(
        Mtx::create(this->exec, gko::dim<2>{5, 5}, std::move(mtx_values),
                    std::move(mtx_col_idxs), std::move(mtx_row_ptrs)));
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 3}, 0));
    // 0-2-4
    prolong_op->read({{5, 3}, {{0, 0, 1}, {2, 1, 1}, {4, 2, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = mglevel_sort->generate(matrix);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_prolong_op()), prolong_op,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
}


}  // namespace
