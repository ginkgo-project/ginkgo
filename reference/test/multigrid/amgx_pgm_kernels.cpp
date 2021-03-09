/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/multigrid/amgx_pgm_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class AmgxPgm : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::AmgxPgm<value_type, index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    using WeightMtx = gko::matrix::Csr<real_type, index_type>;
    AmgxPgm()
        : exec(gko::ReferenceExecutor::create()),
          amgxpgm_factory(MgLevel::build()
                              .with_max_iterations(2u)
                              .with_max_unassigned_ratio(0.1)
                              .on(exec)),
          fine_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({-1.0, 2.0}), I<VT>({0.0, -1.0}),
               I<VT>({3.0, -2.0}), I<VT>({-2.0, 1.0})},
              exec)),
          coarse_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0})}, exec)),
          restrict_ans((gko::initialize<Vec>(
              {I<VT>({0.0, -1.0}), I<VT>({2.0, 0.0})}, exec))),
          prolong_ans(gko::initialize<Vec>(
              {I<VT>({0.0, -2.0}), I<VT>({1.0, -2.0}), I<VT>({1.0, -2.0}),
               I<VT>({0.0, -1.0}), I<VT>({2.0, 1.0})},
              exec)),
          prolong_applyans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0}), I<VT>({2.0, -1.0}),
               I<VT>({0.0, -1.0}), I<VT>({2.0, -1.0})},
              exec)),
          fine_x(gko::initialize<Vec>(
              {I<VT>({-2.0, -1.0}), I<VT>({1.0, -1.0}), I<VT>({-1.0, -1.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, 2.0})},
              exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          weight(WeightMtx::create(
              exec, gko::dim<2>(5, 5), 15,
              std::make_shared<typename WeightMtx::classical>())),
          coarse(Mtx::create(exec, gko::dim<2>(2, 2), 4,
                             std::make_shared<typename Mtx::classical>())),
          agg(exec, 5)
    {
        this->create_mtx(mtx.get(), weight.get(), &agg, coarse.get());
        mg_level = amgxpgm_factory->generate(mtx);
        mtx_diag = weight->extract_diagonal();
    }

    void create_mtx(Mtx *fine, WeightMtx *weight, gko::Array<index_type> *agg,
                    Mtx *coarse)
    {
        auto agg_val = agg->get_data();
        agg_val[0] = 0;
        agg_val[1] = 1;
        agg_val[2] = 0;
        agg_val[3] = 1;
        agg_val[4] = 0;

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

        /* weight matrix is stored:
         * 5   3   3   0   0
         * 3   5   0   2.5 1.5
         * 3   0   5   0   1.5
         * 0   2.5 0   5   0
         * 0   1.5 1.5 0   5
         */
        weight->read({{5, 5},
                      {{0, 0, 5.0},
                       {0, 1, 3.0},
                       {0, 2, 3.0},
                       {1, 0, 3.0},
                       {1, 1, 5.0},
                       {1, 3, 2.5},
                       {1, 4, 1.5},
                       {2, 0, 3.0},
                       {2, 2, 5.0},
                       {2, 4, 1.5},
                       {3, 1, 2.5},
                       {3, 3, 5.0},
                       {4, 1, 1.5},
                       {4, 2, 1.5},
                       {4, 4, 5.0}}});

        /* this coarse is stored:
         *  6 -5
         * -4  5
         */
        coarse->read({{2, 2}, {{0, 0, 6}, {0, 1, -5}, {1, 0, -4}, {1, 1, 5}}});
    }

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
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

    static void assert_same_agg(const index_type *m1, const index_type *m2,
                                gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    std::shared_ptr<WeightMtx> weight;
    std::shared_ptr<gko::matrix::Diagonal<real_type>> mtx_diag;
    gko::Array<index_type> agg;
    std::shared_ptr<Vec> coarse_b;
    std::shared_ptr<Vec> fine_b;
    std::shared_ptr<Vec> restrict_ans;
    std::shared_ptr<Vec> prolong_ans;
    std::shared_ptr<Vec> prolong_applyans;
    std::shared_ptr<Vec> fine_x;
    std::unique_ptr<typename MgLevel::Factory> amgxpgm_factory;
    std::unique_ptr<MgLevel> mg_level;
};

TYPED_TEST_SUITE(AmgxPgm, gko::test::ValueIndexTypes);


TYPED_TEST(AmgxPgm, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->mg_level.get());
    auto copy_mtx = copy->get_system_matrix();
    auto copy_agg = copy->get_const_agg();
    auto copy_coarse = copy->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(copy_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->mg_level));
    auto copy_mtx = copy->get_system_matrix();
    auto copy_agg = copy->get_const_agg();
    auto copy_coarse = copy->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(copy_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto clone = this->mg_level->clone();
    auto clone_mtx = clone->get_system_matrix();
    auto clone_agg = clone->get_const_agg();
    auto clone_coarse = clone->get_coarse_op();

    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(clone_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(clone_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCleared)
{
    using MgLevel = typename TestFixture::MgLevel;

    this->mg_level->clear();
    auto mtx = this->mg_level->get_system_matrix();
    auto coarse = this->mg_level->get_coarse_op();
    auto agg = this->mg_level->get_agg();

    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
    ASSERT_EQ(agg, nullptr);
}


TYPED_TEST(AmgxPgm, MatchEdge)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    gko::Array<index_type> snb(this->exec, 5);
    auto agg_val = agg.get_data();
    auto snb_val = snb.get_data();
    for (int i = 0; i < 5; i++) {
        agg_val[i] = -1;
    }
    snb_val[0] = 2;
    snb_val[1] = 0;
    snb_val[2] = 0;
    snb_val[3] = 1;
    snb_val[4] = 2;

    gko::kernels::reference::amgx_pgm::match_edge(this->exec, snb, agg);

    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], -1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], -1);
    ASSERT_EQ(agg_val[4], -1);
}


TYPED_TEST(AmgxPgm, CountUnagg)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    index_type num_unagg = 0;
    agg_val[0] = 0;
    agg_val[1] = -1;
    agg_val[2] = 0;
    agg_val[3] = -1;
    agg_val[4] = -1;

    gko::kernels::reference::amgx_pgm::count_unagg(this->exec, agg, &num_unagg);

    ASSERT_EQ(num_unagg, 3);
}


TYPED_TEST(AmgxPgm, Renumber)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    index_type num_agg = 0;
    agg_val[0] = 0;
    agg_val[1] = 1;
    agg_val[2] = 0;
    agg_val[3] = 1;
    agg_val[4] = 4;

    gko::kernels::reference::amgx_pgm::renumber(this->exec, agg, &num_agg);

    ASSERT_EQ(num_agg, 3);
    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], 1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], 1);
    ASSERT_EQ(agg_val[4], 2);
}


TYPED_TEST(AmgxPgm, Generate)
{
    auto coarse_fine = this->amgxpgm_factory->generate(this->mtx);

    auto agg_result = coarse_fine->get_const_agg();

    ASSERT_EQ(agg_result[0], 0);
    ASSERT_EQ(agg_result[1], 1);
    ASSERT_EQ(agg_result[2], 0);
    ASSERT_EQ(agg_result[3], 1);
    ASSERT_EQ(agg_result[4], 0);
}


TYPED_TEST(AmgxPgm, CoarseFineRestrictApply)
{
    auto amgx_pgm = this->amgxpgm_factory->generate(this->mtx);
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(gko::lend(this->coarse_b));

    amgx_pgm->get_restrict_op()->apply(this->fine_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans, r<value_type>::value);
}


TYPED_TEST(AmgxPgm, CoarseFineProlongApplyadd)
{
    using value_type = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto amgx_pgm = this->amgxpgm_factory->generate(this->mtx);
    auto one = gko::initialize<Vec>({value_type{1.0}}, this->exec);
    auto x = gko::clone(this->fine_x);

    amgx_pgm->get_prolong_op()->apply(one.get(), this->coarse_b.get(),
                                      one.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->prolong_ans, r<value_type>::value);
}


TYPED_TEST(AmgxPgm, CoarseFineProlongApply)
{
    using value_type = typename TestFixture::value_type;
    auto amgx_pgm = this->amgxpgm_factory->generate(this->mtx);
    auto x = gko::clone(this->fine_x);

    amgx_pgm->get_prolong_op()->apply(this->coarse_b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, this->prolong_applyans, r<value_type>::value);
}


TYPED_TEST(AmgxPgm, Apply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto amgx_pgm = this->amgxpgm_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = amgx_pgm->get_executor();
    auto answer = gko::initialize<Vec>(
        {I<VT>({-23.0, 5.0}), I<VT>({17.0, -5.0}), I<VT>({-23.0, 5.0}),
         I<VT>({17.0, -5.0}), I<VT>({-23.0, 5.0})},
        exec);

    amgx_pgm->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(AmgxPgm, AdvancedApply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto amgx_pgm = this->amgxpgm_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = amgx_pgm->get_executor();
    auto alpha = gko::initialize<Vec>({1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto answer = gko::initialize<Vec>(
        {I<VT>({-27.0, 3.0}), I<VT>({19.0, -7.0}), I<VT>({-25.0, 3.0}),
         I<VT>({17.0, -5.0}), I<VT>({-23.0, 9.0})},
        exec);

    amgx_pgm->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(AmgxPgm, FindStrongestNeighbor)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> strongest_neighbor(this->exec, 5);
    gko::Array<index_type> agg(this->exec, 5);
    auto snb_vals = strongest_neighbor.get_data();
    auto agg_vals = agg.get_data();
    for (int i = 0; i < 5; i++) {
        snb_vals[i] = -1;
        agg_vals[i] = -1;
    }

    gko::kernels::reference::amgx_pgm::find_strongest_neighbor(
        this->exec, this->weight.get(), this->mtx_diag.get(), agg,
        strongest_neighbor);

    ASSERT_EQ(snb_vals[0], 2);
    ASSERT_EQ(snb_vals[1], 0);
    ASSERT_EQ(snb_vals[2], 0);
    ASSERT_EQ(snb_vals[3], 1);
    ASSERT_EQ(snb_vals[4], 2);
}


TYPED_TEST(AmgxPgm, AssignToExistAgg)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> agg(this->exec, 5);
    gko::Array<index_type> intermediate_agg(this->exec, 0);
    auto agg_vals = agg.get_data();
    // 0 - 2, 1 - 3
    agg_vals[0] = 0;
    agg_vals[1] = 1;
    agg_vals[2] = 0;
    agg_vals[3] = 1;
    agg_vals[4] = -1;

    gko::kernels::reference::amgx_pgm::assign_to_exist_agg(
        this->exec, this->weight.get(), this->mtx_diag.get(), agg,
        intermediate_agg);

    ASSERT_EQ(agg_vals[0], 0);
    ASSERT_EQ(agg_vals[1], 1);
    ASSERT_EQ(agg_vals[2], 0);
    ASSERT_EQ(agg_vals[3], 1);
    ASSERT_EQ(agg_vals[4], 0);
}


TYPED_TEST(AmgxPgm, GenerateMtx)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using mtx_type = typename TestFixture::Mtx;
    gko::Array<index_type> agg(this->exec, 5);
    auto agg_vals = agg.get_data();
    // 0 - 2, 1 - 3, 4
    agg_vals[0] = 0;
    agg_vals[1] = 1;
    agg_vals[2] = 0;
    agg_vals[3] = 1;
    agg_vals[4] = 2;
    auto coarse_ans = mtx_type::create(this->exec, gko::dim<2>{3, 3}, 0);
    coarse_ans->read({{3, 3},
                      {{0, 0, 4},
                       {0, 1, -3},
                       {0, 2, -1},
                       {1, 0, -3},
                       {1, 1, 5},
                       {1, 2, -1},
                       {2, 0, -2},
                       {2, 1, -2},
                       {2, 2, 5}}});
    auto csr_coarse = mtx_type::create(this->exec, gko::dim<2>{3, 3}, 0);

    gko::kernels::reference::amgx_pgm::amgx_pgm_generate(
        this->exec, this->mtx.get(), agg, csr_coarse.get());

    GKO_ASSERT_MTX_NEAR(csr_coarse, coarse_ans, r<value_type>::value);
}


}  // namespace
