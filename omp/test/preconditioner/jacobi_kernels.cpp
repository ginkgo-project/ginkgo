/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <gtest/gtest.h>


#include <random>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


class Jacobi : public ::testing::Test {
protected:
    using Bj = gko::preconditioner::Jacobi<>;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using mtx_data = gko::matrix_data<>;

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    void initialize_data(
        std::initializer_list<gko::int32> block_pointers,
        std::initializer_list<gko::precision_reduction> block_precisions,
        std::initializer_list<double> condition_numbers,
        gko::uint32 max_block_size, int min_nnz, int max_nnz, int num_rhs = 1,
        double accuracy = 0.1)
    {
        std::ranlux48 engine(42);
        const auto dim = *(end(block_pointers) - 1);
        if (condition_numbers.size() == 0) {
            mtx = gko::test::generate_random_matrix<Mtx>(
                dim, dim, std::uniform_int_distribution<>(min_nnz, max_nnz),
                std::normal_distribution<>(0.0, 1.0), engine, ref);
        } else {
            std::vector<mtx_data> blocks;
            for (gko::size_type i = 0; i < block_pointers.size() - 1; ++i) {
                const auto size =
                    begin(block_pointers)[i + 1] - begin(block_pointers)[i];
                const auto cond = begin(condition_numbers)[i];
                blocks.push_back(mtx_data::cond(
                    size, cond, std::normal_distribution<>(-1, 1), engine));
            }
            mtx = Mtx::create(ref);
            mtx->read(mtx_data::diag(begin(blocks), end(blocks)));
        }
        gko::Array<gko::int32> block_ptrs(ref, block_pointers);
        gko::Array<gko::precision_reduction> block_prec(ref, block_precisions);
        if (block_prec.get_num_elems() == 0) {
            bj_factory = Bj::build()
                             .with_max_block_size(max_block_size)
                             .with_block_pointers(block_ptrs)
                             .on(ref);
            d_bj_factory = Bj::build()
                               .with_max_block_size(max_block_size)
                               .with_block_pointers(block_ptrs)
                               .on(omp);
        } else {
            bj_factory = Bj::build()
                             .with_max_block_size(max_block_size)
                             .with_block_pointers(block_ptrs)
                             .with_storage_optimization(block_prec)
                             .with_accuracy(accuracy)
                             .on(ref);
            d_bj_factory = Bj::build()
                               .with_max_block_size(max_block_size)
                               .with_block_pointers(block_ptrs)
                               .with_storage_optimization(block_prec)
                               .with_accuracy(accuracy)
                               .on(omp);
        }
        b = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_b = Vec::create(omp);
        d_b->copy_from(b.get());
        x = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_x = Vec::create(omp);
        d_x->copy_from(x.get());
    }

    const gko::precision_reduction dp{};
    const gko::precision_reduction sp{0, 1};
    const gko::precision_reduction hp{0, 2};
    const gko::precision_reduction tp{1, 0};
    const gko::precision_reduction qp{2, 0};
    const gko::precision_reduction up{1, 1};
    const gko::precision_reduction ap{gko::precision_reduction::autodetect()};

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Vec> x;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> d_x;
    std::unique_ptr<Vec> d_b;

    std::unique_ptr<Bj::Factory> bj_factory;
    std::unique_ptr<Bj::Factory> d_bj_factory;
};


TEST_F(Jacobi, OmpFindNaturalBlocksEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
        1       1
        1       1
     */
    auto mtx = share(Mtx::create(ref));
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, 1.0},
                {1, 0, 1.0},
                {1, 1, 1.0},
                {2, 0, 1.0},
                {2, 2, 1.0},
                {3, 0, 1.0},
                {3, 2, 1.0}}});

    auto bj = Bj::build().with_max_block_size(3u).on(ref)->generate(mtx);
    auto d_bj = Bj::build().with_max_block_size(3u).on(omp)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, OmpExecutesSupervariableAgglomerationEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    auto mtx = share(Mtx::create(ref));
    mtx->read({{5, 5},
               {{0, 0, 1.0},
                {0, 1, 1.0},
                {1, 0, 1.0},
                {1, 1, 1.0},
                {2, 2, 1.0},
                {2, 3, 1.0},
                {3, 2, 1.0},
                {3, 3, 1.0},
                {4, 4, 1.0}}});

    auto bj = Bj::build().with_max_block_size(3u).on(ref)->generate(mtx);
    auto d_bj = Bj::build().with_max_block_size(3u).on(omp)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, OmpFindNaturalBlocksInLargeMatrixEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
        1       1
        1       1
        1       1
        1       1
     */
    using data = gko::matrix_data<double, int>;
    auto mtx = share(Mtx::create(ref));
    mtx->read(data::diag({550, 550}, {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                      {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                      {1.0, 0.0, 1.0, 0.0, 0.0, 0.0}}));

    auto bj = Bj::build().with_max_block_size(3u).on(ref)->generate(mtx);
    auto d_bj = Bj::build().with_max_block_size(3u).on(omp)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi,
       OmpExecutesSupervariableAgglomerationInLargeMatrixEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    using data = gko::matrix_data<double, int>;
    auto mtx = share(Mtx::create(ref));
    mtx->read(data::diag({550, 550}, {{1.0, 1.0, 0.0, 0.0, 0.0},
                                      {1.0, 1.0, 0.0, 0.0, 0.0},
                                      {0.0, 0.0, 1.0, 1.0, 0.0},
                                      {0.0, 0.0, 1.0, 1.0, 0.0},
                                      {0.0, 0.0, 0.0, 0.0, 1.0}}));

    auto bj = Bj::build().with_max_block_size(3u).on(ref)->generate(mtx);
    auto d_bj = Bj::build().with_max_block_size(3u).on(omp)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi,
       OmpExecutesSupervarAgglomerationEquivalentToRefFor150NonzerowsPerRow)
{
    /* example matrix duplicated 50 times:
        1   1       1
        1   1       1
        1       1   1
        1       1   1
                1        1
     */
    using data = gko::matrix_data<double, int>;
    auto mtx = share(Mtx::create(ref));
    mtx->read({{50, 50},
               {{1.0, 1.0, 0.0, 1.0, 0.0},
                {1.0, 1.0, 0.0, 1.0, 0.0},
                {1.0, 0.0, 1.0, 1.0, 0.0},
                {1.0, 0.0, 1.0, 1.0, 0.0},
                {0.0, 0.0, 1.0, 0.0, 1.0}}});


    auto bj = Bj::build().with_max_block_size(3u).on(ref)->generate(mtx);
    auto d_bj = Bj::build().with_max_block_size(3u).on(omp)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithBlockSize32)
{
    initialize_data({0, 32, 64, 96, 128}, {}, {}, 32, 100, 110);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 32,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithMPW)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithBlockSize32)
{
    initialize_data({0, 32, 64, 96, 128}, {}, {}, 32, 100, 111);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 32,
                    97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpLinearCombinationApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, omp);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, omp);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());
    d_bj->apply(d_alpha.get(), d_b.get(), d_beta.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpLinearCombinationApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99, 5);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, omp);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, omp);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());
    d_bj->apply(d_alpha.get(), d_b.get(), d_beta.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ComputesTheSameConditionNumberAsRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = clone(ref, d_bj_factory->generate(mtx));

    for (int i = 0; i < gko::as<Bj>(bj.get())->get_num_blocks(); ++i) {
        EXPECT_NEAR(bj->get_conditioning()[i], d_bj->get_conditioning()[i],
                    1e-11);
    }
}


TEST_F(Jacobi, SelectsTheSamePrecisionsAsRef)
{
    initialize_data(
        {0, 2, 14, 27, 40, 51, 61, 70, 80, 92, 100},
        {ap, ap, ap, ap, ap, ap, ap, ap, ap, ap},
        {1e+0, 1e+0, 1e+2, 1e+3, 1e+4, 1e+4, 1e+6, 1e+7, 1e+8, 1e+9}, 13, 97,
        99, 1, 0.2);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = gko::clone(ref, d_bj_factory->generate(mtx));

    auto bj_prec =
        bj->get_parameters().storage_optimization.block_wise.get_const_data();
    auto d_bj_prec =
        d_bj->get_parameters().storage_optimization.block_wise.get_const_data();
    for (int i = 0; i < gko::as<Bj>(bj.get())->get_num_blocks(); ++i) {
        EXPECT_EQ(bj_prec[i], d_bj_prec[i]);
    }
}


TEST_F(Jacobi, AvoidsPrecisionsThatOverflow)
{
    auto mtx = gko::matrix::Csr<>::create(omp);
    // clang-format off
    mtx->read(mtx_data::diag({
        // perfectly conditioned block, small value difference,
        // can use fp16 (5, 10)
        {{2.0, 1.0},
         {1.0, 2.0}},
        // perfectly conditioned block (scaled orthogonal),
        // with large value difference, need fp16 (7, 8)
        {{1e-8, -1e-16},
         {1e-16,  1e-8}}
    }));
    // clang-format on

    auto bj =
        Bj::build()
            .with_max_block_size(13u)
            .with_block_pointers(gko::Array<gko::int32>(omp, {0, 2, 4}))
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(0.1)
            .on(omp)
            ->generate(give(mtx));

    // both blocks are in the same group, both need (7, 8)
    auto h_bj = clone(ref, bj);
    auto prec =
        h_bj->get_parameters().storage_optimization.block_wise.get_const_data();
    EXPECT_EQ(prec[0], gko::precision_reduction(1, 1));
    ASSERT_EQ(prec[1], gko::precision_reduction(1, 1));
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-14);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-7);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithCustomReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {tp, tp, tp, tp, tp, tp, tp, tp, tp, tp, tp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-6);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {hp, hp, hp, hp, hp, hp, hp, hp, hp, hp, hp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-3);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithCustomQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {qp, qp, qp, qp, qp, qp, qp, qp, qp, qp, qp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-1);
}


TEST_F(Jacobi, OmpPreconditionerEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(lend(d_bj), lend(bj), 1e-1);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithCustomReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {tp, tp, tp, tp, tp, tp, tp, tp, tp, tp, tp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-5);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {hp, hp, hp, hp, hp, hp, hp, hp, hp, hp, hp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-3);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithCustomReducedAndReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {up, up, up, up, up, up, up, up, up, up, up}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-2);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithCustomQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {qp, qp, qp, qp, qp, qp, qp, qp, qp, qp, qp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, OmpApplyEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-1);
}


TEST_F(Jacobi, OmpLinearCombinationApplyEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, dp, dp, sp, sp, sp, dp, dp, sp, dp, sp}, {}, 13, 97,
                    99);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, omp);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, omp);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, OmpApplyToMultipleVectorsEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, OmpApplyToMultipleVectorsEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, OmpApplyToMultipleVectorsEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-1);
}


TEST_F(
    Jacobi,
    OmpLinearCombinationApplyToMultipleVectorsEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, dp, dp, sp, sp, sp, dp, dp, sp, dp, sp}, {}, 13, 97,
                    99, 5);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, omp);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, omp);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


}  // namespace
