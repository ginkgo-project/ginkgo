// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class Jacobi : public CommonTestFixture {
protected:
    using Bj = gko::preconditioner::Jacobi<>;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using mtx_data = gko::matrix_data<>;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;

    void initialize_data(
        std::initializer_list<gko::int32> block_pointers,
        std::initializer_list<gko::precision_reduction> block_precisions,
        std::initializer_list<double> condition_numbers,
        gko::uint32 max_block_size, int min_nnz, int max_nnz, int num_rhs = 1,
        double accuracy = 0.1, bool skip_sorting = true)
    {
        std::default_random_engine engine(42);
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
        gko::array<gko::int32> block_ptrs(ref, block_pointers);
        gko::array<gko::precision_reduction> block_prec(ref, block_precisions);
        if (block_prec.get_size() == 0) {
            bj_factory =
                Bj::build()
                    .with_max_block_size(max_block_size)
                    .with_block_pointers(block_ptrs)
#ifdef GKO_COMPILING_HIP
                    .with_max_block_stride(gko::uint32(exec->get_warp_size()))
#endif
                    .with_skip_sorting(skip_sorting)
                    .on(ref);
            d_bj_factory = Bj::build()
                               .with_max_block_size(max_block_size)
                               .with_block_pointers(block_ptrs)
                               .with_skip_sorting(skip_sorting)
                               .on(exec);
        } else {
            bj_factory =
                Bj::build()
                    .with_max_block_size(max_block_size)
                    .with_block_pointers(block_ptrs)
#ifdef GKO_COMPILING_HIP
                    .with_max_block_stride(gko::uint32(exec->get_warp_size()))
#endif
                    .with_storage_optimization(block_prec)
                    .with_accuracy(accuracy)
                    .with_skip_sorting(skip_sorting)
                    .on(ref);
            d_bj_factory = Bj::build()
                               .with_max_block_size(max_block_size)
                               .with_block_pointers(block_ptrs)
                               .with_storage_optimization(block_prec)
                               .with_accuracy(accuracy)
                               .with_skip_sorting(skip_sorting)
                               .on(exec);
        }
        b = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_b = gko::clone(exec, b);
        x = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_x = gko::clone(exec, x);
    }

    const gko::precision_reduction dp{};
    const gko::precision_reduction sp{0, 1};
    const gko::precision_reduction hp{0, 2};
    const gko::precision_reduction tp{1, 0};
    const gko::precision_reduction qp{2, 0};
    const gko::precision_reduction up{1, 1};
    const gko::precision_reduction ap{gko::precision_reduction::autodetect()};

    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Vec> x;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> d_x;
    std::unique_ptr<Vec> d_b;

    std::unique_ptr<Bj::Factory> bj_factory;
    std::unique_ptr<Bj::Factory> d_bj_factory;
};


TEST_F(Jacobi, FindNaturalBlocksEquivalentToRef)
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
    auto d_bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, ExecutesSupervariableAgglomerationEquivalentToRef)
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
    auto d_bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, FindNaturalBlocksInLargeMatrixEquivalentToRef)
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
    auto d_bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, ExecutesSupervariableAgglomerationInLargeMatrixEquivalentToRef)
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
    auto d_bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi,
       ExecutesSupervarAgglomerationEquivalentToRefFor150NonzerowsPerRow)
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
    auto d_bj = Bj::build().with_max_block_size(3u).on(exec)->generate(mtx);

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    // TODO: actually check if the results are the same
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithBlockSize32Sorted)
{
    initialize_data({0, 32, 64, 96, 128}, {}, {}, 32, 100, 110);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-13);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithBlockSize32Unsorted)
{
    std::default_random_engine engine(42);
    initialize_data({0, 32, 64, 96, 128}, {}, {}, 32, 100, 110, 1, 0.1, false);
    gko::test::unsort_matrix(mtx.get(), engine);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-13);
}


#ifdef GKO_COMPILING_HIP
TEST_F(Jacobi, PreconditionerEquivalentToRefWithBlockSize64)
{
    if (exec->get_warp_size() == 32) {
        GTEST_SKIP();
    }
    initialize_data({0, 64, 128, 192, 256}, {}, {}, 64, 100, 110);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-13);
}
#endif  // GKO_COMPILING_HIP


TEST_F(Jacobi, PreconditionerEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 32,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-13);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithMPW)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-13);
}


TEST_F(Jacobi, TransposedPreconditionerEquivalentToRefWithMPW)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);
    d_bj->copy_from(bj);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj->transpose()),
                        gko::as<Bj>(bj->transpose()), 1e-14);
}


TEST_F(Jacobi, ConjTransposedPreconditionerEquivalentToRefWithMPW)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);
    d_bj->copy_from(bj);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj->conj_transpose()),
                        gko::as<Bj>(bj->conj_transpose()), 1e-14);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithBlockSize32)
{
    initialize_data({0, 32, 64, 96, 128}, {}, {}, 32, 100, 111);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


#ifdef GKO_COMPILING_HIP
TEST_F(Jacobi, ApplyEquivalentToRefWithBlockSize64)
{
    if (exec->get_warp_size() == 32) {
        GTEST_SKIP();
    }
    initialize_data({0, 64, 128, 192, 256}, {}, {}, 64, 100, 111);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}
#endif  // GKO_COMPILING_HIP


TEST_F(Jacobi, ApplyEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 32,
                    97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ScalarApplyEquivalentToRef)
{
    gko::size_type dim = 313;
    std::default_random_engine engine(42);
    auto dense_data =
        gko::test::generate_random_matrix_data<value_type, index_type>(
            dim, dim, std::uniform_int_distribution<>(1, dim),
            std::normal_distribution<>(1.0, 2.0), engine);
    gko::utils::make_diag_dominant(dense_data);
    auto dense_smtx = gko::share(Vec::create(ref));
    dense_smtx->read(dense_data);
    auto smtx = gko::share(Mtx::create(ref));
    smtx->copy_from(dense_smtx);
    auto sb = gko::share(gko::test::generate_random_matrix<Vec>(
        dim, 3, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine, ref));
    auto sx = Vec::create(ref, sb->get_size());

    auto d_smtx = gko::share(Mtx::create(exec));
    auto d_sb = gko::share(Vec::create(exec));
    auto d_sx = gko::share(Vec::create(exec, sb->get_size()));
    d_smtx->copy_from(smtx);
    d_sb->copy_from(sb);

    auto sj = Bj::build().with_max_block_size(1u).on(ref)->generate(smtx);
    auto d_sj = Bj::build().with_max_block_size(1u).on(exec)->generate(d_smtx);

    sj->apply(sb, sx);
    d_sj->apply(d_sb, d_sx);

    GKO_ASSERT_MTX_NEAR(sx, d_sx, 1e-12);
}


TEST_F(Jacobi, LinearCombinationApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha, b, beta, x);
    d_bj->apply(d_alpha, d_b, d_beta, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ScalarLinearCombinationApplyEquivalentToRef)
{
    gko::size_type dim = 313;
    std::default_random_engine engine(42);
    auto dense_data =
        gko::test::generate_random_matrix_data<value_type, index_type>(
            dim, dim, std::uniform_int_distribution<>(1, dim),
            std::normal_distribution<>(1.0, 2.0), engine);
    gko::utils::make_diag_dominant(dense_data);
    auto dense_smtx = gko::share(Vec::create(ref));
    dense_smtx->read(dense_data);
    auto smtx = gko::share(Mtx::create(ref));
    smtx->copy_from(dense_smtx);
    auto sb = gko::share(gko::test::generate_random_matrix<Vec>(
        dim, 3, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine, ref, gko::dim<2>(dim, 3),
        4));
    auto sx = gko::share(gko::test::generate_random_matrix<Vec>(
        dim, 3, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine, ref, gko::dim<2>(dim, 3),
        4));

    auto d_smtx = gko::share(gko::clone(exec, smtx));
    auto d_sb = gko::share(gko::clone(exec, sb));
    auto d_sx = gko::share(gko::clone(exec, sx));
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, exec);

    auto sj = Bj::build().with_max_block_size(1u).on(ref)->generate(smtx);
    auto d_sj = Bj::build().with_max_block_size(1u).on(exec)->generate(d_smtx);

    sj->apply(alpha, sb, beta, sx);
    d_sj->apply(d_alpha, d_sb, d_beta, d_sx);

    GKO_ASSERT_MTX_NEAR(sx, d_sx, 1e-12);
}


TEST_F(Jacobi, ApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, LinearCombinationApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, {}, {}, 13,
                    97, 99, 5);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha, b, beta, x);
    d_bj->apply(d_alpha, d_b, d_beta, d_x);

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
                    1e-9);
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

    GKO_ASSERT_ARRAY_EQ(bj->get_parameters().storage_optimization.block_wise,
                        d_bj->get_parameters().storage_optimization.block_wise);
}


TEST_F(Jacobi, AvoidsPrecisionsThatOverflow)
{
    auto mtx = gko::matrix::Csr<>::create(exec);
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
            .with_block_pointers(gko::array<gko::int32>(exec, {0, 2, 4}))
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .with_accuracy(0.1)
            .on(exec)
            ->generate(give(mtx));

    // both blocks are in the same group, both need (7, 8)
    auto h_bj = clone(ref, bj);
    auto prec =
        h_bj->get_parameters().storage_optimization.block_wise.get_const_data();
    EXPECT_EQ(prec[0], gko::precision_reduction(1, 1));
    ASSERT_EQ(prec[1], gko::precision_reduction(1, 1));
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-13);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-7);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithCustomReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {tp, tp, tp, tp, tp, tp, tp, tp, tp, tp, tp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-6);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {hp, hp, hp, hp, hp, hp, hp, hp, hp, hp, hp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-3);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithCustomQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {qp, qp, qp, qp, qp, qp, qp, qp, qp, qp, qp}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-1);
}


TEST_F(Jacobi, PreconditionerEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    GKO_ASSERT_MTX_NEAR(d_bj, bj, 1e-1);
}


TEST_F(Jacobi, TransposedPreconditionerEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);
    d_bj->copy_from(bj);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj->transpose()),
                        gko::as<Bj>(bj->transpose()), 0);
}


TEST_F(Jacobi, ConjTransposedPreconditionerEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);
    d_bj->copy_from(bj);

    GKO_ASSERT_MTX_NEAR(gko::as<Bj>(d_bj->conj_transpose()),
                        gko::as<Bj>(bj->conj_transpose()), 0);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithCustomReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {tp, tp, tp, tp, tp, tp, tp, tp, tp, tp, tp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-5);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {hp, hp, hp, hp, hp, hp, hp, hp, hp, hp, hp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-2);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithCustomReducedAndReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {up, up, up, up, up, up, up, up, up, up, up}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-2);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithCustomQuarteredPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {qp, qp, qp, qp, qp, qp, qp, qp, qp, qp, qp}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, ApplyEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-1);
}


TEST_F(Jacobi, LinearCombinationApplyEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, dp, dp, sp, sp, sp, dp, dp, sp, dp, sp}, {}, 13, 97,
                    99);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, ApplyToMultipleVectorsEquivalentToRefWithFullPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {dp, dp, dp, dp, dp, dp, dp, dp, dp, dp, dp}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(Jacobi, ApplyToMultipleVectorsEquivalentToRefWithReducedPrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(Jacobi, ApplyToMultipleVectorsEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, sp, dp, dp, tp, tp, qp, qp, hp, dp, up}, {}, 13, 97,
                    99, 5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-1);
}


TEST_F(
    Jacobi,
    LinearCombinationApplyToMultipleVectorsEquivalentToRefWithAdaptivePrecision)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100},
                    {sp, dp, dp, sp, sp, sp, dp, dp, sp, dp, sp}, {}, 13, 97,
                    99, 5);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, exec);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b, x);
    d_bj->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}
