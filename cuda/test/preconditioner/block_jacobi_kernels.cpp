/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/preconditioner/block_jacobi.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/matrix/csr.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils.hpp>


namespace {


class BlockJacobi : public ::testing::Test {
protected:
    using BjFactory = gko::preconditioner::BlockJacobiFactory<>;
    using Bj = gko::preconditioner::BlockJacobi<>;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    void initialize_data(std::initializer_list<gko::int32> block_pointers,
                         gko::int32 max_block_size, int min_nnz, int max_nnz,
                         int num_rhs = 1)
    {
        std::ranlux48 engine(42);
        const auto dim = *(end(block_pointers) - 1);
        mtx = gko::test::generate_random_matrix<Mtx>(
            dim, dim, std::uniform_int_distribution<>(min_nnz, max_nnz),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        gko::Array<gko::int32> block_ptrs(ref, block_pointers);
        bj_factory = BjFactory::create(ref, max_block_size);
        bj_factory->set_block_pointers(block_ptrs);
        d_bj_factory = BjFactory::create(cuda, max_block_size);
        d_bj_factory->set_block_pointers(block_ptrs);
        b = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_b = Vec::create(cuda);
        d_b->copy_from(b.get());
        x = gko::test::generate_random_matrix<Vec>(
            dim, num_rhs, std::uniform_int_distribution<>(num_rhs, num_rhs),
            std::normal_distribution<>(0.0, 1.0), engine, ref);
        d_x = Vec::create(cuda);
        d_x->copy_from(x.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Vec> x;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> d_x;
    std::unique_ptr<Vec> d_b;

    std::unique_ptr<BjFactory> bj_factory;
    std::unique_ptr<BjFactory> d_bj_factory;
};


TEST_F(BlockJacobi, CudaFindNaturalBlocksEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
        1       1
        1       1
     */
    auto mtx = Mtx::create(ref);
    mtx->read({{4, 4},
               {{0, 0, 1.0},
                {0, 1, 1.0},
                {1, 0, 1.0},
                {1, 1, 1.0},
                {2, 0, 1.0},
                {2, 2, 1.0},
                {3, 0, 1.0},
                {3, 2, 1.0}}});

    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());

    std::unique_ptr<BjFactory> bj_factory;
    bj_factory = BjFactory::create(ref, 3);
    auto bj_lin_op = bj_factory->generate(std::move(mtx));
    auto bj = static_cast<Bj *>(bj_lin_op.get());
    d_bj_factory = BjFactory::create(cuda, 3);
    auto d_bj_lin_op = d_bj_factory->generate(std::move(d_mtx));
    auto d_bj = static_cast<Bj *>(d_bj_lin_op.get());

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    EXPECT_EQ(d_bj->get_max_block_size(), bj->get_max_block_size());
}


TEST_F(BlockJacobi, CudaExecutesSupervariableAgglomerationEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    auto mtx = Mtx::create(ref);
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

    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());

    std::unique_ptr<BjFactory> bj_factory;
    bj_factory = BjFactory::create(ref, 3);
    auto bj_lin_op = bj_factory->generate(std::move(mtx));
    auto bj = static_cast<Bj *>(bj_lin_op.get());
    d_bj_factory = BjFactory::create(cuda, 3);
    auto d_bj_lin_op = d_bj_factory->generate(std::move(d_mtx));
    auto d_bj = static_cast<Bj *>(d_bj_lin_op.get());

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    EXPECT_EQ(d_bj->get_max_block_size(), bj->get_max_block_size());
}


TEST_F(BlockJacobi, CudaFindNaturalBlocksInLargeMatrixEquivalentToRef)
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
    using nnz = data::nonzero_type;
    auto m =
        data::diag(gko::dim<2>{550, 550}, {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                           {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                           {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                           {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                           {1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                           {1.0, 0.0, 1.0, 0.0, 0.0, 0.0}});
    auto mtx = Mtx::create(ref);
    mtx->read(m);
    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());

    std::unique_ptr<BjFactory> bj_factory;
    bj_factory = BjFactory::create(ref, 3);
    auto bj_lin_op = bj_factory->generate(std::move(mtx));
    auto bj = static_cast<Bj *>(bj_lin_op.get());
    d_bj_factory = BjFactory::create(cuda, 3);
    auto d_bj_lin_op = d_bj_factory->generate(std::move(d_mtx));
    auto d_bj = static_cast<Bj *>(d_bj_lin_op.get());
    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    EXPECT_EQ(d_bj->get_max_block_size(), bj->get_max_block_size());
}


TEST_F(BlockJacobi,
       CudaExecutesSupervariableAgglomerationInLargeMatrixEquivalentToRef)
{
    /* example matrix:
        1   1
        1   1
                1   1
                1   1
                        1
     */
    using data = gko::matrix_data<double, int>;
    using nnz = data::nonzero_type;
    auto m = data::diag(gko::dim<2>{550, 550}, {{1.0, 1.0, 0.0, 0.0, 0.0},
                                                {1.0, 1.0, 0.0, 0.0, 0.0},
                                                {0.0, 0.0, 1.0, 1.0, 0.0},
                                                {0.0, 0.0, 1.0, 1.0, 0.0},
                                                {0.0, 0.0, 0.0, 0.0, 1.0}});
    auto mtx = Mtx::create(ref);
    mtx->read(m);
    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());

    auto bj_factory = BjFactory::create(ref, 3);
    auto bj_lin_op = bj_factory->generate(std::move(mtx));
    auto bj = static_cast<Bj *>(bj_lin_op.get());
    d_bj_factory = BjFactory::create(cuda, 3);
    auto d_bj_lin_op = d_bj_factory->generate(std::move(d_mtx));
    auto d_bj = static_cast<Bj *>(d_bj_lin_op.get());

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    EXPECT_EQ(d_bj->get_max_block_size(), bj->get_max_block_size());
}

TEST_F(BlockJacobi,
       CudaExecutesSupervarAgglomerationEquivalentToRefFor150NonzerowsPerRow)
{
    /* example matrix duplicated 50 times:
        1   1       1
        1   1       1
        1       1   1
        1       1   1
                1        1
     */

    using data = gko::matrix_data<double, int>;
    gko::matrix_data<double, int> m{{1.0, 1.0, 0.0, 1.0, 0.0},
                                    {1.0, 1.0, 0.0, 1.0, 0.0},
                                    {1.0, 0.0, 1.0, 1.0, 0.0},
                                    {1.0, 0.0, 1.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0, 0.0, 1.0}};
    using nnz = data::nonzero_type;
    gko::matrix_data<double, int> mm{{50, 50}, m};

    auto mtx = Mtx::create(ref);
    mtx->read(mm);
    auto d_mtx = Mtx::create(cuda);
    d_mtx->copy_from(mtx.get());

    std::unique_ptr<BjFactory> bj_factory;
    bj_factory = BjFactory::create(ref, 13);
    auto bj_lin_op = bj_factory->generate(std::move(mtx));
    auto bj = static_cast<Bj *>(bj_lin_op.get());
    d_bj_factory = BjFactory::create(cuda, 13);
    auto d_bj_lin_op = d_bj_factory->generate(std::move(d_mtx));
    auto d_bj = static_cast<Bj *>(d_bj_lin_op.get());

    ASSERT_EQ(d_bj->get_num_blocks(), bj->get_num_blocks());
    EXPECT_EQ(d_bj->get_max_block_size(), bj->get_max_block_size());
}


TEST_F(BlockJacobi, CudaPreconditionerEquivalentToRefWithBlockSize32)
{
    initialize_data({0, 32, 64, 96, 128}, 32, 100, 110);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(BlockJacobi, CudaPreconditionerEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 32, 97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(BlockJacobi, CudaPreconditionerEquivalentToRefWithMPW)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 13, 97, 99);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(BlockJacobi, CudaApplyEquivalentToRefWithBlockSize32)
{
    initialize_data({0, 32, 64, 96, 128}, 32, 100, 111);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(BlockJacobi, CudaApplyEquivalentToRefWithDifferentBlockSize)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 32, 97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(BlockJacobi, CudaApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 13, 97, 99);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(BlockJacobi, CudaLinearCombinationApplyEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 13, 97, 99);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, cuda);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, cuda);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());
    d_bj->apply(d_alpha.get(), d_b.get(), d_beta.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(BlockJacobi, CudaApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 13, 97, 99,
                    5);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(b.get(), x.get());
    d_bj->apply(d_b.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


TEST_F(BlockJacobi, CudaLinearCombinationApplyToMultipleVectorsEquivalentToRef)
{
    initialize_data({0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100}, 13, 97, 99,
                    5);
    auto alpha = gko::initialize<Vec>({2.0}, ref);
    auto d_alpha = gko::initialize<Vec>({2.0}, cuda);
    auto beta = gko::initialize<Vec>({-1.0}, ref);
    auto d_beta = gko::initialize<Vec>({-1.0}, cuda);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    bj->apply(alpha.get(), b.get(), beta.get(), x.get());
    d_bj->apply(d_alpha.get(), d_b.get(), d_beta.get(), d_x.get());

    ASSERT_MTX_NEAR(d_x, x, 1e-12);
}


}  // namespace
