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
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::GpuExecutor> gpu;
};


TEST_F(BlockJacobi, GpuPreconditionerEquivalentToRefWithBlockSize32)
{
    std::shared_ptr<Mtx> mtx = gko::test::generate_random_matrix<Mtx>(
        ref, 128, 128, std::uniform_int_distribution<>(100, 110),
        std::normal_distribution<>(0.0, 1.0), std::ranlux48(42));
    gko::Array<gko::int32> block_ptrs(ref, 5);
    gko::test::init_array(block_ptrs.get_data(), {0, 32, 64, 96, 128});
    auto bj_factory = BjFactory::create(ref, 32);
    auto d_bj_factory = BjFactory::create(gpu, 32);
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(BlockJacobi, GpuPreconditionerEquivalentToRefWithDifferentBlockSize)
{
    std::shared_ptr<Mtx> mtx = gko::test::generate_random_matrix<Mtx>(
        ref, 100, 100, std::uniform_int_distribution<>(97, 99),
        std::normal_distribution<>(0.0, 1.0), std::ranlux48(42));
    gko::Array<gko::int32> block_ptrs(ref, 11);
    gko::test::init_array(block_ptrs.get_data(),
                          {0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100});
    auto bj_factory = BjFactory::create(ref, 32);
    auto d_bj_factory = BjFactory::create(gpu, 32);
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


TEST_F(BlockJacobi, GpuPreconditionerEquivalentToRefWithMPW)
{
    std::shared_ptr<Mtx> mtx = gko::test::generate_random_matrix<Mtx>(
        ref, 100, 100, std::uniform_int_distribution<>(97, 99),
        std::normal_distribution<>(0.0, 1.0), std::ranlux48(42));
    gko::Array<gko::int32> block_ptrs(ref, 11);
    gko::test::init_array(block_ptrs.get_data(),
                          {0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100});
    auto bj_factory = BjFactory::create(ref, 13);
    auto d_bj_factory = BjFactory::create(gpu, 13);
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);

    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(mtx);

    ASSERT_MTX_NEAR(gko::as<Bj>(d_bj.get()), gko::as<Bj>(bj.get()), 1e-14);
}


/*

TEST_F(BlockJacobi, GpuApplyEquivalentToRefWithBlockSize32)
{
    std::ranlux48 engine(42);
    auto tmp_mtx = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 128, 128, std::uniform_int_distribution<>(100, 120),
        std::normal_distribution<>(0.0, 1.0), engine);
    std::shared_ptr<Mtx> mtx = gko::test::convert_to<Mtx>(tmp_mtx.get());
    std::shared_ptr<Mtx> d_mtx = Mtx::create(gpu);
    d_mtx->copy_from(mtx.get());
    auto bj_factory = BjFactory::create(ref, 32);
    auto d_bj_factory = BjFactory::create(gpu, 32);
    gko::Array<gko::int32> block_ptrs(ref, 5);
    gko::test::init_array(block_ptrs.get_data(), {0, 32, 64, 96, 128});
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(d_mtx);
    auto b = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 128, 1, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine);
    auto d_b = gko::matrix::Dense<>::create(gpu);
    d_b->copy_from(b.get());
    auto expected = gko::matrix::Dense<>::create(ref, 128, 1, 1);
    auto d_x = gko::matrix::Dense<>::create(gpu, 128, 1, 1);

    bj->apply(b.get(), expected.get());
    d_bj->apply(d_b.get(), d_x.get());

    auto result = gko::matrix::Dense<>::create(ref);
    result->copy_from(d_x.get());
    ASSERT_MTX_NEAR(result, expected, 1e-12);
}


TEST_F(BlockJacobi, GpuApplyEquivalentToRefWithDifferentBlockSize)
{
    std::ranlux48 engine(42);
    auto tmp_mtx = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 100, 100, std::uniform_int_distribution<>(97, 99),
        std::normal_distribution<>(0.0, 1.0), engine);
    std::shared_ptr<Mtx> mtx = gko::test::convert_to<Mtx>(tmp_mtx.get());
    std::shared_ptr<Mtx> d_mtx = Mtx::create(gpu);
    d_mtx->copy_from(mtx.get());
    auto bj_factory = BjFactory::create(ref, 32);
    auto d_bj_factory = BjFactory::create(gpu, 32);
    gko::Array<gko::int32> block_ptrs(ref, 11);
    gko::test::init_array(block_ptrs.get_data(),
                          {0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100});
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(d_mtx);
    auto b = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 100, 1, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine);
    auto d_b = gko::matrix::Dense<>::create(gpu);
    d_b->copy_from(b.get());
    auto expected = gko::matrix::Dense<>::create(ref, 100, 1, 1);
    auto d_x = gko::matrix::Dense<>::create(gpu, 100, 1, 1);

    bj->apply(b.get(), expected.get());
    d_bj->apply(d_b.get(), d_x.get());

    auto result = gko::matrix::Dense<>::create(ref);
    result->copy_from(d_x.get());
    ASSERT_MTX_NEAR(result, expected, 1e-12);
}


TEST_F(BlockJacobi, GpuApplyEquivalentToRefWithMpw)
{
    std::ranlux48 engine(42);
    auto tmp_mtx = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 100, 100, std::uniform_int_distribution<>(97, 99),
        std::normal_distribution<>(0.0, 1.0), engine);
    std::shared_ptr<Mtx> mtx = gko::test::convert_to<Mtx>(tmp_mtx.get());
    std::shared_ptr<Mtx> d_mtx = Mtx::create(gpu);
    d_mtx->copy_from(mtx.get());
    auto bj_factory = BjFactory::create(ref, 13);
    auto d_bj_factory = BjFactory::create(gpu, 13);
    gko::Array<gko::int32> block_ptrs(ref, 11);
    gko::test::init_array(block_ptrs.get_data(),
                          {0, 11, 24, 33, 45, 55, 67, 70, 80, 92, 100});
    bj_factory->set_block_pointers(block_ptrs);
    d_bj_factory->set_block_pointers(block_ptrs);
    auto bj = bj_factory->generate(mtx);
    auto d_bj = d_bj_factory->generate(d_mtx);
    auto b = gko::test::generate_random_matrix<gko::matrix::Dense<>>(
        ref, 100, 1, std::uniform_int_distribution<>(1, 1),
        std::normal_distribution<>(0.0, 1.0), engine);
    auto d_b = gko::matrix::Dense<>::create(gpu);
    d_b->copy_from(b.get());
    auto expected = gko::matrix::Dense<>::create(ref, 100, 1, 1);
    auto d_x = gko::matrix::Dense<>::create(gpu, 100, 1, 1);

    bj->apply(b.get(), expected.get());
    d_bj->apply(d_b.get(), d_x.get());

    auto result = gko::matrix::Dense<>::create(ref);
    result->copy_from(d_x.get());
    ASSERT_MTX_NEAR(result, expected, 1e-12);
}

*/

}  // namespace
