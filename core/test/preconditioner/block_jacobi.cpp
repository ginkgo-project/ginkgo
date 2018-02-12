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


#include <core/matrix/csr.hpp>


namespace {


class BlockJacobiFactory : public ::testing::Test {
protected:
    using BjFactory = gko::preconditioner::BlockJacobiFactory<>;
    using Bj = gko::preconditioner::BlockJacobi<>;

    BlockJacobiFactory()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(BjFactory::create(exec, 3)),
          block_pointers(exec, 2),
          mtx(gko::matrix::Csr<>::create(exec, 5, 5, 13))
    {
        block_pointers.get_data()[0] = 2;
        block_pointers.get_data()[1] = 3;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BjFactory> bj_factory;
    gko::Array<gko::int32> block_pointers;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
};


TEST_F(BlockJacobiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(bj_factory->get_executor(), exec);
}


TEST_F(BlockJacobiFactory, SavesMaximumBlockSize)
{
    ASSERT_EQ(bj_factory->get_max_block_size(), 3);
}


TEST_F(BlockJacobiFactory, CanSetBlockPointers)
{
    bj_factory->set_block_pointers(block_pointers);

    auto ptrs = bj_factory->get_block_pointers();
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TEST_F(BlockJacobiFactory, CanMoveBlockPointers)
{
    bj_factory->set_block_pointers(std::move(block_pointers));

    auto ptrs = bj_factory->get_block_pointers();
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


class AdaptiveBlockJacobiFactory : public ::testing::Test {
protected:
    using BjFactory = gko::preconditioner::AdaptiveBlockJacobiFactory<>;
    using Bj = gko::preconditioner::AdaptiveBlockJacobi<>;

    AdaptiveBlockJacobiFactory()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(BjFactory::create(exec, 3)),
          block_pointers(exec, 2),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<>::create(exec, 5, 5, 13))
    {
        block_pointers.get_data()[0] = 2;
        block_pointers.get_data()[1] = 3;
        block_precisions.get_data()[0] = Bj::single_precision;
        block_precisions.get_data()[1] = Bj::double_precision;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BjFactory> bj_factory;
    gko::Array<gko::int32> block_pointers;
    gko::Array<Bj::precision> block_precisions;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
};


TEST_F(AdaptiveBlockJacobiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(bj_factory->get_executor(), exec);
}


TEST_F(AdaptiveBlockJacobiFactory, SavesMaximumBlockSize)
{
    ASSERT_EQ(bj_factory->get_max_block_size(), 3);
}


TEST_F(AdaptiveBlockJacobiFactory, CanSetBlockPointers)
{
    bj_factory->set_block_pointers(block_pointers);

    auto ptrs = bj_factory->get_block_pointers();
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TEST_F(AdaptiveBlockJacobiFactory, CanMoveBlockPointers)
{
    bj_factory->set_block_pointers(std::move(block_pointers));

    auto ptrs = bj_factory->get_block_pointers();
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TEST_F(AdaptiveBlockJacobiFactory, CanSetBlockPrecisions)
{
    bj_factory->set_block_precisions(block_precisions);

    auto prec = bj_factory->get_block_precisions();
    EXPECT_EQ(prec.get_data()[0], Bj::single_precision);
    EXPECT_EQ(prec.get_data()[1], Bj::double_precision);
}


TEST_F(AdaptiveBlockJacobiFactory, CanMoveBlockPrecisions)
{
    bj_factory->set_block_precisions(std::move(block_precisions));

    auto prec = bj_factory->get_block_precisions();
    EXPECT_EQ(prec.get_data()[0], Bj::single_precision);
    EXPECT_EQ(prec.get_data()[1], Bj::double_precision);
}


}  // namespace
