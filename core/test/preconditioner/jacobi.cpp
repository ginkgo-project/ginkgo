/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <ginkgo/core/matrix/csr.hpp>


namespace {


class JacobiFactory : public ::testing::Test {
protected:
    using Bj = gko::preconditioner::Jacobi<>;

    JacobiFactory()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(Bj::build().with_max_block_size(3u).on(exec)),
          block_pointers(exec, 2),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<>::create(exec, gko::dim<2>{5, 5}, 13))
    {
        block_pointers.get_data()[0] = 2;
        block_pointers.get_data()[1] = 3;
        block_precisions.get_data()[0] = gko::precision_reduction(0, 1);
        block_precisions.get_data()[1] = gko::precision_reduction(0, 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Bj::Factory> bj_factory;
    gko::Array<gko::int32> block_pointers;
    gko::Array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
};


TEST_F(JacobiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(bj_factory->get_executor(), exec);
}


TEST_F(JacobiFactory, SavesMaximumBlockSize)
{
    ASSERT_EQ(bj_factory->get_parameters().max_block_size, 3);
}


TEST_F(JacobiFactory, CanSetBlockPointers)
{
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_block_pointers(block_pointers)
                          .on(exec);

    auto ptrs = bj_factory->get_parameters().block_pointers;
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TEST_F(JacobiFactory, CanMoveBlockPointers)
{
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_block_pointers(std::move(block_pointers))
                          .on(exec);

    auto ptrs = bj_factory->get_parameters().block_pointers;
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TEST_F(JacobiFactory, CanSetBlockPrecisions)
{
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_storage_optimization(block_precisions)
                          .on(exec);

    auto prec = bj_factory->get_parameters().storage_optimization.block_wise;
    EXPECT_EQ(prec.get_data()[0], gko::precision_reduction(0, 1));
    EXPECT_EQ(prec.get_data()[1], gko::precision_reduction(0, 0));
}


TEST_F(JacobiFactory, CanMoveBlockPrecisions)
{
    auto bj_factory =
        Bj::build()
            .with_max_block_size(3u)
            .with_storage_optimization(std::move(block_precisions))
            .on(exec);

    auto prec = bj_factory->get_parameters().storage_optimization.block_wise;
    EXPECT_EQ(prec.get_data()[0], gko::precision_reduction(0, 1));
    EXPECT_EQ(prec.get_data()[1], gko::precision_reduction(0, 0));
}


class BlockInterleavedStorageScheme : public ::testing::Test {
protected:
    // groups of 4 blocks, offset of 3 within the group and 16 between groups
    gko::preconditioner::block_interleaved_storage_scheme<gko::int32> s{3, 16,
                                                                        2};
};


TEST_F(BlockInterleavedStorageScheme, ComputesStorageSpace)
{
    ASSERT_EQ(s.compute_storage_space(10), 16 * 3);  // 3 groups of 16 elements
}


TEST_F(BlockInterleavedStorageScheme, ComputesGroupOffset)
{
    ASSERT_EQ(s.get_group_offset(17), 16 * 4);  // 5th group
}


TEST_F(BlockInterleavedStorageScheme, ComputesBlockOffset)
{
    ASSERT_EQ(s.get_block_offset(17), 1 * 3);  // 2nd in group
}


TEST_F(BlockInterleavedStorageScheme, ComputesGlobalBlockOffset)
{
    ASSERT_EQ(s.get_global_block_offset(17), 16 * 4 + 1 * 3);
}


TEST_F(BlockInterleavedStorageScheme, ComputesStride)
{
    ASSERT_EQ(s.get_stride(), 4 * 3);  // 4 offsets of 3
}


}  // namespace
