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


#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class JacobiFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Bj = gko::preconditioner::Jacobi<value_type, index_type>;

    JacobiFactory()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(Bj::build().with_max_block_size(3u).on(exec)),
          block_pointers(exec, 2),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<value_type, index_type>::create(
              exec, gko::dim<2>{5, 5}, 13))
    {
        block_pointers.get_data()[0] = 2;
        block_pointers.get_data()[1] = 3;
        block_precisions.get_data()[0] = gko::precision_reduction(0, 1);
        block_precisions.get_data()[1] = gko::precision_reduction(0, 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Bj::Factory> bj_factory;
    gko::Array<index_type> block_pointers;
    gko::Array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> mtx;
};

TYPED_TEST_CASE(JacobiFactory, gko::test::ValueIndexTypes);


TYPED_TEST(JacobiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->bj_factory->get_executor(), this->exec);
}


TYPED_TEST(JacobiFactory, SavesMaximumBlockSize)
{
    ASSERT_EQ(this->bj_factory->get_parameters().max_block_size, 3);
}


TYPED_TEST(JacobiFactory, CanSetBlockPointers)
{
    using Bj = typename TestFixture::Bj;
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_block_pointers(this->block_pointers)
                          .on(this->exec);

    auto ptrs = bj_factory->get_parameters().block_pointers;
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TYPED_TEST(JacobiFactory, CanMoveBlockPointers)
{
    using Bj = typename TestFixture::Bj;
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_block_pointers(std::move(this->block_pointers))
                          .on(this->exec);

    auto ptrs = bj_factory->get_parameters().block_pointers;
    EXPECT_EQ(ptrs.get_data()[0], 2);
    EXPECT_EQ(ptrs.get_data()[1], 3);
}


TYPED_TEST(JacobiFactory, CanSetBlockPrecisions)
{
    using Bj = typename TestFixture::Bj;
    auto bj_factory = Bj::build()
                          .with_max_block_size(3u)
                          .with_storage_optimization(this->block_precisions)
                          .on(this->exec);

    auto prec = bj_factory->get_parameters().storage_optimization.block_wise;
    EXPECT_EQ(prec.get_data()[0], gko::precision_reduction(0, 1));
    EXPECT_EQ(prec.get_data()[1], gko::precision_reduction(0, 0));
}


TYPED_TEST(JacobiFactory, CanMoveBlockPrecisions)
{
    using Bj = typename TestFixture::Bj;
    auto bj_factory =
        Bj::build()
            .with_max_block_size(3u)
            .with_storage_optimization(std::move(this->block_precisions))
            .on(this->exec);

    auto prec = bj_factory->get_parameters().storage_optimization.block_wise;
    EXPECT_EQ(prec.get_data()[0], gko::precision_reduction(0, 1));
    EXPECT_EQ(prec.get_data()[1], gko::precision_reduction(0, 0));
}


template <typename T>
class BlockInterleavedStorageScheme : public ::testing::Test {
protected:
    using index_type = T;
    // groups of 4 blocks, offset of 3 within the group and 16 between groups
    gko::preconditioner::block_interleaved_storage_scheme<index_type> s{3, 16,
                                                                        2};
};

TYPED_TEST_CASE(BlockInterleavedStorageScheme, gko::test::IndexTypes);


TYPED_TEST(BlockInterleavedStorageScheme, ComputesStorageSpace)
{
    ASSERT_EQ(this->s.compute_storage_space(10),
              16 * 3);  // 3 groups of 16 elements
}


TYPED_TEST(BlockInterleavedStorageScheme, ComputesGroupOffset)
{
    ASSERT_EQ(this->s.get_group_offset(17), 16 * 4);  // 5th group
}


TYPED_TEST(BlockInterleavedStorageScheme, ComputesBlockOffset)
{
    ASSERT_EQ(this->s.get_block_offset(17), 1 * 3);  // 2nd in group
}


TYPED_TEST(BlockInterleavedStorageScheme, ComputesGlobalBlockOffset)
{
    ASSERT_EQ(this->s.get_global_block_offset(17), 16 * 4 + 1 * 3);
}


TYPED_TEST(BlockInterleavedStorageScheme, ComputesStride)
{
    ASSERT_EQ(this->s.get_stride(), 4 * 3);  // 4 offsets of 3
}


}  // namespace
