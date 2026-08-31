// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>

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
    gko::array<index_type> block_pointers;
    gko::array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> mtx;
};

TYPED_TEST_SUITE(JacobiFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


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


TYPED_TEST(JacobiFactory, CanSetL1)
{
    using Bj = typename TestFixture::Bj;
    auto bj_factory =
        Bj::build().with_max_block_size(3u).with_aggregate_l1(true).on(
            this->exec);

    EXPECT_TRUE(bj_factory->get_parameters().aggregate_l1);
}


template <typename T>
class BlockInterleavedStorageScheme : public ::testing::Test {
protected:
    using index_type = T;
    // groups of 4 blocks, offset of 3 within the group and 16 between groups
    gko::preconditioner::block_interleaved_storage_scheme<index_type> s{3, 16,
                                                                        2};
};

TYPED_TEST_SUITE(BlockInterleavedStorageScheme, gko::test::IndexTypes,
                 TypenameNameGenerator);


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


TYPED_TEST(JacobiFactory, ValidateBlockPointers)
{
    using Bj = typename TestFixture::Bj;
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> ptrs(this->exec, {0, 2, 4, 5});
    auto bj = Bj::build()
                  .with_max_block_size(3u)
                  .with_block_pointers(ptrs)
                  .on(this->exec)
                  ->generate(this->mtx);

    EXPECT_NO_THROW(bj->validate_data());
}


TYPED_TEST(JacobiFactory, ValidateFiniteBlocks)
{
    using Bj = typename TestFixture::Bj;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    auto dist_mtx = Csr::create(this->exec, gko::dim<2>{2, 2}, 2);
    dist_mtx->get_row_ptrs()[0] = 0;
    dist_mtx->get_row_ptrs()[1] = 1;
    dist_mtx->get_row_ptrs()[2] = 2;
    dist_mtx->get_col_idxs()[0] = 0;
    dist_mtx->get_col_idxs()[1] = 1;
    dist_mtx->get_values()[0] = 1.0;
    dist_mtx->get_values()[1] = 1.0;
    auto bj = Bj::build()
                  .with_max_block_size(2u)
                  .on(this->exec)
                  ->generate(gko::give(dist_mtx));

    EXPECT_NO_THROW(bj->validate_data());
}


}  // namespace
