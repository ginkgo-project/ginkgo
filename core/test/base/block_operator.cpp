// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/block_operator.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


struct DummyOperator : public gko::EnableLinOp<DummyOperator> {
    explicit DummyOperator(std::shared_ptr<const gko::Executor> exec)
        : DummyOperator(std::move(exec), {3, 3})
    {}

    explicit DummyOperator(std::shared_ptr<const gko::Executor> exec,
                           gko::dim<2> size)
        : gko::EnableLinOp<DummyOperator>(std::move(exec), size)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}
};


class BlockOperator : public ::testing::Test {
protected:
    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> operators = {
        {std::make_shared<DummyOperator>(exec),
         std::make_shared<DummyOperator>(exec)},
        {std::make_shared<DummyOperator>(exec),
         std::make_shared<DummyOperator>(exec)}};
    std::default_random_engine engine;
};


template <typename Fn>
void enumerate_for_all_blocks(gko::ptr_param<const gko::BlockOperator> bop,
                              Fn&& fn)
{
    auto size = bop->get_block_size();
    for (gko::size_type row = 0; row < size[0]; ++row) {
        for (gko::size_type col = 0; col < size[1]; ++col) {
            fn(row, col);
        }
    }
}


template <typename Fn>
void for_all_blocks(gko::ptr_param<const gko::BlockOperator> bop, Fn&& fn)
{
    enumerate_for_all_blocks(
        bop, [&](auto row, auto col) { fn(bop->block_at(row, col)); });
}


TEST_F(BlockOperator, CanBeEmpty)
{
    auto bop = gko::BlockOperator::create(exec);

    ASSERT_EQ(bop->get_executor(), exec);
    ASSERT_FALSE(bop->get_size());
    ASSERT_FALSE(bop->get_block_size());
}


TEST_F(BlockOperator, CanBeConstructedFromVectors)
{
    auto bop = gko::BlockOperator::create(exec, operators);

    gko::dim<2> block_size{2u, 2u};
    gko::dim<2> global_size{6u, 6u};
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_block_size(), block_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_size(), global_size);
    for_all_blocks(bop, [](auto block) { ASSERT_TRUE(block); });
}


TEST_F(BlockOperator, CanBeConstructedFromInitializerList)
{
    auto bop = gko::BlockOperator::create(
        exec, {{std::make_shared<DummyOperator>(exec),
                std::make_shared<DummyOperator>(exec)},
               {std::make_shared<DummyOperator>(exec),
                std::make_shared<DummyOperator>(exec)}});

    gko::dim<2> block_size{2u, 2u};
    gko::dim<2> global_size{6u, 6u};
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_block_size(), block_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_size(), global_size);
    for_all_blocks(bop, [](auto block) { ASSERT_TRUE(block); });
}


TEST_F(BlockOperator, CanBeConstructedWithNull)
{
    auto bop = gko::BlockOperator::create(
        exec, {{std::make_shared<DummyOperator>(exec), nullptr},
               {nullptr, std::make_shared<DummyOperator>(exec)}});

    gko::dim<2> block_size{2u, 2u};
    gko::dim<2> global_size{6u, 6u};
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_block_size(), block_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_size(), global_size);
    ASSERT_TRUE(bop->block_at(0, 0));
    ASSERT_TRUE(bop->block_at(1, 1));
    ASSERT_FALSE(bop->block_at(0, 1));
    ASSERT_FALSE(bop->block_at(1, 0));
}


TEST_F(BlockOperator, ThrowsOnUnequalNumberOfColumns)
{
    // I need to pull out the temporary because NVHPC can't handle that.
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> ops{
        {std::make_shared<DummyOperator>(exec),
         std::make_shared<DummyOperator>(exec)},
        {std::make_shared<DummyOperator>(exec)}};
    ASSERT_THROW(gko::BlockOperator::create(exec, ops), gko::ValueMismatch);
}


TEST_F(BlockOperator, ThrowsOnEmptyRow)
{
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> ops{
        {std::make_shared<DummyOperator>(exec),
         std::make_shared<DummyOperator>(exec)},
        {nullptr, nullptr}};
    ASSERT_THROW(gko::BlockOperator::create(exec, ops), gko::InvalidStateError);
}


TEST_F(BlockOperator, ThrowsOnEmptyColumn)
{
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> ops{
        {std::make_shared<DummyOperator>(exec), nullptr},
        {std::make_shared<DummyOperator>(exec), nullptr}};
    ASSERT_THROW(gko::BlockOperator::create(exec, ops), gko::InvalidStateError);
}


TEST_F(BlockOperator, ThrowsOnNonMatchingRowsWithinBlocks)
{
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> ops{
        {nullptr, std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2}),
         std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 2})},
        {std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2}), nullptr,
         std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2})}};
    ASSERT_THROW(gko::BlockOperator::create(exec, ops), gko::DimensionMismatch);
}


TEST_F(BlockOperator, ThrowsOnNonMatchingColsWithinBlocks)
{
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> ops{
        {nullptr, std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2}),
         std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1})},
        {std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2}), nullptr,
         std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 2})}};
    ASSERT_THROW(gko::BlockOperator::create(exec, ops), gko::DimensionMismatch);
}


TEST_F(BlockOperator, ThrowsOnOutOfBoundsBlockAccess)
{
    auto bop = gko::BlockOperator::create(
        exec, {{std::make_shared<DummyOperator>(exec, gko::dim<2>{8, 8}),
                std::make_shared<DummyOperator>(exec, gko::dim<2>{8, 8})},
               {std::make_shared<DummyOperator>(exec, gko::dim<2>{8, 8}),
                std::make_shared<DummyOperator>(exec, gko::dim<2>{8, 8})}});

    ASSERT_THROW(bop->block_at(5, 5), gko::OutOfBoundsError);
}


TEST_F(BlockOperator, CanBeCopied)
{
    using Mtx = gko::matrix::Dense<>;
    auto bop = gko::BlockOperator::create(
        exec, {{gko::initialize<Mtx>({{1, 2}, {2, 1}}, exec), nullptr},
               {nullptr, gko::initialize<Mtx>({{3, 4}, {4, 3}}, exec)}});
    auto bop_copy = gko::BlockOperator::create(exec);

    bop_copy->copy_from(bop);

    GKO_ASSERT_EQUAL_DIMENSIONS(bop, bop_copy);
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_block_size(),
                                bop_copy->get_block_size());
    enumerate_for_all_blocks(bop, [&](auto row, auto col) {
        if (bop->block_at(row, col) == nullptr) {
            ASSERT_TRUE(bop_copy->block_at(row, col) == nullptr);
        } else {
            ASSERT_NE(bop->block_at(row, col), bop_copy->block_at(row, col));
            GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(bop->block_at(row, col)),
                                gko::as<Mtx>(bop_copy->block_at(row, col)),
                                0.0);
        }
    });
}


TEST_F(BlockOperator, CanBeMoved)
{
    using Mtx = gko::matrix::Dense<>;
    auto bop = gko::BlockOperator::create(
        exec, {{gko::initialize<Mtx>({{1, 2}, {2, 1}}, exec), nullptr},
               {nullptr, gko::initialize<Mtx>({{3, 4}, {4, 3}}, exec)}});
    auto bop_copy = gko::clone(bop);
    auto bop_move = gko::BlockOperator::create(exec);

    bop_move->move_from(bop);

    GKO_ASSERT_EQUAL_DIMENSIONS(bop_copy, bop_move);
    GKO_ASSERT_EQUAL_DIMENSIONS(bop_copy->get_block_size(),
                                bop_move->get_block_size());
    GKO_ASSERT_EQUAL_DIMENSIONS(bop, gko::dim<2>{});
    GKO_ASSERT_EQUAL_DIMENSIONS(bop->get_block_size(), gko::dim<2>{});
    enumerate_for_all_blocks(bop_copy, [&](auto row, auto col) {
        if (bop_copy->block_at(row, col) == nullptr) {
            ASSERT_TRUE(bop_move->block_at(row, col) == nullptr);
        } else {
            GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(bop_copy->block_at(row, col)),
                                gko::as<Mtx>(bop_move->block_at(row, col)),
                                0.0);
        }
    });
}


TEST_F(BlockOperator, CanApply)
{
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    auto bop = gko::BlockOperator::create(
        exec, {{gko::initialize<Mtx>({{1, 2}, {2, 1}}, exec),
                gko::initialize<Mtx>({{5, 6}, {6, 5}}, exec)},
               {nullptr, gko::initialize<Mtx>({{3, 4}, {4, 3}}, exec)}});
    auto x = gko::initialize<Mtx>({{1, 10}, {2, 20}, {3, 30}, {4, 40}}, exec);
    auto y = Mtx::create_with_config_of(x);

    bop->apply(x, y);

    GKO_ASSERT_MTX_NEAR(
        y, l(I<I<vtype>>{{44, 440}, {42, 420}, {25, 250}, {24, 240}}),
        r<vtype>::value);
}


TEST_F(BlockOperator, CanAdvancedApply)
{
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    auto bop = gko::BlockOperator::create(
        exec, {{gko::initialize<Mtx>({{1, 2}, {2, 1}}, exec),
                gko::initialize<Mtx>({{5, 6}, {6, 5}}, exec)},
               {nullptr, gko::initialize<Mtx>({{3, 4}, {4, 3}}, exec)}});
    auto x = gko::initialize<Mtx>({1, 2, 3, 4}, exec);
    auto y = gko::initialize<Mtx>({-4, -3, -2, -1}, exec);
    auto alpha = gko::initialize<Mtx>({0.5}, exec);
    auto beta = gko::initialize<Mtx>({-1}, exec);

    bop->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l(I<vtype>{26, 24, 14.5, 13}), r<vtype>::value);
}


TEST_F(BlockOperator, CanApplyAndAdvancedApplyLarge)
{
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    gko::size_type local_num_rows = 4;
    gko::size_type local_num_cols = 6;
    gko::size_type block_num_rows = 8;
    gko::size_type block_num_cols = 5;
    auto dense = gko::test::generate_random_dense_matrix<vtype>(
        block_num_rows * local_num_rows, block_num_cols * local_num_cols,
        std::uniform_real_distribution<vtype>(-1, 1), engine, exec);
    auto get_submatrix = [&](auto i, auto j) {
        return dense->create_submatrix(
            {i * local_num_rows, (i + 1) * local_num_rows},
            {j * local_num_cols, (j + 1) * local_num_cols});
    };
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> blocks(
        block_num_rows,
        std::vector<std::shared_ptr<const gko::LinOp>>(block_num_cols));
    for (gko::size_type i = 0; i < block_num_rows; ++i) {
        for (gko::size_type j = 0; j < block_num_cols; ++j) {
            blocks[i][j] = get_submatrix(i, j);
        }
    }
    auto bop = gko::BlockOperator::create(exec, blocks);
    auto x = gko::test::generate_random_dense_matrix<vtype>(
        block_num_cols * local_num_cols, 3,
        std::uniform_real_distribution<vtype>(-1, 1), engine, exec);
    auto y = Mtx::create(exec, gko::dim<2>{block_num_rows * local_num_rows, 3});
    auto result_y = gko::clone(y);

    {
        SCOPED_TRACE("Apply");
        y->fill(1.0);
        result_y->fill(1.0);

        dense->apply(x, result_y);
        bop->apply(x, y);

        GKO_ASSERT_MTX_NEAR(y, result_y, r<vtype>::value);
    }
    {
        SCOPED_TRACE("Advanced Apply");
        auto alpha = gko::initialize<Mtx>({0.5}, exec);
        auto beta = gko::initialize<Mtx>({-1}, exec);
        y->fill(1.0);
        result_y->fill(1.0);

        dense->apply(alpha, x, beta, result_y);
        bop->apply(alpha, x, beta, y);

        GKO_ASSERT_MTX_NEAR(y, result_y, r<vtype>::value);
    }
}
