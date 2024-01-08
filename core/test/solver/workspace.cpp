// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/workspace.hpp>


#include <typeinfo>


#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {},
               gko::size_type stride = 0)
        : EnableLinOp<DummyLinOp>(exec, size), stride_{stride}
    {}

    gko::size_type get_stride() { return stride_; }

protected:
    gko::size_type stride_;

    void apply_impl(const gko::LinOp*, gko::LinOp*) const override {}

    void apply_impl(const gko::LinOp*, const gko::LinOp*, const gko::LinOp*,
                    gko::LinOp*) const override
    {}
};


class DummyLinOp2 : public gko::EnableLinOp<DummyLinOp2>,
                    public gko::EnableCreateMethod<DummyLinOp2> {
public:
    DummyLinOp2(std::shared_ptr<const gko::Executor> exec,
                gko::dim<2> size = {}, gko::size_type stride = 0)
        : EnableLinOp<DummyLinOp2>(exec, size), stride_{stride}
    {}

    gko::size_type get_stride() { return stride_; }

protected:
    gko::size_type stride_;

    void apply_impl(const gko::LinOp*, gko::LinOp*) const override {}

    void apply_impl(const gko::LinOp*, const gko::LinOp*, const gko::LinOp*,
                    gko::LinOp*) const override
    {}
};


class DerivedDummyLinOp : public DummyLinOp {
public:
    DerivedDummyLinOp(std::shared_ptr<const gko::Executor> exec,
                      gko::dim<2> size = {}, gko::size_type stride = 0)
        : DummyLinOp(exec, size, stride)
    {}
};


class Workspace : public ::testing::Test {
protected:
    Workspace() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TEST_F(Workspace, AnyArrayDefaultConstructedIsEmpty)
{
    gko::solver::detail::any_array array;

    ASSERT_TRUE(array.empty());
    ASSERT_FALSE(array.template contains<int>());
}


TEST_F(Workspace, AnyArrayDefaultConstructedIsEmptyAfterClear)
{
    gko::solver::detail::any_array array;

    array.clear();

    ASSERT_TRUE(array.empty());
    ASSERT_FALSE(array.template contains<int>());
}


TEST_F(Workspace, AnyArrayInitWorks)
{
    gko::solver::detail::any_array array;

    auto& arr = array.template init<int>(exec, 1);

    ASSERT_FALSE(array.empty());
    ASSERT_TRUE(array.template contains<int>());
    ASSERT_FALSE(array.template contains<double>());
    ASSERT_EQ(&array.template get<int>(), &arr);
    ASSERT_EQ(arr.get_size(), 1);
    ASSERT_EQ(arr.get_executor(), exec);
}


TEST_F(Workspace, AnyArrayClearAfterInitWorks)
{
    gko::solver::detail::any_array array;
    auto& arr = array.template init<int>(exec, 1);

    array.clear();

    ASSERT_TRUE(array.empty());
    ASSERT_FALSE(array.template contains<int>());
}


TEST_F(Workspace, CanCreateArrays)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(0, 2);

    auto& arr1 = ws.create_or_get_array<int>(1, 2);
    auto& arr2 = ws.create_or_get_array<double>(0, 3);

    ASSERT_EQ(arr1.get_size(), 2);
    ASSERT_EQ(arr2.get_size(), 3);
    ASSERT_EQ(arr1.get_executor(), exec);
    ASSERT_EQ(arr2.get_executor(), exec);
}


TEST_F(Workspace, CanReuseArrays)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(0, 2);
    auto& arr1 = ws.create_or_get_array<int>(1, 2);
    auto& arr2 = ws.create_or_get_array<double>(0, 3);

    auto& arr1_reuse = ws.create_or_get_array<int>(1, 2);
    auto& arr2_reuse = ws.create_or_get_array<double>(0, 3);

    ASSERT_EQ(arr1.get_size(), 2);
    ASSERT_EQ(arr2.get_size(), 3);
    ASSERT_EQ(arr1.get_executor(), exec);
    ASSERT_EQ(arr2.get_executor(), exec);
    ASSERT_EQ(&arr1, &arr1_reuse);
    ASSERT_EQ(&arr2, &arr2_reuse);
}


TEST_F(Workspace, CanResizeArrays)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(0, 2);
    auto& arr1 = ws.create_or_get_array<int>(1, 2);
    auto& arr2 = ws.create_or_get_array<double>(0, 3);

    auto& arr1_reuse = ws.create_or_get_array<int>(1, 4);
    auto& arr2_reuse = ws.create_or_get_array<double>(0, 5);

    ASSERT_EQ(arr1.get_size(), 4);
    ASSERT_EQ(arr2.get_size(), 5);
    ASSERT_EQ(arr1.get_executor(), exec);
    ASSERT_EQ(arr2.get_executor(), exec);
    ASSERT_EQ(&arr1, &arr1_reuse);
    ASSERT_EQ(&arr2, &arr2_reuse);
}


#ifndef NDEBUG


bool check_assertion_exit_code(int exit_code)
{
#ifdef _MSC_VER
    // MSVC picks up the exit code incorrectly,
    // so we can only check that it exits
    return true;
#else
    return exit_code != 0;
#endif
}


TEST_F(Workspace, AbortsOnDifferentArrayTypes)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(0, 1);
    ws.create_or_get_array<double>(0, 3);

    EXPECT_EXIT(ws.create_or_get_array<int>(0, 4), check_assertion_exit_code,
                "");
}


#endif


TEST_F(Workspace, CanCreateOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(2, 0);
    const gko::dim<2> size1{1, 2};
    const gko::dim<2> size2{5};
    const gko::size_type stride1 = 3;
    const gko::size_type stride2 = 6;

    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        1, [&] { return DummyLinOp::create(exec, size1, stride1); },
        typeid(DummyLinOp), size1, stride1);
    auto op2 = ws.template create_or_get_op<DummyLinOp2>(
        0, [&] { return DummyLinOp2::create(exec, size2, stride2); },
        typeid(DummyLinOp2), size2, stride2);

    ASSERT_EQ(op1->get_executor(), exec);
    ASSERT_EQ(op2->get_executor(), exec);
    ASSERT_EQ(op1->get_size(), size1);
    ASSERT_EQ(op2->get_size(), size2);
    ASSERT_EQ(op1->get_stride(), stride1);
    ASSERT_EQ(op2->get_stride(), stride2);
    GKO_ASSERT_DYNAMIC_TYPE(op1, DummyLinOp);
    GKO_ASSERT_DYNAMIC_TYPE(op2, DummyLinOp2);
    ASSERT_EQ(op1, ws.get_op(1));
    ASSERT_EQ(op2, ws.get_op(0));
}


TEST_F(Workspace, CanReuseOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    auto op1_reuse = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    ASSERT_EQ(op1, op1_reuse);
}


TEST_F(Workspace, ChecksExactOperatorType)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return std::make_unique<DerivedDummyLinOp>(exec); },
        typeid(DerivedDummyLinOp), {}, 0);

    GKO_ASSERT_DYNAMIC_TYPE(op1, DerivedDummyLinOp);
}


TEST_F(Workspace, ChecksOperatorSize)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    const gko::dim<2> size{1, 2};
    ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec, size); }, typeid(DummyLinOp),
        size, 0);

    ASSERT_EQ(op1->get_size(), size);
}


TEST_F(Workspace, ChecksOperatorStride)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec, gko::dim<2>{}, 1); },
        typeid(DummyLinOp), {}, 1);

    ASSERT_EQ(op1->get_stride(), 1);
}


TEST_F(Workspace, ClearResetsOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    ws.clear();

    ASSERT_EQ(ws.get_op(0), nullptr);
}


TEST_F(Workspace, MoveResetsOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.template create_or_get_op<DummyLinOp>(
        0, [&] { return DummyLinOp::create(exec); }, typeid(DummyLinOp), {}, 0);

    gko::solver::detail::workspace ws2{std::move(ws)};

    ASSERT_EQ(ws.get_op(0), nullptr);
}
