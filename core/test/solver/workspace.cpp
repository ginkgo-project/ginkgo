// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/workspace.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/dummy_vector.hpp"


class Vector1 : public AbstractDummyVector<Vector1>,
                public gko::EnableCreateMethod<Vector1> {
public:
    using EnableCreateMethod::create;

    Vector1(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {},
            gko::size_type stride = 0)
        : AbstractDummyVector(exec, size), stride_{stride}
    {}

    gko::size_type get_stride() { return stride_; }

protected:
    gko::size_type stride_;
};


class Vector2 : public AbstractDummyVector<Vector2>,
                public gko::EnableCreateMethod<Vector2> {
public:
    using EnableCreateMethod::create;

    Vector2(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {},
            gko::size_type stride = 0)
        : AbstractDummyVector(exec, size), stride_{stride}
    {}

    gko::size_type get_stride() { return stride_; }

protected:
    gko::size_type stride_;
};


class DerivedVector : public Vector1,
                      public gko::EnableCreateMethod<DerivedVector> {
public:
    using EnableCreateMethod<DerivedVector>::create;

    DerivedVector(std::shared_ptr<const gko::Executor> exec,
                  gko::dim<2> size = {}, gko::size_type stride = 0)
        : Vector1(exec, size, stride)
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

    auto op1 = ws.create_or_get_vector(
        1, [&] { return Vector1::create(exec, size1, stride1); },
        typeid(Vector1), size1);
    auto op2 = ws.create_or_get_vector(
        0, [&] { return Vector2::create(exec, size2, stride2); },
        typeid(Vector2), size2);

    ASSERT_EQ(op1->get_executor(), exec);
    ASSERT_EQ(op2->get_executor(), exec);
    ASSERT_EQ(op1->get_size(), size1);
    ASSERT_EQ(op2->get_size(), size2);
    GKO_ASSERT_DYNAMIC_TYPE(op1, Vector1);
    GKO_ASSERT_DYNAMIC_TYPE(op2, Vector2);
    ASSERT_EQ(gko::as<Vector1>(op1)->get_stride(), stride1);
    ASSERT_EQ(gko::as<Vector2>(op2)->get_stride(), stride2);
    ASSERT_EQ(op1, ws.get_vector(1));
    ASSERT_EQ(op2, ws.get_vector(0));
}


TEST_F(Workspace, CanReuseOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.create_or_get_vector(0, [&] { return Vector1::create(exec); },
                                       typeid(Vector1), {});

    auto op1_reuse = ws.create_or_get_vector(
        0, [&] { return Vector1::create(exec); }, typeid(Vector1), {});

    ASSERT_EQ(op1, op1_reuse);
}


TEST_F(Workspace, ChecksExactOperatorType)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    ws.create_or_get_vector(0, [&] { return Vector1::create(exec); },
                            typeid(Vector1), {});

    auto op1 = ws.create_or_get_vector(
        0, [&] { return std::make_unique<DerivedVector>(exec); },
        typeid(DerivedVector), {});

    GKO_ASSERT_DYNAMIC_TYPE(op1, DerivedVector);
}


TEST_F(Workspace, ChecksOperatorSize)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    const gko::dim<2> size{1, 2};
    ws.create_or_get_vector(0, [&] { return Vector1::create(exec); },
                            typeid(Vector1), {});

    auto op1 = ws.create_or_get_vector(
        0, [&] { return Vector1::create(exec, size); }, typeid(Vector1), size);

    ASSERT_EQ(op1->get_size(), size);
}


TEST_F(Workspace, ClearResetsOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.create_or_get_vector(0, [&] { return Vector1::create(exec); },
                                       typeid(Vector1), {});

    ws.clear();

    ASSERT_EQ(ws.get_vector(0), nullptr);
}


TEST_F(Workspace, MoveResetsOperators)
{
    gko::solver::detail::workspace ws{exec};
    ws.set_size(1, 0);
    auto op1 = ws.create_or_get_vector(0, [&] { return Vector1::create(exec); },
                                       typeid(Vector1), {});

    gko::solver::detail::workspace ws2{std::move(ws)};

    ASSERT_EQ(ws.get_vector(0), nullptr);
}
