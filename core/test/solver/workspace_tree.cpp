// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/workspace.hpp>


namespace {


class WorkspaceTest : public ::testing::Test {
protected:
    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(WorkspaceTest, DefaultConstructsEmpty)
{
    gko::solver::Workspace node{exec};

    ASSERT_FALSE(node.has_child("anything"));
}


TEST_F(WorkspaceTest, GetOrCreateChildCreatesNew)
{
    gko::solver::Workspace node{exec};

    auto child = node.get_or_create_child("preconditioner");

    ASSERT_NE(child, nullptr);
    ASSERT_TRUE(node.has_child("preconditioner"));
}


TEST_F(WorkspaceTest, GetOrCreateChildReturnsExisting)
{
    gko::solver::Workspace node{exec};

    auto child1 = node.get_or_create_child("preconditioner");
    auto child2 = node.get_or_create_child("preconditioner");

    ASSERT_EQ(child1, child2);
}


TEST_F(WorkspaceTest, GetChildReturnsNullForMissing)
{
    gko::solver::Workspace node{exec};

    ASSERT_EQ(node.get_child("missing"), nullptr);
}


TEST_F(WorkspaceTest, NewChildInheritsExecutor)
{
    gko::solver::Workspace node{exec};

    auto child = node.get_or_create_child("child");

    ASSERT_EQ(child->get_executor(), exec);
}


TEST_F(WorkspaceTest, MultipleChildrenCoexist)
{
    gko::solver::Workspace node{exec};

    auto a = node.get_or_create_child("pre_smoother");
    auto b = node.get_or_create_child("post_smoother");

    ASSERT_NE(a, b);
    ASSERT_TRUE(node.has_child("pre_smoother"));
    ASSERT_TRUE(node.has_child("post_smoother"));
}


TEST_F(WorkspaceTest, DescribeOutputContainsTag)
{
    gko::solver::Workspace node{exec};
    node.get_or_create_child("preconditioner");
    std::ostringstream oss;

    node.describe(oss);

    ASSERT_NE(oss.str().find("preconditioner"), std::string::npos);
}


TEST_F(WorkspaceTest, CreateReturnsOwning)
{
    auto ws = gko::solver::Workspace::create(exec);

    ASSERT_NE(ws, nullptr);
    ASSERT_EQ(ws->get_executor(), exec);
}


TEST_F(WorkspaceTest, DescribeDoesNotCrash)
{
    auto ws = gko::solver::Workspace::create(exec);
    std::ostringstream oss;

    ws->describe(oss);

    ASSERT_FALSE(oss.str().empty());
}


}  // namespace
