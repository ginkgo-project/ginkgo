// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/workspace_tree.hpp>


namespace {


class WorkspaceNodeTest : public ::testing::Test {
protected:
    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(WorkspaceNodeTest, DefaultConstructsEmpty)
{
    gko::solver::detail::WorkspaceNode node{exec};
    ASSERT_FALSE(node.has_child("anything"));
}


TEST_F(WorkspaceNodeTest, GetOrCreateChildCreatesNew)
{
    gko::solver::detail::WorkspaceNode node{exec};
    auto child = node.get_or_create_child("preconditioner");
    ASSERT_NE(child, nullptr);
    ASSERT_TRUE(node.has_child("preconditioner"));
}


TEST_F(WorkspaceNodeTest, GetOrCreateChildReturnsExisting)
{
    gko::solver::detail::WorkspaceNode node{exec};
    auto child1 = node.get_or_create_child("preconditioner");
    auto child2 = node.get_or_create_child("preconditioner");
    ASSERT_EQ(child1, child2);
}


TEST_F(WorkspaceNodeTest, GetChildReturnsNullForMissing)
{
    gko::solver::detail::WorkspaceNode node{exec};
    ASSERT_EQ(node.get_child("missing"), nullptr);
}


TEST_F(WorkspaceNodeTest, BindExecutorSetsOnLocalStorage)
{
    gko::solver::detail::WorkspaceNode node{exec};
    node.bind_executor(exec);
    ASSERT_EQ(node.get_local_storage().get_executor(), exec);
}


TEST_F(WorkspaceNodeTest, NumRhsDefaultsToZero)
{
    gko::solver::detail::WorkspaceNode node{exec};
    ASSERT_EQ(node.get_num_rhs(), gko::size_type{0});
}


TEST_F(WorkspaceNodeTest, SetNumRhsOnNode)
{
    gko::solver::detail::WorkspaceNode node{exec};
    node.set_num_rhs(4);
    ASSERT_EQ(node.get_num_rhs(), gko::size_type{4});
}


TEST_F(WorkspaceNodeTest, NewChildInheritsNumRhs)
{
    gko::solver::detail::WorkspaceNode node{exec};
    node.set_num_rhs(4);
    auto child = node.get_or_create_child("child");
    ASSERT_EQ(child->get_num_rhs(), gko::size_type{4});
}


TEST_F(WorkspaceNodeTest, NewChildInheritsExecutor)
{
    gko::solver::detail::WorkspaceNode node{exec};
    auto child = node.get_or_create_child("child");
    ASSERT_EQ(child->get_local_storage().get_executor(), exec);
}


TEST_F(WorkspaceNodeTest, MultipleChildrenCoexist)
{
    gko::solver::detail::WorkspaceNode node{exec};
    auto a = node.get_or_create_child("pre_smoother");
    auto b = node.get_or_create_child("post_smoother");
    ASSERT_NE(a, b);
    ASSERT_TRUE(node.has_child("pre_smoother"));
    ASSERT_TRUE(node.has_child("post_smoother"));
}


TEST_F(WorkspaceNodeTest, DescribeOutputContainsTag)
{
    gko::solver::detail::WorkspaceNode node{exec};
    node.get_or_create_child("preconditioner");
    std::ostringstream oss;
    node.describe(oss);
    ASSERT_NE(oss.str().find("preconditioner"), std::string::npos);
}


// --- Workspace tests ---


class WorkspaceTest : public ::testing::Test {
protected:
    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(WorkspaceTest, CreateReturnsOwningWithDefaultNumRhs)
{
    auto ws = gko::solver::Workspace::create(exec);
    ASSERT_NE(ws, nullptr);
    ASSERT_EQ(ws->get_num_rhs(), gko::size_type{1});
}


TEST_F(WorkspaceTest, CreateWithCustomNumRhs)
{
    auto ws = gko::solver::Workspace::create(exec, 4);
    ASSERT_EQ(ws->get_num_rhs(), gko::size_type{4});
}


TEST_F(WorkspaceTest, RootReturnsNonNull)
{
    auto ws = gko::solver::Workspace::create(exec);
    ASSERT_NE(ws->root(), nullptr);
}


TEST_F(WorkspaceTest, CreateNonOwningPointsToNode)
{
    gko::solver::detail::WorkspaceNode node{exec};
    auto ws = gko::solver::Workspace::create_non_owning(&node);
    ASSERT_NE(ws, nullptr);
    ASSERT_EQ(ws->root(), &node);
}


TEST_F(WorkspaceTest, DescribeDoesNotCrash)
{
    auto ws = gko::solver::Workspace::create(exec);
    std::ostringstream oss;
    ws->describe(oss);
    ASSERT_FALSE(oss.str().empty());
}


}  // namespace
