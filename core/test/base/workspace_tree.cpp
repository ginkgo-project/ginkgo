// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/workspace.hpp>


namespace {


class WorkspaceTest : public ::testing::Test {
protected:
    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(WorkspaceTest, DefaultConstructsEmpty)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    ASSERT_FALSE(node.has_child("anything"));
}


TEST_F(WorkspaceTest, GetOrCreateChildCreatesNew)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    auto child = node.get_or_create_child("preconditioner");

    ASSERT_NE(child, nullptr);
    ASSERT_TRUE(node.has_child("preconditioner"));
}


TEST_F(WorkspaceTest, GetOrCreateChildReturnsExisting)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    auto child1 = node.get_or_create_child("preconditioner");
    auto child2 = node.get_or_create_child("preconditioner");

    ASSERT_EQ(child1, child2);
}


TEST_F(WorkspaceTest, GetChildReturnsNullForMissing)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    ASSERT_EQ(node.get_child("missing"), nullptr);
}


TEST_F(WorkspaceTest, NewChildInheritsExecutor)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    auto child = node.get_or_create_child("child");

    ASSERT_EQ(child->get_executor(), exec);
}


TEST_F(WorkspaceTest, MultipleChildrenCoexist)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;

    auto a = node.get_or_create_child("pre_smoother");
    auto b = node.get_or_create_child("post_smoother");

    ASSERT_NE(a, b);
    ASSERT_TRUE(node.has_child("pre_smoother"));
    ASSERT_TRUE(node.has_child("post_smoother"));
}


TEST_F(WorkspaceTest, DescribeOutputContainsTag)
{
    auto node_ptr = gko::Workspace::create(exec);
    auto& node = *node_ptr;
    node.get_or_create_child("preconditioner");
    std::ostringstream oss;

    node.describe(oss);

    ASSERT_NE(oss.str().find("preconditioner"), std::string::npos);
}


TEST_F(WorkspaceTest, CreateReturnsOwning)
{
    auto ws = gko::Workspace::create(exec);

    ASSERT_NE(ws, nullptr);
    ASSERT_EQ(ws->get_executor(), exec);
}


TEST_F(WorkspaceTest, DescribeOnEmptyRootShowsZeroChildren)
{
    auto ws = gko::Workspace::create(exec);
    std::ostringstream oss;

    ws->describe(oss);

    ASSERT_EQ(oss.str(), "Workspace (children=0)\n");
}


}  // namespace
