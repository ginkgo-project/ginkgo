// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


using Vec = gko::matrix::Dense<double>;


struct AllocationCounter : gko::log::Logger {
    AllocationCounter()
        : gko::log::Logger(gko::log::Logger::allocation_completed_mask)
    {}

    void on_allocation_completed(const gko::Executor*, const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        ++count;
    }

    mutable int count = 0;
};


class DummyHolder : public gko::LinOp, public gko::detail::WorkspaceHolder {
public:
    explicit DummyHolder(std::shared_ptr<const gko::Executor> exec)
        : gko::LinOp(exec, gko::dim<2>{2, 2}),
          gko::detail::WorkspaceHolder(exec)
    {}

    DummyHolder(std::shared_ptr<const gko::Executor> exec,
                gko::LinOpGenerateComponents& components)
        : gko::LinOp(exec, gko::dim<2>{2, 2}),
          gko::detail::WorkspaceHolder(exec, components)
    {}

    // LinOp copies don't carry the executor, so copy like Ginkgo's LinOps do
    DummyHolder(const DummyHolder& other)
        : gko::LinOp(other.get_executor(), other.get_size()),
          gko::detail::WorkspaceHolder(other)
    {}

    DummyHolder(DummyHolder&& other)
        : gko::LinOp(other.get_executor(), other.get_size()),
          gko::detail::WorkspaceHolder(std::move(other))
    {}

    DummyHolder& operator=(DummyHolder&&) = default;

    using gko::detail::WorkspaceHolder::get_workspace_node;

    // the holder's only workspace vector
    Vec* vector() const
    {
        this->set_workspace_size(1, 0);
        return this->template create_workspace_op<Vec>(0, gko::dim<2>{2, 1});
    }

    // recreated whenever `size` differs from the stored one
    Vec* vector_if(gko::dim<2> size) const
    {
        this->set_workspace_size(1, 0);
        return this->template create_workspace_op_if<Vec>(
            0, [size](const Vec& v) { return v.get_size() != size; },
            [&] { return Vec::create(this->get_executor(), size); });
    }

protected:
    void apply_impl(const gko::LinOp*, gko::LinOp*) const override {}

    void apply_impl(const gko::LinOp*, const gko::LinOp*, const gko::LinOp*,
                    gko::LinOp*) const override
    {}
};


class OtherHolder : public DummyHolder {
public:
    using DummyHolder::DummyHolder;
};


class WorkspaceHolder : public ::testing::Test {
protected:
    WorkspaceHolder() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
};


TEST_F(WorkspaceHolder, ReusesItsVector)
{
    DummyHolder holder{exec};
    auto* first = holder.vector();

    auto* second = holder.vector();

    ASSERT_EQ(first, second);
}


TEST_F(WorkspaceHolder, CopyGetsFreshWorkspace)
{
    DummyHolder holder{exec};
    holder.vector();

    DummyHolder copy{holder};

    ASSERT_NE(copy.get_workspace_node(), holder.get_workspace_node());
    ASSERT_TRUE(copy.get_workspace_node()->empty());
}


TEST_F(WorkspaceHolder, MoveTransfersWorkspace)
{
    DummyHolder holder{exec};
    auto* node = holder.get_workspace_node();

    DummyHolder moved{std::move(holder)};

    ASSERT_EQ(moved.get_workspace_node(), node);
    // the moved-from holder stays usable with a new workspace
    ASSERT_NE(holder.get_workspace_node(), node);
    ASSERT_EQ(holder.vector()->get_executor(), exec);
}


TEST_F(WorkspaceHolder, MoveAssignmentKeepsExecutor)
{
    auto other = gko::ReferenceExecutor::create();
    DummyHolder holder{exec};
    DummyHolder source{other};
    auto* node = source.get_workspace_node();

    holder = std::move(source);

    ASSERT_EQ(holder.get_workspace_node(), node);
    ASSERT_EQ(holder.vector()->get_executor(), exec);
}


TEST_F(WorkspaceHolder, AdoptsOwnedWorkspaceFromComponents)
{
    auto root = gko::Workspace::create(exec);
    auto* node = root.get();
    gko::LinOpGenerateComponents components{nullptr, std::move(root)};

    DummyHolder holder{exec, components};

    ASSERT_EQ(holder.get_workspace_node(), node);
    ASSERT_FALSE(components.has_owned_workspace());
}


TEST_F(WorkspaceHolder, AttachBorrowsChildNode)
{
    auto root = gko::Workspace::create(exec);
    auto* child = root->get_or_create_child("child");
    auto holder = std::make_unique<DummyHolder>(exec);

    gko::detail::attach_workspace(holder.get(), child);

    ASSERT_EQ(holder->get_workspace_node(), child);
}


TEST_F(WorkspaceHolder, BorrowedNodeOutlivesRoot)
{
    auto root = gko::Workspace::create(exec);
    auto holder = std::make_unique<DummyHolder>(exec);
    gko::detail::attach_workspace(holder.get(),
                                  root->get_or_create_child("child"));
    auto* vec = holder->vector();

    root.reset();

    ASSERT_EQ(holder->vector(), vec);
}


TEST_F(WorkspaceHolder, GiveMakesHolderOwnRoot)
{
    auto root = gko::Workspace::create(exec);
    auto* node = root.get();
    std::unique_ptr<gko::LinOp> holder = std::make_unique<DummyHolder>(exec);

    gko::detail::give_workspace(holder.get(), std::move(root));

    ASSERT_EQ(gko::invalidate_and_extract_workspace(std::move(holder)).get(),
              node);
}


TEST_F(WorkspaceHolder, ExtractingBorrowingHolderThrows)
{
    auto root = gko::Workspace::create(exec);
    std::unique_ptr<gko::LinOp> holder = std::make_unique<DummyHolder>(exec);
    gko::detail::attach_workspace(holder.get(),
                                  root->get_or_create_child("child"));

    ASSERT_THROW(gko::invalidate_and_extract_workspace(std::move(holder)),
                 gko::InvalidStateError);
}


TEST_F(WorkspaceHolder, ExtractingUnusedHolderGivesEmptyRoot)
{
    std::unique_ptr<gko::LinOp> holder = std::make_unique<DummyHolder>(exec);

    auto ws = gko::invalidate_and_extract_workspace(std::move(holder));

    ASSERT_NE(ws, nullptr);
    ASSERT_TRUE(ws->empty());
    ASSERT_EQ(holder, nullptr);
}


TEST_F(WorkspaceHolder, OtherTypeOnSameNodeRecreatesSlots)
{
    auto root = gko::Workspace::create(exec);
    auto* child = root->get_or_create_child("child");
    auto first = std::make_unique<DummyHolder>(exec);
    auto same = std::make_unique<DummyHolder>(exec);
    auto other = std::make_unique<OtherHolder>(exec);
    gko::detail::attach_workspace(first.get(), child);
    gko::detail::attach_workspace(same.get(), child);
    gko::detail::attach_workspace(other.get(), child);
    first->vector();
    auto counter = std::make_shared<AllocationCounter>();
    exec->add_logger(counter);

    same->vector();
    const auto same_type_allocations = counter->count;
    other->vector();
    const auto other_type_allocations = counter->count - same_type_allocations;
    exec->remove_logger(counter);

    ASSERT_EQ(same_type_allocations, 0);
    ASSERT_EQ(other_type_allocations, 1);
}


TEST_F(WorkspaceHolder, CreateIfRecreatesOnPredicate)
{
    DummyHolder holder{exec};
    auto* small = holder.vector_if(gko::dim<2>{2, 1});
    auto* same = holder.vector_if(gko::dim<2>{2, 1});

    auto* large = holder.vector_if(gko::dim<2>{3, 1});

    ASSERT_NE(small, nullptr);
    ASSERT_EQ(small, same);
    ASSERT_NE(large, small);
    ASSERT_EQ(large->get_size(), (gko::dim<2>{3, 1}));
}


TEST_F(WorkspaceHolder, BorrowRebindsNodeToHolderExecutor)
{
    auto other = gko::ReferenceExecutor::create();
    auto root = gko::Workspace::create(other);
    auto* child = root->get_or_create_child("child");
    auto holder = std::make_unique<DummyHolder>(exec);

    gko::detail::attach_workspace(holder.get(), child);

    ASSERT_EQ(child->get_executor(), exec);
    ASSERT_EQ(holder->vector()->get_executor(), exec);
}


TEST_F(WorkspaceHolder, NodeFollowsTheExecutorOfItsUser)
{
    auto other = gko::ReferenceExecutor::create();
    auto root = gko::Workspace::create(exec);
    auto* child = root->get_or_create_child("child");
    auto first = std::make_unique<DummyHolder>(exec);
    auto second = std::make_unique<DummyHolder>(other);
    gko::detail::attach_workspace(first.get(), child);
    first->vector();
    // a later user on another executor rebinds the shared node
    gko::detail::attach_workspace(second.get(), child);
    second->vector();

    auto* vec = first->vector();

    ASSERT_EQ(vec->get_executor(), exec);
    ASSERT_EQ(child->get_executor(), exec);
    ASSERT_EQ(second->vector()->get_executor(), other);
    ASSERT_EQ(child->get_executor(), other);
}


}  // namespace
