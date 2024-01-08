// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/log/profiler_hook.hpp>


#include "test/utils/executor.hpp"


class ProfilerHook : public CommonTestFixture {
#ifdef GKO_COMPILING_DPCPP
public:
    ProfilerHook()
    {
        // require profiling capability
        const auto property = dpcpp_queue_property::in_order |
                              dpcpp_queue_property::enable_profiling;
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            exec = gko::DpcppExecutor::create(0, ref, "gpu", property);
        } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
            exec = gko::DpcppExecutor::create(0, ref, "cpu", property);
        } else {
            throw std::runtime_error{"No suitable DPC++ devices"};
        }
    }
#endif
};


void call_ranges_unique(std::shared_ptr<gko::log::ProfilerHook> logger)
{
    auto range1 = logger->user_range("foo");
    {
        auto range2 = logger->user_range("bar");
    }
    {
        auto range2 = logger->user_range("bar");
    }
    {
        auto range3 = logger->user_range("baz");
        {
            auto range4 = logger->user_range("bazz");
        }
        {
            auto range5 = logger->user_range("bazzz");
        }
    }
    auto range6 = logger->user_range("bazzzz");
}

struct TestSummaryWriter : gko::log::ProfilerHook::SummaryWriter {
    void write(const std::vector<gko::log::ProfilerHook::summary_entry>& e,
               std::chrono::nanoseconds overhead) override
    {
        /*
         * total(
         *   foo(
         *     bar()
         *     bar()
         *     baz(
         *       bazz()
         *       bazzz()
         *     )
         *     bazzzz()
         *   )
         * )
         */
        ASSERT_EQ(e.size(), 7);
        ASSERT_EQ(e[0].name, "total");
        ASSERT_EQ(e[0].count, 1);
        ASSERT_EQ(e[1].name, "foo");
        ASSERT_EQ(e[1].count, 1);
        ASSERT_EQ(e[2].name, "bar");
        ASSERT_EQ(e[2].count, 2);
        ASSERT_EQ(e[3].name, "baz");
        ASSERT_EQ(e[3].count, 1);
        ASSERT_EQ(e[4].name, "bazz");
        ASSERT_EQ(e[4].count, 1);
        ASSERT_EQ(e[5].name, "bazzz");
        ASSERT_EQ(e[5].count, 1);
        ASSERT_EQ(e[6].name, "bazzzz");
        ASSERT_EQ(e[6].count, 1);
        ASSERT_EQ(e[0].inclusive, e[0].exclusive + e[1].inclusive);
        ASSERT_EQ(e[1].inclusive, e[1].exclusive + e[2].inclusive +
                                      e[3].inclusive + e[6].inclusive);
        ASSERT_EQ(e[2].inclusive, e[2].exclusive);
        ASSERT_EQ(e[3].inclusive,
                  e[3].exclusive + e[4].inclusive + e[5].inclusive);
        ASSERT_EQ(e[4].inclusive, e[4].exclusive);
        ASSERT_EQ(e[5].inclusive, e[5].exclusive);
        ASSERT_EQ(e[6].inclusive, e[6].exclusive);
    }
};

TEST_F(ProfilerHook, SummaryWorks)
{
    auto logger = gko::log::ProfilerHook::create_summary(
        gko::Timer::create_for_executor(this->exec),
        std::make_unique<TestSummaryWriter>());

    call_ranges_unique(logger);

    // The assertions happen in the destructor of `logger`
}


void call_ranges(std::shared_ptr<gko::log::ProfilerHook> logger)
{
    auto range1 = logger->user_range("foo");
    {
        auto range2 = logger->user_range("foo");
    }
    {
        auto range2 = logger->user_range("foo");
    }
    {
        auto range3 = logger->user_range("bar");
        {
            auto range4 = logger->user_range("baz");
        }
        {
            auto range5 = logger->user_range("bazz");
        }
    }
    auto range6 = logger->user_range("baz");
}

struct TestNestedSummaryWriter : gko::log::ProfilerHook::NestedSummaryWriter {
    void write_nested(const gko::log::ProfilerHook::nested_summary_entry& e,
                      std::chrono::nanoseconds overhead) override
    {
        /*
         * total(
         *   foo(
         *     foo()
         *     foo()
         *     bar(
         *       baz()
         *       bazz()
         *     )
         *     baz()
         *   )
         * )
         */
        ASSERT_EQ(e.name, "total");
        ASSERT_EQ(e.count, 1);
        ASSERT_EQ(e.children.size(), 1);
        auto& f = e.children[0];
        ASSERT_EQ(f.name, "foo");
        ASSERT_EQ(f.count, 1);
        ASSERT_EQ(f.children.size(), 3);
        ASSERT_EQ(f.children[0].name, "foo");
        ASSERT_EQ(f.children[0].count, 2);
        ASSERT_EQ(f.children[0].children.size(), 0);
        ASSERT_EQ(f.children[1].name, "bar");
        ASSERT_EQ(f.children[1].count, 1);
        ASSERT_EQ(f.children[1].children.size(), 2);
        ASSERT_EQ(f.children[2].name, "baz");
        ASSERT_EQ(f.children[2].count, 1);
        ASSERT_EQ(f.children[2].children.size(), 0);
        auto& b = f.children[1];
        ASSERT_EQ(b.children[0].name, "baz");
        ASSERT_EQ(b.children[0].count, 1);
        ASSERT_EQ(b.children[1].name, "bazz");
        ASSERT_EQ(b.children[1].count, 1);
    }
};

TEST_F(ProfilerHook, NestedSummaryWorks)
{
    auto logger = gko::log::ProfilerHook::create_nested_summary(
        gko::Timer::create_for_executor(this->exec),
        std::make_unique<TestNestedSummaryWriter>());

    call_ranges(logger);

    // The assertions happen in the destructor of `logger`
}
