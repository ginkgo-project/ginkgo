// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/profiler_hook.hpp>


#include <chrono>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/log/profiler_hook.hpp"
#include "core/test/utils.hpp"


std::pair<gko::log::ProfilerHook::hook_function,
          gko::log::ProfilerHook::hook_function>
make_hooks(std::vector<std::string>& output)
{
    return std::make_pair(
        [&output](const char* msg, gko::log::profile_event_category) {
            output.push_back(std::string{"begin:"} + msg);
        },
        [&output](const char* msg, gko::log::profile_event_category) {
            output.push_back(std::string{"end:"} + msg);
        });
}


class DummyOperation : public gko::Operation {
public:
    void run(std::shared_ptr<const gko::OmpExecutor>) const override {}

    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override {}

    void run(std::shared_ptr<const gko::HipExecutor>) const override {}

    void run(std::shared_ptr<const gko::DpcppExecutor>) const override {}

    void run(std::shared_ptr<const gko::CudaExecutor>) const override {}

    const char* get_name() const noexcept override { return "op"; }
};


TEST(ProfilerHook, LogsAllocateCopyOperation)
{
    std::vector<std::string> expected{
        "begin:allocate", "end:allocate", "begin:copy", "end:copy",
        "begin:op",       "end:op",       "begin:free", "end:free"};
    std::vector<std::string> output;
    auto hooks = make_hooks(output);
    auto exec = gko::ReferenceExecutor::create();
    exec->add_logger(gko::log::ProfilerHook::create_custom(
        std::move(hooks.first), std::move(hooks.second)));

    {
        int i = 0;
        gko::array<int> data{exec, 1};
        exec->copy(1, &i, data.get_data());
        exec->run(DummyOperation{});
    }

    ASSERT_EQ(output, expected);
}


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOp, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOp(const Factory* factory, std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOp>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        this->get_executor()->run(DummyOperation{});
    }

    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        this->get_executor()->run(DummyOperation{});
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        this->get_executor()->run(DummyOperation{});
    }
};


TEST(ProfilerHook, LogsPolymorphicObjectLinOp)
{
    std::vector<std::string> expected{"begin:copy(obj,obj)",
                                      "end:copy(obj,obj)",
                                      "begin:move(obj,obj)",
                                      "end:move(obj,obj)",
                                      "begin:apply(obj)",
                                      "begin:op",
                                      "end:op",
                                      "end:apply(obj)",
                                      "begin:advanced_apply(obj)",
                                      "begin:op",
                                      "end:op",
                                      "end:advanced_apply(obj)",
                                      "begin:generate(obj_factory)",
                                      "begin:op",
                                      "end:op",
                                      "end:generate(obj_factory)",
                                      "begin:check(nullptr)",
                                      "end:check(nullptr)"};
    std::vector<std::string> output;
    auto hooks = make_hooks(output);
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::ProfilerHook::create_custom(
        std::move(hooks.first), std::move(hooks.second));
    auto linop = gko::share(DummyLinOp::create(exec));
    auto factory = DummyLinOp::build().on(exec);
    auto scalar = DummyLinOp::create(exec, gko::dim<2>{1, 1});
    logger->set_object_name(linop, "obj");
    logger->set_object_name(factory, "obj_factory");
    exec->add_logger(logger);

    linop->copy_from(linop);
    linop->move_from(linop);
    linop->apply(linop, linop);
    linop->apply(scalar, linop, scalar, linop);
    factory->generate(linop);
    logger->on_criterion_check_started(nullptr, 0, nullptr, nullptr, nullptr, 0,
                                       false);
    logger->on_criterion_check_completed(nullptr, 0, nullptr, nullptr, nullptr,
                                         nullptr, 0, false, nullptr, false,
                                         false);

    exec->remove_logger(logger);
    ASSERT_EQ(output, expected);
}


TEST(ProfilerHook, LogsIteration)
{
    using Vec = gko::matrix::Dense<>;
    std::vector<std::string> expected{"begin:apply(solver)",
                                      "begin:iteration",
                                      "end:iteration",
                                      "end:apply(solver)",
                                      "begin:advanced_apply(solver)",
                                      "begin:iteration",
                                      "end:iteration",
                                      "end:advanced_apply(solver)"};
    std::vector<std::string> output;
    auto hooks = make_hooks(output);
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::ProfilerHook::create_custom(
        std::move(hooks.first), std::move(hooks.second));
    auto mtx = gko::share(Vec::create(exec));
    auto alpha = gko::share(gko::initialize<Vec>({1.0}, exec));
    auto solver =
        gko::solver::Ir<>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec)
            ->generate(mtx);
    logger->set_object_name(solver, "solver");
    logger->set_object_name(mtx, "mtx");
    solver->add_logger(logger);

    solver->apply(mtx, mtx);
    solver->apply(alpha, mtx, alpha, mtx);

    solver->remove_logger(logger);
    ASSERT_EQ(output, expected);
}


TEST(ProfilerHook, ScopeGuard)
{
    std::vector<std::string> expected{"foo", "bar", "bar", "baz", "baz", "foo"};
    static std::vector<std::string> output;
    output.clear();
    class profiling_scope_guard : gko::log::profiling_scope_guard {
    public:
        profiling_scope_guard(const char* name)
            : gko::log::profiling_scope_guard{
                  name, gko::log::profile_event_category::user,
                  [](const char* msg, gko::log::profile_event_category) {
                      output.push_back(msg);
                  },
                  [](const char* msg, gko::log::profile_event_category) {
                      output.push_back(msg);
                  }}
        {}
    };

    {
        GKO_PROFILE_RANGE(foo);
        {
            GKO_PROFILE_RANGE(bar);
        }
        {
            GKO_PROFILE_RANGE(baz);
        }
    }

    ASSERT_EQ(output, expected);
}


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

TEST(ProfilerHook, SummaryWorks)
{
    auto logger = gko::log::ProfilerHook::create_summary(
        std::make_unique<gko::CpuTimer>(),
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

TEST(ProfilerHook, NestedSummaryWorks)
{
    auto logger = gko::log::ProfilerHook::create_nested_summary(
        std::make_unique<gko::CpuTimer>(),
        std::make_unique<TestNestedSummaryWriter>());

    call_ranges(logger);

    // The assertions happen in the destructor of `logger`
}


TEST(ProfilerHookTableSummaryWriter, SummaryWorks)
{
    using gko::log::ProfilerHook;
    using namespace std::chrono_literals;
    std::stringstream ss;
    ProfilerHook::TableSummaryWriter writer(ss, "Test header");
    std::vector<ProfilerHook::summary_entry> entries;
    entries.push_back({"empty", 0ns, 0ns, 0});  // division by zero
    entries.push_back({"short", 1ns, 0ns, 1});
    entries.push_back({"shortish", 1200ns, 1000ns, 1});
    entries.push_back({"medium", 1ms, 500us, 4});  // check division by count
    entries.push_back({"long", 120s, 60s, 1});
    entries.push_back({"eternal", 24h, 24h, 1});
    const auto expected = R"(Test header
Overhead estimate 1.0 s 
|   name   | total  | total (self) | count |   avg    | avg (self) |
|----------|-------:|-------------:|------:|---------:|-----------:|
| eternal  | 1.0 d  |       1.0 d  |     1 |   1.0 d  |     1.0 d  |
| long     | 2.0 m  |       1.0 m  |     1 |   2.0 m  |     1.0 m  |
| medium   | 1.0 ms |     500.0 us |     4 | 250.0 us |   125.0 us |
| shortish | 1.2 us |       1.0 us |     1 |   1.2 us |     1.0 us |
| short    | 1.0 ns |       0.0 ns |     1 |   1.0 ns |     0.0 ns |
| empty    | 0.0 ns |       0.0 ns |     0 |   0.0 ns |     0.0 ns |
)";

    writer.write(entries, 1s);

    ASSERT_EQ(ss.str(), expected);
}


TEST(ProfilerHookTableSummaryWriter, NestedSummaryWorks)
{
    using gko::log::ProfilerHook;
    using namespace std::chrono_literals;
    std::stringstream ss;
    ProfilerHook::TableSummaryWriter writer(ss, "Test header");
    ProfilerHook::nested_summary_entry entry{
        "root",
        2us,
        1,
        {ProfilerHook::nested_summary_entry{"foo", 100ns, 5, {}},
         ProfilerHook::nested_summary_entry{
             "bar",
             1000ns,
             2,
             {ProfilerHook::nested_summary_entry{"child", 100ns, 2, {}}}},
         ProfilerHook::nested_summary_entry{"baz", 1ns, 2, {}}}};
    const auto expected = R"(Test header
Overhead estimate 1.0 ns
|    name    |  total   | fraction | count |   avg    |
|------------|---------:|---------:|------:|---------:|
| root       |   2.0 us |  100.0 % |     1 |   2.0 us |
|   bar      |   1.0 us |   50.0 % |     2 | 500.0 ns |
|     (self) | 900.0 ns |   90.0 % |     2 | 450.0 ns |
|     child  | 100.0 ns |   10.0 % |     2 |  50.0 ns |
|   (self)   | 899.0 ns |   45.0 % |     1 | 899.0 ns |
|   foo      | 100.0 ns |    5.0 % |     5 |  20.0 ns |
|   baz      |   1.0 ns |    0.1 % |     2 |   0.0 ns |
)";

    writer.write_nested(entry, 1ns);

    ASSERT_EQ(ss.str(), expected);
}
