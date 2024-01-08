// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_
#define GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_


#include <iostream>
#include <unordered_map>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


/** Categorization of logger events. */
enum class profile_event_category {
    /** Memory allocation. */
    memory,
    /** Kernel execution and data movement. */
    operation,
    /** PolymorphicObject events. */
    object,
    /** LinOp events. */
    linop,
    /** LinOpFactory events. */
    factory,
    /** Solver events. */
    solver,
    /** Stopping criterion events. */
    criterion,
    /** User-defined events. */
    user,
    /** For development use. */
    internal,
};


class profiling_scope_guard;


/**
 * This Logger can be used to annotate the execution of Ginkgo functionality
 * with profiler-specific ranges. It currently supports TAU, VTune,
 * NSightSystems (NVTX) and rocPROF(ROCTX) and custom profiler hooks.
 *
 * The Logger should be attached to the Executor that is being used to run the
 * application for a full, program-wide annotation, or to individual objects to
 * only highlight events caused directly by them (not operations and memory
 * allocations though)
 */
class ProfilerHook : public Logger {
public:
    using hook_function =
        std::function<void(const char*, profile_event_category)>;

    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type&) const override;

    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type&,
                                 const gko::uintptr&) const override;

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr&) const override;

    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr&) const override;

    void on_copy_started(const gko::Executor* from, const gko::Executor* to,
                         const gko::uintptr&, const gko::uintptr&,
                         const gko::size_type&) const override;

    void on_copy_completed(const gko::Executor* from, const gko::Executor* to,
                           const gko::uintptr&, const gko::uintptr&,
                           const gko::size_type&) const override;

    /* Operation events */
    void on_operation_launched(const Executor* exec,
                               const Operation* operation) const override;

    void on_operation_completed(const Executor* exec,
                                const Operation* operation) const override;

    /* PolymorphicObject events */
    void on_polymorphic_object_copy_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_copy_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_move_started(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    void on_polymorphic_object_move_completed(
        const Executor* exec, const PolymorphicObject* from,
        const PolymorphicObject* to) const override;

    /* LinOp events */
    void on_linop_apply_started(const LinOp* A, const LinOp* b,
                                const LinOp* x) const override;

    void on_linop_apply_completed(const LinOp* A, const LinOp* b,
                                  const LinOp* x) const override;

    void on_linop_advanced_apply_started(const LinOp* A, const LinOp* alpha,
                                         const LinOp* b, const LinOp* beta,
                                         const LinOp* x) const override;

    void on_linop_advanced_apply_completed(const LinOp* A, const LinOp* alpha,
                                           const LinOp* b, const LinOp* beta,
                                           const LinOp* x) const override;

    /* LinOpFactory events */
    void on_linop_factory_generate_started(const LinOpFactory* factory,
                                           const LinOp* input) const override;

    void on_linop_factory_generate_completed(
        const LinOpFactory* factory, const LinOp* input,
        const LinOp* output) const override;

    /* Criterion events */
    void on_criterion_check_started(const stop::Criterion* criterion,
                                    const size_type& num_iterations,
                                    const LinOp* residual,
                                    const LinOp* residual_norm,
                                    const LinOp* solution,
                                    const uint8& stopping_id,
                                    const bool& set_finalized) const override;

    void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* solution, const uint8& stopping_id,
        const bool& set_finalized, const array<stopping_status>* status,
        const bool& one_changed, const bool& all_stopped) const override;

    void on_criterion_check_completed(
        const stop::Criterion* criterion, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_resnorm, const LinOp* solution,
        const uint8& stopping_id, const bool& set_finalized,
        const array<stopping_status>* status, const bool& one_changed,
        const bool& all_stopped) const override;

    /* Internal solver events */
    void on_iteration_complete(
        const LinOp* solver, const LinOp* right_hand_side,
        const LinOp* solution, const size_type& num_iterations,
        const LinOp* residual, const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm,
        const array<stopping_status>* status, bool stopped) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(const LinOp* solver,
                               const size_type& num_iterations,
                               const LinOp* residual, const LinOp* solution,
                               const LinOp* residual_norm) const override;

    GKO_DEPRECATED(
        "Please use the version with the additional stopping "
        "information.")
    void on_iteration_complete(
        const LinOp* solver, const size_type& num_iterations,
        const LinOp* residual, const LinOp* solution,
        const LinOp* residual_norm,
        const LinOp* implicit_sq_residual_norm) const override;

    bool needs_propagation() const override;

    /**
     * Sets the name for an object to be profiled. Every instance of that object
     * in the profile will be replaced by the name instead of its runtime type.
     *
     * @param obj  the object
     * @param name  its name
     */
    void set_object_name(ptr_param<const PolymorphicObject> obj,
                         std::string name);

    /**
     * Should the events call executor->synchronize on operations and
     * copy/allocation? This leads to a certain overhead, but makes the
     * execution timeline of kernels synchronous.
     */
    void set_synchronization(bool synchronize);

    /**
     * Creates a scope guard for a user-defined range to be included in the
     * profile.
     *
     * @param name  the name of the range
     *
     * @return  the scope guard. It will begin a range immediately and end it at
     *          the end of its scope.
     */
    profiling_scope_guard user_range(const char* name) const;

    /** The Ginkgo yellow background color as packed 32 bit ARGB value. */
    constexpr static uint32 color_yellow_argb = 0xFFFFCB05U;

    /**
     * Creates a logger annotating Ginkgo events with TAU ranges via PerfStubs.
     *
     * @param initialize  Should we call TAU's initialization and finalization
     *                    functions, or does the application take care of it?
     *                    The initialization will happen immediately, the
     *                    finalization at program exit.
     */
    static std::shared_ptr<ProfilerHook> create_tau(bool initialize = true);

    /**
     * Creates a logger annotating Ginkgo events with VTune ITT ranges.
     */
    static std::shared_ptr<ProfilerHook> create_vtune();

    /**
     * Creates a logger annotating Ginkgo events with NVTX ranges for CUDA.
     * @param color_argb  The color of the NVTX ranges in the NSight Systems
     *                    output. It has to be a 32 bit packed ARGB value.
     */
    static std::shared_ptr<ProfilerHook> create_nvtx(
        uint32 color_argb = color_yellow_argb);

    /**
     * Creates a logger annotating Ginkgo events with ROCTX ranges for HIP.
     */
    static std::shared_ptr<ProfilerHook> create_roctx();

    /**
     * Creates a logger annotating Ginkgo events with the most suitable backend
     * for the given executor: NVTX for NSight Systems in CUDA, ROCTX for
     * rocprof in HIP, TAU for everything else.
     */
    static std::shared_ptr<ProfilerHook> create_for_executor(
        std::shared_ptr<const Executor> exec);

    struct summary_entry {
        /** The name of the range. */
        std::string name;
        /** The total runtime of all invocations of the range in nanoseconds. */
        std::chrono::nanoseconds inclusive{0};
        /**
         * The total runtime of all invocations of the range in nanoseconds,
         * excluding the runtime of all nested ranges.
         */
        std::chrono::nanoseconds exclusive{0};
        /** The total number of invocations of the range. */
        int64 count{};
    };

    struct nested_summary_entry {
        /** The name of the range. */
        std::string name;
        /** The total runtime of all invocations of the range in nanoseconds. */
        std::chrono::nanoseconds elapsed{0};
        /** The total number of invocations of the range. */
        int64 count{};
        /** The nested ranges inside this range. */
        std::vector<nested_summary_entry> children{};
    };

    /** Receives the results from ProfilerHook::create_summary(). */
    class SummaryWriter {
    public:
        virtual ~SummaryWriter() = default;

        /**
         * Callback to write out the summary results.
         *
         * @param entries  the vector of ranges with runtime and count.
         * @param overhead  an estimate of the profiler overhead
         */
        virtual void write(const std::vector<summary_entry>& entries,
                           std::chrono::nanoseconds overhead) = 0;
    };

    /** Receives the results from ProfilerHook::create_nested_summary(). */
    class NestedSummaryWriter {
    public:
        virtual ~NestedSummaryWriter() = default;

        /**
         * Callback to write out the summary results.
         *
         * @param root  the root range with runtime and count.
         * @param overhead  an estimate of the profiler overhead
         */
        virtual void write_nested(const nested_summary_entry& root,
                                  std::chrono::nanoseconds overhead) = 0;
    };

    /**
     * Writes the results from ProfilerHook::create_summary() and
     * ProfilerHook::create_nested_summary() to a ASCII table in Markdown
     * format.
     */
    class TableSummaryWriter : public SummaryWriter,
                               public NestedSummaryWriter {
    public:
        /**
         * Constructs a writer on an output stream.
         *
         * @param output  the output stream to write the table to.
         * @param header  the header to write above the table.
         */
        TableSummaryWriter(std::ostream& output = std::cerr,
                           std::string header = "Runtime summary");

        void write(const std::vector<summary_entry>& entries,
                   std::chrono::nanoseconds overhead) override;

        void write_nested(const nested_summary_entry& root,
                          std::chrono::nanoseconds overhead) override;

    private:
        std::ostream* output_;
        std::string header_;
    };

    /**
     * Creates a logger measuring the runtime of Ginkgo events and printing a
     * summary when it is destroyed.
     *
     * @param timer  The timer used to record time points.
     * @param writer  The SummaryWriter to receive the performance results.
     * @param debug_check_nesting  Enable this flag if the output looks like it
     *                             might contain incorrect nesting. This
     *                             increases the overhead slightly, but
     *                             recognizes mismatching push/pop pairs on the
     *                             range stack.
     *
     * @note For this logger to provide reliable GPU timings, either use
     *       Timer::create_for_executor or enable synchronization via
     *       `set_synchronization(true)`.
     */
    static std::shared_ptr<ProfilerHook> create_summary(
        std::shared_ptr<Timer> timer = std::make_shared<CpuTimer>(),
        std::unique_ptr<SummaryWriter> writer =
            std::make_unique<TableSummaryWriter>(),
        bool debug_check_nesting = false);

    /**
     * Creates a logger measuring the runtime of Ginkgo events in a nested
     * fashion and printing a summary when it is destroyed.
     *
     * @param timer  The timer used to record time points.
     * @param writer  The NestedSummaryWriter to receive the performance
     *                results.
     * @param debug_check_nesting  Enable this flag if the output looks like it
     *                             might contain incorrect nesting. This
     *                             increases the overhead slightly, but
     *                             recognizes mismatching push/pop pairs on the
     *                             range stack.
     *
     * @note For this logger to provide reliable GPU timings, either use
     *       Timer::create_for_executor or enable synchronization via
     *       `set_synchronization(true)`.
     */
    static std::shared_ptr<ProfilerHook> create_nested_summary(
        std::shared_ptr<Timer> timer = std::make_shared<CpuTimer>(),
        std::unique_ptr<NestedSummaryWriter> writer =
            std::make_unique<TableSummaryWriter>(),
        bool debug_check_nesting = false);

    /**
     * Creates a logger annotating Ginkgo events with a custom set of functions
     * for range begin and end.
     */
    static std::shared_ptr<ProfilerHook> create_custom(hook_function begin,
                                                       hook_function end);

private:
    ProfilerHook(hook_function begin, hook_function end);

    void maybe_synchronize(const Executor* exec) const;

    std::string stringify_object(const PolymorphicObject* obj) const;

    std::unordered_map<const PolymorphicObject*, std::string> name_map_;
    bool synchronize_;
    hook_function begin_hook_;
    hook_function end_hook_;
};


/**
 * Scope guard that annotates its scope with the provided profiler hooks.
 */
class profiling_scope_guard {
public:
    /** Creates an empty (moved-from) scope guard. */
    profiling_scope_guard();

    /**
     * Creates the scope guard
     *
     * @param name  the name of the profiler range
     * @param category  the category of the profiler range
     * @param begin  the hook function to begin a range
     * @param end  the hook function to end a range
     */
    profiling_scope_guard(const char* name, profile_event_category category,
                          ProfilerHook::hook_function begin,
                          ProfilerHook::hook_function end);

    /** Calls the range end function if the scope guard was not moved from. */
    ~profiling_scope_guard();

    profiling_scope_guard(const profiling_scope_guard&) = delete;

    // TODO17: unnecessary with guaranteed RVO
    /** Move-constructs from another scope guard, other will be left empty. */
    profiling_scope_guard(profiling_scope_guard&& other);

    profiling_scope_guard& operator=(const profiling_scope_guard&) = delete;

    profiling_scope_guard& operator=(profiling_scope_guard&&) = delete;

private:
    bool empty_;
    const char* name_;
    profile_event_category category_;
    ProfilerHook::hook_function end_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_
