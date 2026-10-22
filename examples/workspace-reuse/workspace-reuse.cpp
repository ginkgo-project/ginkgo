// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#ifdef GKO_EXAMPLE_HAVE_CUDSS
#include <ginkgo/extensions/cuda/solver/cudss.hpp>
#endif


namespace {


// Counts the allocations (and the allocated bytes) of the executors it is
// attached to.
struct AllocationCounter : gko::log::Logger {
    AllocationCounter()
        : gko::log::Logger(gko::log::Logger::allocation_completed_mask)
    {}

    void on_allocation_completed(const gko::Executor*,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr&) const override
    {
        ++count;
        bytes += num_bytes;
    }

    void reset()
    {
        count = 0;
        bytes = 0;
    }

    mutable gko::size_type count = 0;
    mutable gko::size_type bytes = 0;
};


// Allocations and bytes of one measured region
struct Allocations {
    gko::size_type count;
    gko::size_type bytes;
};


std::ostream& operator<<(std::ostream& os, const Allocations& allocs)
{
    return os << allocs.count << " allocations (" << allocs.bytes << " bytes)";
}


// Runs `op` and returns the allocations it made
Allocations measure(AllocationCounter& counter, const std::function<void()>& op)
{
    counter.reset();
    op();
    return {counter.count, counter.bytes};
}


// 5-point stencil of the 2D Poisson problem on an n x n grid
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> poisson_2d(gko::size_type n)
{
    const auto size = n * n;
    gko::matrix_data<ValueType, IndexType> data{gko::dim<2>{size, size}};
    for (gko::size_type i = 0; i < n; i++) {
        for (gko::size_type j = 0; j < n; j++) {
            const auto row = static_cast<IndexType>(i * n + j);
            const auto n_idx = static_cast<IndexType>(n);
            if (i > 0) {
                data.nonzeros.emplace_back(row, row - n_idx, -1.0);
            }
            if (j > 0) {
                data.nonzeros.emplace_back(row, row - 1, -1.0);
            }
            data.nonzeros.emplace_back(row, row, 4.0);
            if (j + 1 < n) {
                data.nonzeros.emplace_back(row, row + 1, -1.0);
            }
            if (i + 1 < n) {
                data.nonzeros.emplace_back(row, row + n_idx, -1.0);
            }
        }
    }
    return data;
}


}  // namespace


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
    using isai = gko::preconditioner::SpdIsai<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [grid size] [options...]" << std::endl
                  << "Options: profile  show where the allocations happen"
                  << std::endl
                  << "         cudss    solve the coarsest level with cuDSS"
                  << std::endl
                  << "         reuse    regenerate the solver with "
                     "generate_reuse()"
                  << std::endl;
        std::exit(-1);
    }

    const std::string executor_string = argc >= 2 ? argv[1] : "reference";
    const gko::size_type n = argc >= 3 ? std::stoi(argv[2]) : 64;
    bool profile = false;
    bool use_cudss = false;
    bool reuse = false;
    for (int i = 3; i < argc; i++) {
        const std::string option = argv[i];
        if (option == "profile") {
            profile = true;
        } else if (option == "cudss") {
            use_cudss = true;
        } else if (option == "reuse") {
            reuse = true;
        } else {
            std::cerr << "Unknown option " << option << std::endl;
            std::exit(-1);
        }
    }
#ifndef GKO_EXAMPLE_HAVE_CUDSS
    if (use_cudss) {
        std::cerr << "This example was built without the cuDSS extension"
                  << std::endl;
        std::exit(-1);
    }
#endif
    if (use_cudss && executor_string != "cuda") {
        std::cerr << "cuDSS requires the cuda executor" << std::endl;
        std::exit(-1);
    }
    const int num_repetitions = 5;

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Set up the problem: 2D Poisson, right-hand side 1, initial guess 0
    auto A = gko::share(mtx::create(exec));
    A->read(poisson_2d<ValueType, IndexType>(n));
    const auto size = A->get_size()[0];
    auto b = vec::create(exec, gko::dim<2>{size, 1});
    b->fill(1.0);
    auto x0 = vec::create(exec, gko::dim<2>{size, 1});
    x0->fill(0.0);
    auto x = gko::clone(x0);

    // CG preconditioned by multigrid: the workspace tree contains the CG
    // vectors, the multigrid levels with their smoothers, and the coarsest
    // solver. On large grids, the coarsest level stays large, since the equal
    // weights of the stencil make Pgm aggregate slowly. It is solved by CG
    // with an SPD ISAI preconditioner, or factorized by cuDSS on the GPU.
    auto mg_gen =
        mg::build()
            .with_mg_level(pgm::build().with_deterministic(true))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u));
    if (use_cudss) {
#ifdef GKO_EXAMPLE_HAVE_CUDSS
        mg_gen.with_coarsest_solver(
            gko::ext::cuda::solver::Cudss<ValueType, IndexType>::build());
#endif
    } else {
        mg_gen.with_coarsest_solver(
            cg::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(50u),
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(1e-4))
                .with_preconditioner(isai::build()));
    }
    std::shared_ptr<gko::LinOpFactory> solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(1e-8))
            .with_preconditioner(mg_gen)
            .on(exec);

    // Count the allocations on the executor and, for devices, on the host
    auto counter = std::make_shared<AllocationCounter>();
    exec->add_logger(counter);
    if (exec->get_master() != exec) {
        exec->get_master()->add_logger(counter);
    }
    // Optionally show where the allocations happen
    std::shared_ptr<gko::log::ProfilerHook> profiler;
    if (profile) {
        profiler = gko::log::ProfilerHook::create_nested_summary(
            gko::Timer::create_for_executor(exec),
            std::make_unique<gko::log::ProfilerHook::TableSummaryWriter>(
                std::cout, "Allocation profile"));
        exec->add_logger(profiler);
    }

    std::cout << "Problem: 2D Poisson on a " << n << " x " << n << " grid ("
              << size << " unknowns)" << std::endl;
    if (use_cudss) {
        // cuDSS allocates its memory itself, which the counter doesn't see
        std::cout << "Coarsest solver: cuDSS, its own allocations are not "
                     "counted"
                  << std::endl;
    }
    if (reuse) {
        std::cout << "The solver is regenerated with generate_reuse()"
                  << std::endl;
    }

    // Phase 1: the same solver is applied several times. Only the first apply
    // has to allocate the workspace.
    std::cout << std::endl
              << "Phase 1: repeated apply on one solver" << std::endl;
    auto solver = solver_gen->generate(A);
    const auto mg_solver =
        gko::as<mg>(gko::as<cg>(solver.get())->get_preconditioner());
    std::cout << "multigrid: " << mg_solver->get_mg_level_list().size()
              << " levels, coarsest level "
              << mg_solver->get_coarsest_solver()->get_size()[0] << " rows"
              << std::endl;
    std::vector<Allocations> applies;
    for (int i = 0; i < num_repetitions; i++) {
        x->copy_from(x0);
        applies.push_back(measure(*counter, [&] { solver->apply(b, x); }));
        std::cout << "apply " << i << ": " << applies.back() << std::endl;
    }
    solver.reset();
    const auto steady_count = applies[1].count;
    bool reused_applies = steady_count < applies[0].count;
    for (int i = 1; i < num_repetitions; i++) {
        reused_applies = reused_applies && applies[i].count == steady_count;
    }

    // Phase 2: the solver is regenerated in every step, as in time stepping.
    // Handing the extracted workspace to the next generation avoids
    // allocating it again, which plain generation can't.
    std::cout << std::endl
              << "Phase 2: regenerate the solver in every step" << std::endl;
    auto ws = gko::Workspace::create(exec);
    // With the reuse option, each variant keeps the reuse data of its solver,
    // so that the multigrid reuses its hierarchy
    auto ws_reuse_data = solver_gen->create_empty_reuse_data();
    auto plain_reuse_data = solver_gen->create_empty_reuse_data();
    bool reused_regenerations = true;
    for (int step = 0; step < num_repetitions; step++) {
        std::unique_ptr<gko::LinOp> step_solver;
        const auto ws_generate = measure(*counter, [&] {
            step_solver = reuse ? solver_gen->generate_reuse(A, *ws_reuse_data,
                                                             std::move(ws))
                                : solver_gen->generate(A, std::move(ws));
        });
        x->copy_from(x0);
        const auto ws_apply =
            measure(*counter, [&] { step_solver->apply(b, x); });
        ws = gko::invalidate_and_extract_workspace(std::move(step_solver));

        const auto plain_generate = measure(*counter, [&] {
            step_solver = reuse
                              ? solver_gen->generate_reuse(A, *plain_reuse_data)
                              : solver_gen->generate(A);
        });
        x->copy_from(x0);
        const auto plain_apply =
            measure(*counter, [&] { step_solver->apply(b, x); });
        step_solver.reset();

        std::cout << "step " << step << ":" << std::endl
                  << "  with workspace:    generate " << ws_generate
                  << ", apply " << ws_apply << std::endl
                  << "  without workspace: generate " << plain_generate
                  << ", apply " << plain_apply << std::endl;
        if (step > 0) {
            reused_regenerations = reused_regenerations &&
                                   ws_apply.count == steady_count &&
                                   plain_apply.count > steady_count;
        }
    }

    std::cout << std::endl
              << "Workspace tree after the last step:" << std::endl;
    ws->describe(std::cout);

    std::cout << std::endl
              << "workspace reused across applies: "
              << (reused_applies ? "yes" : "no") << std::endl
              << "workspace reused across regenerations: "
              << (reused_regenerations ? "yes" : "no") << std::endl;

    // print the allocation profile now, before the results go out of scope
    if (profiler) {
        exec->remove_logger(profiler);
        profiler.reset();
    }
    return reused_applies && reused_regenerations ? 0 : 1;
}
