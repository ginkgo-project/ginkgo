// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include <ginkgo/ginkgo.hpp>


// The pool's name is the only thing the application and Ginkgo have to agree
// on: Ginkgo looks the pool up by name in Umpire's ResourceManager.
const char pool_name[] = "application_pool";

constexpr std::size_t KiB = std::size_t{1} << 10;
constexpr std::size_t MiB = std::size_t{1} << 20;


// Creates a 2D Poisson matrix (five-point stencil) on a grid x grid grid, with
// the nonzeros of every row in increasing column order.
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_poisson(IndexType grid)
{
    const auto size = static_cast<gko::size_type>(grid) * grid;
    gko::matrix_data<ValueType, IndexType> data{gko::dim<2>{size, size}};
    for (IndexType j = 0; j < grid; ++j) {
        for (IndexType i = 0; i < grid; ++i) {
            const auto row = j * grid + i;
            if (j > 0) {
                data.nonzeros.emplace_back(row, row - grid, ValueType{-1});
            }
            if (i > 0) {
                data.nonzeros.emplace_back(row, row - 1, ValueType{-1});
            }
            data.nonzeros.emplace_back(row, row, ValueType{4});
            if (i < grid - 1) {
                data.nonzeros.emplace_back(row, row + 1, ValueType{-1});
            }
            if (j < grid - 1) {
                data.nonzeros.emplace_back(row, row + grid, ValueType{-1});
            }
        }
    }
    return data;
}


// doc/results.dox contains the PASS lines, so a FAIL also fails the example's
// test, which ignores the numbers printed above them.
bool check(const char* what, bool ok)
{
    std::cout << (ok ? "PASS: " : "FAIL: ") << what << std::endl;
    return ok;
}


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const std::string executor_string = argc >= 2 ? argv[1] : "reference";
    const bool on_device =
        executor_string == "cuda" || executor_string == "hip";

    // Step 1: the application sets Umpire up on its own, exactly as it would
    // without Ginkgo.
    auto& resource_manager = umpire::ResourceManager::getInstance();
    // A plain "DEVICE" binds to whichever device is current when Umpire first
    // creates it, which Ginkgo then refuses if its executor uses another one.
    const std::string resource = on_device ? "DEVICE::0" : "HOST";
    // Umpire aligns to 16 bytes by default; 256 is what cudaMalloc and
    // hipMalloc hand out.
    auto pool = resource_manager.makeAllocator<umpire::strategy::QuickPool>(
        pool_name, resource_manager.getAllocator(resource), 128 * MiB, 16 * MiB,
        256);

    // Memory the application keeps for itself.
    void* application_buffer = pool.allocate(1 * MiB);
    const auto reserved_before = pool.getActualSize();
    const auto in_use_before = pool.getCurrentSize();

    // Step 2: hand the pool to Ginkgo. The device executors keep Ginkgo's
    // default allocator for their master, so a device pool only ever sees
    // device allocations.
    const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"reference",
             [] {
                 return gko::ReferenceExecutor::create(
                     std::make_shared<gko::UmpireCpuAllocator>(pool_name));
             }},
            {"omp",
             [] {
                 return gko::OmpExecutor::create(
                     std::make_shared<gko::UmpireCpuAllocator>(pool_name));
             }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(
                     0, gko::OmpExecutor::create(),
                     std::make_shared<gko::UmpireCudaAllocator>(pool_name));
             }},
            {"hip", [] {
                 return gko::HipExecutor::create(
                     0, gko::OmpExecutor::create(),
                     std::make_shared<gko::UmpireHipAllocator>(pool_name));
             }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Step 3: use Ginkgo as usual. Every allocation below comes from the
    // application's pool.
    const IndexType grid_points = 64;
    const gko::size_type max_iters = 1000u;
    const RealValueType reduction_factor{1e-10};

    std::size_t in_use_during{};
    std::size_t reserved_during{};
    std::size_t allocations_during{};
    gko::size_type iterations{};
    bool converged{};
    RealValueType relative_residual{};
    {
        auto A = gko::share(mtx::create(exec));
        A->read(generate_2d_poisson<ValueType, IndexType>(grid_points));
        const auto size = A->get_size()[0];
        auto b = vec::create(exec, gko::dim<2>{size, 1});
        auto x = vec::create(exec, gko::dim<2>{size, 1});
        b->fill(1.0);
        x->fill(0.0);

        auto logger = gko::share(gko::log::Convergence<ValueType>::create());
        auto solver =
            cg::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(max_iters),
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(reduction_factor))
                .on(exec)
                ->generate(A);
        solver->add_logger(logger);
        solver->apply(b, x);
        exec->synchronize();

        // Read while every Ginkgo object of the solve is still alive.
        in_use_during = pool.getCurrentSize();
        reserved_during = pool.getActualSize();
        allocations_during = pool.getAllocationCount();
        iterations = logger->get_num_iterations();
        converged = logger->has_converged();

        // the explicit residual norm, ||b - A x|| / ||b||
        auto one = gko::initialize<vec>({1.0}, exec);
        auto neg_one = gko::initialize<vec>({-1.0}, exec);
        auto residual = gko::clone(b);
        A->apply(neg_one, x, one, residual);
        auto residual_norm = real_vec::create(exec, gko::dim<2>{1, 1});
        auto rhs_norm = real_vec::create(exec, gko::dim<2>{1, 1});
        residual->compute_norm2(residual_norm);
        b->compute_norm2(rhs_norm);
        relative_residual =
            exec->copy_val_to_host(residual_norm->get_const_values()) /
            exec->copy_val_to_host(rhs_norm->get_const_values());
    }
    const auto in_use_after = pool.getCurrentSize();
    const auto reserved_after = pool.getActualSize();

    std::cout << "The application reserved " << reserved_before / MiB
              << " MiB in its Umpire pool and was holding "
              << in_use_before / MiB << " MiB of it before Ginkgo started."
              << std::endl;
    std::cout << "During the solve the pool was holding " << in_use_during / KiB
              << " KiB in " << allocations_during
              << " live allocations, one of them the application's own."
              << std::endl;
    std::cout << "CG stopped after " << iterations
              << " iterations at a relative residual of " << relative_residual
              << "." << std::endl;

    bool ok = true;
    ok &= check("Ginkgo allocated from the application's pool",
                in_use_during > in_use_before);
    ok &= check("Ginkgo only used memory the application had reserved",
                reserved_during == reserved_before);
    ok &= check(
        "Ginkgo returned everything it had allocated",
        in_use_after == in_use_before && reserved_after == reserved_before);
    // The stopping criterion watches the implicit residual, so the explicit
    // one is only expected to be of the same order of magnitude.
    ok &= check("the solve converged",
                converged && relative_residual <= 100 * reduction_factor);

    pool.deallocate(application_buffer);

    if (!ok) {
        std::cerr << "The Umpire integration did not behave as expected."
                  << std::endl;
        return 1;
    }
    return 0;
}
