// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


template <typename ValueType>
using vec = gko::matrix::Dense<ValueType>;


template <typename ValueType>
using real_vec = gko::matrix::Dense<gko::remove_complex<ValueType>>;


namespace utils {


// creates a zero vector
template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size,
    ValueType value)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType>(gko::dim<2>{size, 1}, value));
    return res;
}


// utilities for computing norms and residuals
template <typename ValueType>
ValueType get_first_element(const vec<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_norm(const vec<ValueType>* b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<real_vec<ValueType>>({0.0}, exec);
    b->compute_norm2(b_norm);
    return get_first_element(b_norm.get());
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_residual_norm(
    const gko::LinOp* system_matrix, const vec<ValueType>* b,
    const vec<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = gko::clone(b);
    system_matrix->apply(one, x, neg_one, res);
    return compute_norm(res.get());
}


}  // namespace utils


namespace loggers {


// A logger that accumulates the time of all operations. For each operation type
// (allocations, free, copy, internal operations i.e. kernels), the timing is
// taken before and after. This can create significant overhead since to ensure
// proper timings, calls to `synchronize` are required.
struct OperationLogger : gko::log::Logger {
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type&) const override
    {
        this->start_operation(exec, "allocate");
    }

    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        this->end_operation(exec, "allocate");
    }

    void on_free_started(const gko::Executor* exec,
                         const gko::uintptr&) const override
    {
        this->start_operation(exec, "free");
    }

    void on_free_completed(const gko::Executor* exec,
                           const gko::uintptr&) const override
    {
        this->end_operation(exec, "free");
    }

    void on_copy_started(const gko::Executor* from, const gko::Executor* to,
                         const gko::uintptr&, const gko::uintptr&,
                         const gko::size_type&) const override
    {
        from->synchronize();
        this->start_operation(to, "copy");
    }

    void on_copy_completed(const gko::Executor* from, const gko::Executor* to,
                           const gko::uintptr&, const gko::uintptr&,
                           const gko::size_type&) const override
    {
        from->synchronize();
        this->end_operation(to, "copy");
    }

    void on_operation_launched(const gko::Executor* exec,
                               const gko::Operation* op) const override
    {
        this->start_operation(exec, op->get_name());
    }

    void on_operation_completed(const gko::Executor* exec,
                                const gko::Operation* op) const override
    {
        this->end_operation(exec, op->get_name());
    }

    void write_data(std::ostream& ostream)
    {
        for (const auto& entry : total) {
            ostream << "\t" << entry.first.c_str() << ": "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           entry.second)
                           .count()
                    << std::endl;
        }
    }

private:
    // Helper which synchronizes and starts the time before every operation.
    void start_operation(const gko::Executor* exec,
                         const std::string& name) const
    {
        nested.emplace_back(0);
        exec->synchronize();
        start[name] = std::chrono::steady_clock::now();
    }

    // Helper to compute the end time and store the operation's time at its
    // end. Also time nested operations.
    void end_operation(const gko::Executor* exec, const std::string& name) const
    {
        exec->synchronize();
        const auto end = std::chrono::steady_clock::now();
        const auto diff = end - start[name];
        // make sure timings for nested operations are not counted twice
        total[name] += diff - nested.back();
        nested.pop_back();
        if (nested.size() > 0) {
            nested.back() += diff;
        }
    }

    mutable std::map<std::string, std::chrono::steady_clock::time_point> start;
    mutable std::map<std::string, std::chrono::steady_clock::duration> total;
    // the position i of this vector holds the total time spend on child
    // operations on nesting level i
    mutable std::vector<std::chrono::steady_clock::duration> nested;
};


// This logger tracks the persistently allocated data
struct StorageLogger : gko::log::Logger {
    // Store amount of bytes allocated on every allocation
    void on_allocation_completed(const gko::Executor*,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr& location) const override
    {
        storage[location] = num_bytes;
    }

    // Reset the amount of bytes on every free
    void on_free_completed(const gko::Executor*,
                           const gko::uintptr& location) const override
    {
        storage[location] = 0;
    }

    // Write the data after summing the total from all allocations
    void write_data(std::ostream& ostream)
    {
        gko::size_type total{};
        for (const auto& e : storage) {
            total += e.second;
        }
        ostream << "Storage: " << total << std::endl;
    }

private:
    mutable std::unordered_map<gko::uintptr, gko::size_type> storage;
};


// Logs true and recurrent residuals of the solver
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    // Depending on the available information, store the norm or compute it from
    // the residual. If the true residual norm could not be computed, store the
    // value `-1.0`.
    void on_iteration_complete(const gko::LinOp*, const gko::size_type&,
                               const gko::LinOp* residual,
                               const gko::LinOp* solution,
                               const gko::LinOp* residual_norm) const override
    {
        if (residual_norm) {
            rec_res_norms.push_back(utils::get_first_element(
                gko::as<real_vec<ValueType>>(residual_norm)));
        } else {
            rec_res_norms.push_back(
                utils::compute_norm(gko::as<vec<ValueType>>(residual)));
        }
        if (solution) {
            true_res_norms.push_back(utils::compute_residual_norm(
                matrix, b, gko::as<vec<ValueType>>(solution)));
        } else {
            true_res_norms.push_back(-1.0);
        }
    }

    ResidualLogger(const gko::LinOp* matrix, const vec<ValueType>* b)
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b}
    {}

    void write_data(std::ostream& ostream)
    {
        ostream << "Recurrent Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : rec_res_norms) {
            ostream << "\t" << entry << std::endl;
        }
        ostream << "];" << std::endl;

        ostream << "True Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : true_res_norms) {
            ostream << "\t" << entry << std::endl;
        }
        ostream << "];" << std::endl;
    }

private:
    const gko::LinOp* matrix;
    const vec<ValueType>* b;
    mutable std::vector<gko::remove_complex<ValueType>> rec_res_norms;
    mutable std::vector<gko::remove_complex<ValueType>> true_res_norms;
};


}  // namespace loggers


namespace {


// Print usage help
void print_usage(const char* filename)
{
    std::cerr << "Usage: " << filename << " [executor] [matrix file]"
              << std::endl;
    std::cerr << "matrix file should be a file in matrix market format. "
                 "The file data/A.mtx is provided as an example."
              << std::endl;
    std::exit(-1);
}


template <typename ValueType>
void print_vector(const gko::matrix::Dense<ValueType>* vec)
{
    auto elements_to_print = std::min(gko::size_type(10), vec->get_size()[0]);
    std::cout << "[" << std::endl;
    for (int i = 0; i < elements_to_print; ++i) {
        std::cout << "\t" << vec->at(i) << std::endl;
    }
    std::cout << "];" << std::endl;
}


}  // namespace


int main(int argc, char* argv[])
{
    // Parametrize the benchmark here
    // Pick a value type
    using ValueType = double;
    using IndexType = int;
    // Pick a matrix format
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // Pick a solver
    using solver = gko::solver::Cg<ValueType>;
    // Pick a preconditioner type
    using preconditioner = gko::matrix::IdentityFactory<ValueType>;
    // Pick a residual norm reduction value
    const gko::remove_complex<ValueType> reduction_factor = 1e-12;
    // Pick an output file name
    const auto of_name = "log.txt";


    // Simple shortcut
    using vec = gko::matrix::Dense<ValueType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Figure out where to run the code
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Read the input matrix file directory
    std::string input_mtx = "data/A.mtx";
    if (argc == 3) {
        input_mtx = std::string(argv[2]);
    }

    // Read data: A is read from disk
    // Create a StorageLogger to track the size of A
    auto storage_logger = std::make_shared<loggers::StorageLogger>();
    // Add the logger to the executor
    exec->add_logger(storage_logger);
    // Read the matrix A from file
    auto A = gko::share(gko::read<mtx>(std::ifstream(input_mtx), exec));
    // Remove the storage logger
    exec->remove_logger(storage_logger);

    // Pick a maximum iteration count
    const auto max_iters = A->get_size()[0];
    // Generate b and x vectors
    auto b = utils::create_vector<ValueType>(exec, A->get_size()[0], 1.0);
    auto x = utils::create_vector<ValueType>(exec, A->get_size()[0], 0.0);

    // Declare the solver factory. The preconditioner's arguments should be
    // adapted if needed.
    auto solver_factory =
        solver::build()
            .with_criteria(
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(reduction_factor),
                gko::stop::Iteration::build().with_max_iters(max_iters))
            .with_preconditioner(preconditioner::create(exec))
            .on(exec);

    // Declare the output file for all our loggers
    std::ofstream output_file(of_name);

    // Do a warmup run
    {
        // Clone x to not overwrite the original one
        auto x_clone = gko::clone(x);

        // Generate and call apply on a solver
        solver_factory->generate(A)->apply(b, x_clone);
        exec->synchronize();
    }

    // Do a timed run
    {
        // Clone x to not overwrite the original one
        auto x_clone = gko::clone(x);

        // Synchronize ensures no operation are ongoing
        exec->synchronize();
        // Time before generate
        auto g_tic = std::chrono::steady_clock::now();
        // Generate a solver
        auto generated_solver = solver_factory->generate(A);
        exec->synchronize();
        // Time after generate
        auto g_tac = std::chrono::steady_clock::now();
        // Compute the generation time
        auto generate_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(g_tac - g_tic);
        // Write the generate time to the output file
        output_file << "Generate time (ns): " << generate_time.count()
                    << std::endl;

        // Similarly time the apply
        exec->synchronize();
        auto a_tic = std::chrono::steady_clock::now();
        generated_solver->apply(b, x_clone);
        exec->synchronize();
        auto a_tac = std::chrono::steady_clock::now();
        auto apply_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(a_tac - a_tic);
        output_file << "Apply time (ns): " << apply_time.count() << std::endl;

        // Compute the residual norm
        auto residual =
            utils::compute_residual_norm(A.get(), b.get(), x_clone.get());
        output_file << "Residual_norm: " << residual << std::endl;
    }

    // Log the internal operations using the OperationLogger without timing
    {
        // Create an OperationLogger to analyze the generate step
        auto gen_logger = std::make_shared<loggers::OperationLogger>();
        // Add the generate logger to the executor
        exec->add_logger(gen_logger);
        // Generate a solver
        auto generated_solver = solver_factory->generate(A);
        // Remove the generate logger from the executor
        exec->remove_logger(gen_logger);
        // Write the data to the output file
        output_file << "Generate operations times (ns):" << std::endl;
        gen_logger->write_data(output_file);

        // Create an OperationLogger to analyze the apply step
        auto apply_logger = std::make_shared<loggers::OperationLogger>();
        exec->add_logger(apply_logger);
        // Create a ResidualLogger to log the recurent residual
        auto res_logger = std::make_shared<loggers::ResidualLogger<ValueType>>(
            A.get(), b.get());
        generated_solver->add_logger(res_logger);
        // Solve the system
        generated_solver->apply(b, x);
        exec->remove_logger(apply_logger);
        // Write the data to the output file
        output_file << "Apply operations times (ns):" << std::endl;
        apply_logger->write_data(output_file);
        res_logger->write_data(output_file);
    }

    // Print solution
    std::cout << "Solution, first ten entries: \n";
    print_vector(x.get());

    // Print output file location
    std::cout << "The performance and residual data can be found in " << of_name
              << std::endl;
}
