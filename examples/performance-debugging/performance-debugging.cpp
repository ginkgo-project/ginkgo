/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
double get_norm(const vec<ValueType> *norm)
{
    return clone(norm->get_executor()->get_master(), norm)->at(0, 0);
}


template <typename ValueType>
double compute_norm(const vec<ValueType> *b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<vec<ValueType>>({0.0}, exec);
    b->compute_norm2(gko::lend(b_norm));
    return get_norm(gko::lend(b_norm));
}


template <typename ValueType>
double compute_residual_norm(const gko::LinOp *system_matrix,
                             const vec<ValueType> *b, const vec<ValueType> *x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = gko::clone(b);
    system_matrix->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one),
                         gko::lend(res));
    return compute_norm(gko::lend(res));
}


}  // namespace utils


namespace loggers {


// A logger that accumulates the time of all operations. For each operation type
// (allocations, free, copy, internal operations i.e. kernels), the timing is
// taken before and after. This can create significant overhead since to ensure
// proper timings, calls to `synchronize` are required.
struct OperationLogger : gko::log::Logger {
    void on_allocation_started(const gko::Executor *exec,
                               const gko::size_type &) const override
    {
        this->start_operation(exec, "allocate");
    }

    void on_allocation_completed(const gko::Executor *exec,
                                 const gko::size_type &,
                                 const gko::uintptr &) const override
    {
        this->end_operation(exec, "allocate");
    }

    void on_free_started(const gko::Executor *exec,
                         const gko::uintptr &) const override
    {
        this->start_operation(exec, "free");
    }

    void on_free_completed(const gko::Executor *exec,
                           const gko::uintptr &) const override
    {
        this->end_operation(exec, "free");
    }

    void on_copy_started(const gko::Executor *from, const gko::Executor *to,
                         const gko::uintptr &, const gko::uintptr &,
                         const gko::size_type &) const override
    {
        from->synchronize();
        this->start_operation(to, "copy");
    }

    void on_copy_completed(const gko::Executor *from, const gko::Executor *to,
                           const gko::uintptr &, const gko::uintptr &,
                           const gko::size_type &) const override
    {
        from->synchronize();
        this->end_operation(to, "copy");
    }

    void on_operation_launched(const gko::Executor *exec,
                               const gko::Operation *op) const override
    {
        this->start_operation(exec, op->get_name());
    }

    void on_operation_completed(const gko::Executor *exec,
                                const gko::Operation *op) const override
    {
        this->end_operation(exec, op->get_name());
    }

    void write_data(std::ostream &ostream)
    {
        for (const auto &entry : total) {
            ostream << "\t" << entry.first.c_str() << ": "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           entry.second)
                           .count()
                    << std::endl;
        }
    }

    OperationLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec)
    {}

private:
    // Helper which synchronizes and starts the time before every operation.
    void start_operation(const gko::Executor *exec,
                         const std::string &name) const
    {
        nested.emplace_back(0);
        exec->synchronize();
        start[name] = std::chrono::steady_clock::now();
    }

    // Helper to compute the end time and store the operation's time at its
    // end. Also time nested operations.
    void end_operation(const gko::Executor *exec, const std::string &name) const
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
    void on_allocation_completed(const gko::Executor *,
                                 const gko::size_type &num_bytes,
                                 const gko::uintptr &location) const override
    {
        storage[location] = num_bytes;
    }

    // Reset the amount of bytes on every free
    void on_free_completed(const gko::Executor *,
                           const gko::uintptr &location) const override
    {
        storage[location] = 0;
    }

    // Write the data after summing the total from all allocations
    void write_data(std::ostream &ostream)
    {
        gko::size_type total{};
        for (const auto &e : storage) {
            total += e.second;
        }
        ostream << "Storage: " << total << std::endl;
    }

    StorageLogger(std::shared_ptr<const gko::Executor> exec)
        : gko::log::Logger(exec)
    {}

private:
    mutable std::unordered_map<gko::uintptr, gko::size_type> storage;
};


// Logs true and recurrent residuals of the solver
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    // Depending on the available information, store the norm or compute it from
    // the residual. If the true residual norm could not be computed, store the
    // value `-1.0`.
    void on_iteration_complete(const gko::LinOp *, const gko::size_type &,
                               const gko::LinOp *residual,
                               const gko::LinOp *solution,
                               const gko::LinOp *residual_norm) const override
    {
        if (residual_norm) {
            rec_res_norms.push_back(
                utils::get_norm(gko::as<vec<ValueType>>(residual_norm)));
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

    ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                   const gko::LinOp *matrix, const vec<ValueType> *b)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b}
    {}

    void write_data(std::ostream &ostream)
    {
        ostream << "Recurrent Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto &entry : rec_res_norms) {
            ostream << "\t" << entry << std::endl;
        }
        ostream << "];" << std::endl;

        ostream << "True Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto &entry : true_res_norms) {
            ostream << "\t" << entry << std::endl;
        }
        ostream << "];" << std::endl;
    }

private:
    const gko::LinOp *matrix;
    const vec<ValueType> *b;
    mutable std::vector<ValueType> rec_res_norms;
    mutable std::vector<ValueType> true_res_norms;
};


}  // namespace loggers


namespace {


// Print usage help
void print_usage(const char *filename)
{
    std::cerr << "Usage: " << filename << " [executor] [matrix file]"
              << std::endl;
    std::cerr << "matrix file should be a file in matrix market format. "
                 "The file data/A.mtx is provided as an example."
              << std::endl;
    std::exit(-1);
}


template <typename ValueType>
void print_vector(const gko::matrix::Dense<ValueType> *vec)
{
    auto elements_to_print = std::min(gko::size_type(10), vec->get_size()[0]);
    std::cout << "[" << std::endl;
    for (int i = 0; i < elements_to_print; ++i) {
        std::cout << "\t" << vec->at(i) << std::endl;
    }
    std::cout << "];" << std::endl;
}


}  // namespace


int main(int argc, char *argv[])
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
    ValueType reduction_factor = 1e-8;
    // Pick a maximum iteration count
    auto max_iters = 2000u;
    // Pick an output file name
    auto of_name = "log.txt";


    // Simple shortcut
    using vec = gko::matrix::Dense<ValueType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc > 1 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc > 1 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else {
        print_usage(argv[0]);
    }

    // Read the input matrix file directory
    std::string input_mtx = "data/A.mtx";
    if (argc == 3) {
        input_mtx = std::string(argv[2]);
    }

    // Read data: A is read from disk
    // Create a StorageLogger to track the size of A
    auto storage_logger = std::make_shared<loggers::StorageLogger>(exec);
    // Add the logger to the executor
    exec->add_logger(storage_logger);
    // Read the matrix A from file
    auto A = gko::share(gko::read<mtx>(std::ifstream(input_mtx), exec));
    // Remove the storage logger
    exec->remove_logger(gko::lend(storage_logger));

    // Generate b and x vectors
    auto b = utils::create_vector<ValueType>(exec, A->get_size()[0], 1.0);
    auto x = utils::create_vector<ValueType>(exec, A->get_size()[0], 0.0);

    // Declare the solver factory. The preconditioner's arguments should be
    // adapted if needed.
    auto solver_factory =
        solver::build()
            .with_criteria(
                gko::stop::ResidualNormReduction<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec),
                gko::stop::Iteration::build().with_max_iters(max_iters).on(
                    exec))
            .with_preconditioner(preconditioner::create(exec))
            .on(exec);

    // Declare the output file for all our loggers
    std::ofstream output_file(of_name);

    // Do a warmup run
    {
        // Clone x to not overwrite the original one
        auto x_clone = gko::clone(x);

        // Generate and call apply on a solver
        solver_factory->generate(A)->apply(gko::lend(b), gko::lend(x_clone));
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
        generated_solver->apply(gko::lend(b), gko::lend(x_clone));
        exec->synchronize();
        auto a_tac = std::chrono::steady_clock::now();
        auto apply_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(a_tac - a_tic);
        output_file << "Apply time (ns): " << apply_time.count() << std::endl;

        // Compute the residual norm
        auto residual = utils::compute_residual_norm(gko::lend(A), gko::lend(b),
                                                     gko::lend(x_clone));
        output_file << "Residual_norm: " << residual << std::endl;
    }

    // Log the internal operations using the OperationLogger without timing
    {
        // Clone x to not overwrite the original one
        auto x_clone = gko::clone(x);

        // Create an OperationLogger to analyze the generate step
        auto gen_logger = std::make_shared<loggers::OperationLogger>(exec);
        // Add the generate logger to the executor
        exec->add_logger(gen_logger);
        // Generate a solver
        auto generated_solver = solver_factory->generate(A);
        // Remove the generate logger from the executor
        exec->remove_logger(gko::lend(gen_logger));
        // Write the data to the output file
        output_file << "Generate operations times (ns):" << std::endl;
        gen_logger->write_data(output_file);

        // Create an OperationLogger to analyze the apply step
        auto apply_logger = std::make_shared<loggers::OperationLogger>(exec);
        exec->add_logger(apply_logger);
        // Create a ResidualLogger to log the recurent residual
        auto res_logger = std::make_shared<loggers::ResidualLogger<ValueType>>(
            exec, gko::lend(A), gko::lend(b));
        generated_solver->add_logger(res_logger);
        // Solve the system
        generated_solver->apply(gko::lend(b), gko::lend(x_clone));
        exec->remove_logger(gko::lend(apply_logger));
        // Write the data to the output file
        output_file << "Apply operations times (ns):" << std::endl;
        apply_logger->write_data(output_file);
        res_logger->write_data(output_file);
    }

    // Print solution
    std::cout << "Solution, first ten entries: \n";
    print_vector(gko::lend(x));

    // Print output file location
    std::cout << "The performance and residual data can be found in " << of_name
              << std::endl;
}
