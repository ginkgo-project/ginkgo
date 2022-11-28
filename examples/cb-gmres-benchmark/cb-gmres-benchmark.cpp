/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

struct solver_settings {
    unsigned krylov_dim;
    unsigned stop_iter;
    double stop_rel_res;
    gko::solver::cb_gmres::storage_precision storage_prec;
    std::shared_ptr<const gko::LinOp> precond;
    std::string lp_config;
};

struct solver_result {
    unsigned iters;
    double time_s;
    double init_res_norm;
    double res_norm;
};


// Helper function which measures the time of `solver->apply(b, x)` in seconds
// To get an accurate result, the solve is repeated multiple times (while
// ensuring the initial guess is always the same). The result of the solve will
// be written to x.
template <typename ValueType>
solver_result benchmark_solver(
    std::shared_ptr<const gko::Executor> exec, solver_settings s_s,
    std::shared_ptr<gko::matrix::Csr<ValueType, int>> A,
    const gko::matrix::Dense<ValueType>* b,
    const gko::matrix::Dense<ValueType>* x)
{
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    constexpr int repeats{1};
    double duration{0};
    solver_result result{};
    // Make a copy of x, so we can re-use the same initial guess multiple times
    auto x_copy = x->clone();

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    auto res_norm = gko::initialize<real_vec>({0.0}, exec);
    auto tmp = gko::clone(b);

    A->apply(one, x_copy, neg_one, tmp);
    tmp->compute_norm2(res_norm);
    result.init_res_norm = exec->copy_val_to_host(res_norm->get_const_values());

    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(s_s.stop_iter).on(exec));
    auto tol_stop = gko::share(
        gko::stop::ResidualNorm<ValueType>::build()
            .with_reduction_factor(static_cast<RealValueType>(s_s.stop_rel_res))
            .with_baseline(gko::stop::mode::rhs_norm)
            .on(exec));
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Create solver:
    auto solver_gen = gko::solver::CbGmres<ValueType>::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_krylov_dim(s_s.krylov_dim)
                          .with_storage_precision(s_s.storage_prec)
                          .with_generated_preconditioner(s_s.precond)
                          .with_lp_config(s_s.lp_config)
                          .on(exec);

    // Generate the actual solver from the factory and the matrix.
    auto solver = solver_gen->generate(A);

    for (int i = 0; i < repeats; ++i) {
        // No need to copy it in the first iteration
        if (i != 0) {
            x_copy->copy_from(x);
        }
        // Make sure all previous executor operations have finished before
        // starting the time
        exec->synchronize();
        auto tic = std::chrono::steady_clock::now();
        solver->apply(b, x_copy);
        // Make sure all computations are done before stopping the time
        exec->synchronize();
        auto tac = std::chrono::steady_clock::now();
        duration += std::chrono::duration<double>(tac - tic).count();
    }
    // Copy the solution back to x, so the caller has the result
    // x->copy_from(x_copy);

    // To measure if your solution has actually converged, the error of the
    // solution is measured.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // the (advanced) apply method is used.
    // tmp = Ax - tmp
    tmp->copy_from(b);
    A->apply(one, x_copy, neg_one, tmp);
    tmp->compute_norm2(res_norm);

    result.iters = logger->get_num_iterations();
    result.time_s = duration / static_cast<double>(repeats);
    result.res_norm = exec->copy_val_to_host(res_norm->get_const_values());
    return result;
}


template <typename ValueType, typename IndexType>
void run_benchmarks(std::shared_ptr<gko::Executor> exec,
                    const std::string matrix_path,
                    const std::string compression_json_folder,
                    const unsigned max_iters, const double rel_res_norm)
{
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cb_gmres = gko::solver::CbGmres<ValueType>;

    auto A = share(gko::read<mtx>(std::ifstream(matrix_path), exec));

    const auto A_size = A->get_size();
    auto b = vec::create(exec, gko::dim<2>{A_size[0], 1});
    auto x = vec::create(exec, gko::dim<2>{A_size[1], 1});

    // Make sure the output is in scientific notation for easier comparison
    std::cout << std::scientific << std::setprecision(4);

    std::cout << "Matrix: "
              << matrix_path.substr(matrix_path.find_last_of('/') + 1)
              << "; size: " << A_size[0] << " x " << A_size[1] << '\n';

    {  // Prepare values and delete all temporaries afterwards
        auto res_host =
            vec::create(exec->get_master(), gko::dim<2>{A_size[1], 1});
        ValueType tmp_norm{};
        for (gko::size_type i = 0; i < res_host->get_size()[0]; ++i) {
            const auto val = std::sin(static_cast<ValueType>(i));
            res_host->at(i, 0) = val;
            tmp_norm += val * val;
        }
        tmp_norm = std::sqrt(tmp_norm);
        for (gko::size_type i = 0; i < res_host->get_size()[0]; ++i) {
            res_host->at(i, 0) /= tmp_norm;
        }
        // Write out the actual RHS b
        A->apply(res_host, b);
        auto b_host_norm = gko::initialize<real_vec>({0.0}, exec->get_master());
        b->compute_norm2(b_host_norm);

        std::cout << "b-norm: " << b_host_norm->at(0, 0) << '\n';

        // As an initial guess, use the right-hand side
        auto x_host = clone(exec->get_master(), x);
        for (gko::size_type i = 0; i < x_host->get_size()[0]; ++i) {
            x_host->at(i, 0) = 0;
        }
        x->copy_from(x_host);
    }

    using precond_type = gko::preconditioner::Jacobi<ValueType, IndexType>;
    // Default_settings
    solver_settings default_ss{};
    default_ss.stop_iter = max_iters;
    default_ss.stop_rel_res = rel_res_norm;
    default_ss.krylov_dim = 100u;
    default_ss.storage_prec = gko::solver::cb_gmres::storage_precision::keep;
    /*
    default_ss.precond = precond_type::build()
                             .with_max_block_size(32u)
                             .with_skip_sorting(true)
                             .on(exec)
                             ->generate(A);
    /*/
    default_ss.precond = nullptr;
    //*/
    default_ss.lp_config = "ieee";

    std::cout << "Stopping criteria: " << default_ss.stop_iter << " iters; "
              << default_ss.stop_rel_res << " res norm; ";
    std::cout << "Jacobi BS: "
              << (default_ss.precond == nullptr
                      ? 0
                      : dynamic_cast<const precond_type*>(
                            default_ss.precond.get())
                            ->get_storage_scheme()
                            .block_offset)
              << '\n';
    struct bench_type {
        std::string name;
        solver_settings settings;
        solver_result result;
    };

    const auto tt_str = [](int reduction) {
        const std::array<char, 4> types{'d', 'f', 'h', '?'};
        const int base = std::is_same<ValueType, double>::value      ? 0
                         : std::is_same<ValueType, float>::value     ? 1
                         : std::is_same<ValueType, gko::half>::value ? 2
                                                                     : 3;
        if (base == 3) {
            return types[base];
        }
        const int idx = base + reduction;
        return types[idx < types.size() - 1 ? idx : types.size() - 2];
    };
    const std::string str_pre = std::string{"CbGmres<"} + tt_str(0) + ",";
    const std::string str_post{">"};
    const auto get_name = [&str_pre, &str_post, &tt_str](int reduction) {
        return str_pre + tt_str(reduction) + str_post;
    };
    std::vector<bench_type> benchmarks;
    benchmarks.emplace_back();
    benchmarks.back().name = get_name(0);
    benchmarks.back().settings = default_ss;
    benchmarks.back().settings.storage_prec =
        gko::solver::cb_gmres::storage_precision::keep;

    /*
    benchmarks.emplace_back();
    benchmarks.back().name = get_name(1);
    benchmarks.back().settings = default_ss;
    benchmarks.back().settings.storage_prec =
        gko::solver::cb_gmres::storage_precision::reduce1;

    benchmarks.emplace_back();
    benchmarks.back().name = get_name(2);
    benchmarks.back().settings = default_ss;
    benchmarks.back().settings.storage_prec =
        gko::solver::cb_gmres::storage_precision::reduce2;

    std::vector<std::string> compression_json_files;
    for (auto config_path :
         std::filesystem::directory_iterator(compression_json_folder)) {
        compression_json_files.emplace_back(config_path.path().string());
    }
    std::sort(compression_json_files.begin(), compression_json_files.end());
    for (auto config_file : compression_json_files) {
        benchmarks.emplace_back();
        benchmarks.back().name = str_pre + "lp" + str_post;
        benchmarks.back().settings = default_ss;
        benchmarks.back().settings.storage_prec =
            gko::solver::cb_gmres::storage_precision::use_sz;
        benchmarks.back().settings.lp_config = config_file;
    }
    */

    // Note: The time might not be significantly different since the matrix is
    //       quite small
    const std::array<int, 7> widths{15, 12, 11, 17, 16, 15, 15};
    const char delim = ';';
    std::cout << std::setw(widths[0]) << "Name" << delim << std::setw(widths[1])
              << "Time [s]" << delim << std::setw(widths[2]) << "Iterations"
              << delim << std::setw(widths[3]) << "res norm before" << delim
              << std::setw(widths[4]) << "res norm after" << delim
              << std::setw(widths[5]) << "rel res norm" << delim
              << std::setw(widths[6]) << "compression" << '\n';
    for (auto&& val : benchmarks) {
        // if (val.name.find("sz4") == std::string::npos) { continue; }
        val.result = benchmark_solver(exec, val.settings, A, b.get(), x.get());
        const auto file_path = val.settings.lp_config;
        auto begin_file_name = file_path.rfind('/');
        begin_file_name =
            begin_file_name == std::string::npos ? 0 : begin_file_name + 1;
        const auto file_name = file_path.substr(
            begin_file_name, file_path.size() - begin_file_name - 5);
        std::cout << std::setw(widths[0]) << val.name << delim
                  << std::setw(widths[1]) << val.result.time_s << delim
                  << std::setw(widths[2]) << val.result.iters << delim
                  << std::setw(widths[3]) << val.result.init_res_norm << delim
                  << std::setw(widths[4]) << val.result.res_norm << delim
                  << std::setw(widths[5])
                  << val.result.res_norm / val.result.init_res_norm << delim
                  << std::setw(widths[6]) << file_name << '\n';
    }
}


int main(int argc, char* argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [path/to/matrix.mtx] [path/to/compresson/json/folder]"
                     " [max_iters] [rel_res_norm] [{double,float}] "
                     "[{cuda,omp,hip,dpcpp,reference}]"
                  << std::endl;
        std::exit(-1);
    }

    int c_param = 1;  // stores the current parameter index
    const std::string matrix_path =
        argc >= c_param + 1 ? argv[c_param] : "data/A.mtx";
    const std::string compression_json_folder =
        argc >= ++c_param + 1 ? argv[c_param] : "lp_configs";
    const unsigned max_iters =
        argc >= ++c_param + 1 ? std::stoi(argv[c_param]) : 101;
    const double rel_res_norm =
        argc >= ++c_param + 1 ? std::stof(argv[c_param]) : 1e-16;
    const std::string precision =
        argc >= ++c_param + 1 ? argv[c_param] : "double";
    const auto executor_string =
        argc >= ++c_param + 1 ? argv[c_param] : "reference";
    // Map which generates the appropriate executor
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    if (precision == std::string("double")) {
        run_benchmarks<double, int>(exec, matrix_path, compression_json_folder,
                                    max_iters, rel_res_norm);
    } else if (precision == std::string("float")) {
        run_benchmarks<float, int>(exec, matrix_path, compression_json_folder,
                                   max_iters, rel_res_norm);
    } else {
        std::cerr << "Unknown precision string \"" << argv[4]
                  << "\". Supported values: \"double\", \"float\"\n";
        std::exit(-1);
    }
}
