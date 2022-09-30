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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>

std::unique_ptr<gko::matrix::Csr<double>> gen_laplacian(
    std::shared_ptr<const gko::Executor> exec, int grid, bool is_2d = true)
{
    int size = (is_2d ? grid * grid : grid * grid * grid);
    gko::matrix_data<> mtx_data{gko::dim<2>(size, size)};
    int npt = (is_2d ? 5 : 7);
    double coef_val = -1.0 / (npt - 1);
    if (is_2d) {
        int y[] = {0, -1, 0, 1, 0};
        int x[] = {-1, 0, 0, 0, 1};
        double coef[] = {coef_val, coef_val, 1, coef_val, coef_val};
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                auto c = i * grid + j;
                for (int idx = 0; idx < npt; idx++) {
                    auto ii = i + x[idx];
                    auto jj = j + y[idx];
                    auto cc = ii * grid + jj;
                    if (0 <= ii && ii < grid && 0 <= jj && jj < grid) {
                        mtx_data.nonzeros.emplace_back(c, cc, coef[idx]);
                    }
                }
            }
        }
    } else {
        // 3d
        int z[] = {0, 0, -1, 0, 1, 0, 0};
        int y[] = {0, -1, 0, 0, 0, 1, 0};
        int x[] = {-1, 0, 0, 0, 0, 0, 1};
        double coef[] = {coef_val, coef_val, coef_val, 1,
                         coef_val, coef_val, coef_val};
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                for (int k = 0; k < grid; k++) {
                    auto c = i * grid * grid + j * grid + k;
                    for (int idx = 0; idx < npt; idx++) {
                        auto ii = i + x[idx];
                        auto jj = j + y[idx];
                        auto kk = k + z[idx];
                        auto cc = ii * grid * grid + jj * grid + kk;
                        if (0 <= ii && ii < grid && 0 <= jj && jj < grid &&
                            0 <= kk && kk < grid) {
                            mtx_data.nonzeros.emplace_back(c, cc, coef[idx]);
                        }
                    }
                }
            }
        }
    }
    mtx_data.ensure_row_major_order();
    auto mtx = gko::matrix::Csr<double>::create(
        exec, std::make_shared<gko::matrix::Csr<>::classical>());
    mtx->read(mtx_data);

    return std::move(mtx);
}


std::unique_ptr<gko::matrix::Csr<double>> gen_mask(
    std::shared_ptr<const gko::Executor> exec, int grid)
{
    int size = grid * grid;
    int y[] = {0, -1, 0, 1, 0};
    int x[] = {-1, 0, 0, 0, 1};
    double coef[] = {4, 3, 2, 1, 0};
    gko::matrix_data<> mtx_data{gko::dim<2>(size, size)};
    for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
            auto c = i * grid + j;
            for (int k = 0; k < 5; k++) {
                auto ii = i + x[k];
                auto jj = j + y[k];
                auto cc = ii * grid + jj;
                if (0 <= ii && ii < grid && 0 <= jj && jj < grid) {
                    mtx_data.nonzeros.emplace_back(c, cc, coef[k]);
                }
            }
        }
    }
    mtx_data.ensure_row_major_order();
    auto mtx = gko::matrix::Csr<double>::create(
        exec, std::make_shared<gko::matrix::Csr<>::classical>());
    mtx->read(mtx_data);

    return std::move(mtx);
}


void print(int i, std::uint64_t n, std::ostream& output)
{
    std::uint64_t cap[] = {0xFFFull << (4 * 12), 0xFFFull << (3 * 12),
                           0xFFFull << (2 * 12), 0xFFFull << (1 * 12),
                           0xFFFull << (0 * 12)};
    output << std::setw(6) << i;
    for (int i = 0; i < 5; i++) {
        output << ", " << std::setw(6) << ((n & cap[i]) >> ((4 - i) * 12));
    }
    output << std::endl;
}

void print_time(int i, std::uint64_t n, std::ostream& output)
{
    // std::cout << std::setw(6);
    output << std::setw(6) << i << ", " << std::setw(40)
           << ((n >> 32) & 0xFFFFFFFF) << ", " << std::setw(40)
           << (n & 0xFFFFFFFF) << std::endl;
    // std::cout << std::endl;
}

double avg(std::vector<double>& input)
{
    auto sum = std::accumulate(input.begin(), input.end(), 0.0);
    return sum / input.size();
}

// th: 0(min) 1(25 th) 2(med) 3(75th) 4(max)
double quartile(std::vector<double>& input, int th)
{
    if (th == 4) {
        return input.back();
    } else if (th == 0) {
        return input.front();
    } else {
        int k = (th * (input.size() + 1) / 4) - 1;
        double alpha = (th * (input.size() + 1) / 4.0) - k;
        return (1 - alpha) * input.at(k) + alpha * input.at(k + 1);
    }
}

int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using async_solver = gko::solver::AsyncRichardson<ValueType>;
    using normal_solver = gko::solver::Richardson<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // Print help on how to execute this example.
    if (argc < 6 || (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [type] [normal/normal_3d/flow/halfflow/time] "
                     "[problem_size] [iteration] [folder(optional)]"
                  << std::endl;
        std::exit(-1);
    }

    std::string executor_string(argv[1]);
    std::string type_string(argv[2]);
    std::string check_string(argv[3]);
    bool is_2d = (check_string != "normal_3d");
    int problem_size = std::stoi(argv[4]);
    int iteration = std::stoi(argv[5]);
    std::string folder_string;
    if (argc >= 7) {
        folder_string = argv[6];
    }

    std::cout << "Perform " << type_string << " richardson on "
              << executor_string << std::endl;
    if (check_string != "normal_3d") {
        std::cout << "Problem size " << problem_size << " dim "
                  << problem_size * problem_size << std::endl;
    } else {
        std::cout << "Problem size " << problem_size << " dim "
                  << problem_size * problem_size * problem_size << std::endl;
    }
    std::cout << "update " << iteration << " times" << std::endl;
    std::cout << "check " << check_string << std::endl;
    if (folder_string != "" && check_string != "normal") {
        std::cout << "write the detail to folder - " << folder_string
                  << std::endl;
    }
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

    std::shared_ptr<mtx> A = nullptr;
    if (check_string == "normal" || check_string == "normal_3d") {
        A = gko::share(gen_laplacian(exec, problem_size, is_2d));
    } else {
        A = gko::share(gen_mask(exec, problem_size));
    }
    auto b_host =
        vec::create(exec->get_master(), gko::dim<2>(A->get_size()[1], 1));
    // generate right hand side
    std::default_random_engine rand_engine(77);
    auto dist = std::uniform_real_distribution<>(-0.125, 0.125);
    for (int i = 0; i < b_host->get_size()[0]; i++) {
        for (int j = 0; j < b_host->get_size()[1]; j++) {
            b_host->at(i, j) = dist(rand_engine);
        }
    }
    auto b = b_host->clone(exec);
    auto x = vec::create(exec, gko::dim<2>(A->get_size()[0], 1));
    x->fill(0.0);

    std::shared_ptr<gko::LinOpFactory> solver_gen = nullptr;
    if (type_string == std::string("async")) {
        solver_gen =
            async_solver::build()
                .with_max_iters(iteration)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(iteration).on(
                        exec))
                .with_check(check_string)
                .on(exec);
    } else {
        solver_gen =
            normal_solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(iteration).on(
                        exec))
                .on(exec);
    }
    auto solver = solver_gen->generate(A);

    // warmup
    for (int i = 0; i < 10; i++) {
        auto x_clone = x->clone();
        exec->synchronize();
        solver->apply(lend(b), lend(x_clone));
    }
    exec->synchronize();
    // Finally, solve the system. The solver, being a gko::LinOp, can be applied
    // to a right hand side, b to
    // obtain the solution, x.
    int num_re = 100;
    auto residual_norm = vec::create(exec, gko::dim<2>{1, b->get_size()[1]});
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto one = gko::initialize<vec>({1.0}, exec);
    std::vector<double> norm(num_re, 0.0);
    std::vector<double> time(num_re, 0.0);
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    double initial_norm = 0;
    if (check_string == "normal" || check_string == "normal_3d") {
        b->compute_norm2(residual_norm.get());
        initial_norm = exec->copy_val_to_host(residual_norm->get_values());
    }
    for (int i = 0; i < num_re; i++) {
        auto x_clone = x->clone();
        exec->synchronize();
        start = std::chrono::steady_clock::now();
        solver->apply(lend(b), lend(x_clone));
        exec->synchronize();
        stop = std::chrono::steady_clock::now();
        if (check_string == "normal" || check_string == "normal_3d") {
            auto b_clone = b->clone();
            A->apply(lend(neg_one), lend(x_clone), lend(one), lend(b_clone));
            b_clone->compute_norm2(residual_norm.get());
            norm.at(i) = exec->copy_val_to_host(residual_norm->get_values()) /
                         initial_norm;
        } else if (folder_string != "") {
            auto host_x = x_clone->clone(exec->get_master());
            std::stringstream output_file;
            output_file << folder_string << "/" << check_string << "_update"
                        << iteration << "_re" << i << ".csv";
            std::ofstream output(output_file.str());
            for (int i = 0; i < host_x->get_size()[0]; i++) {
                std::uint64_t n = 0;
                std::memcpy(&n, host_x->get_const_values() + i,
                            sizeof(std::uint64_t));
                if (check_string == "time") {
                    print_time(i, n, output);
                } else {
                    print(i, n, output);
                }
            }
        }
        std::chrono::duration<double> duration_time = stop - start;
        time.at(i) = duration_time.count();
    }

    std::cout << type_string << " grid " << problem_size << " update "
              << iteration << ", mean, min, q1, median, q3, max" << std::endl;
    std::sort(time.begin(), time.end());
    std::cout << "time, " << avg(time) << ", " << quartile(time, 0) << ", "
              << quartile(time, 1) << ", " << quartile(time, 2) << ", "
              << quartile(time, 3) << ", " << quartile(time, 4) << std::endl;
    if (check_string == "normal" || check_string == "normal_3d") {
        std::sort(norm.begin(), norm.end());
        std::cout << "relative_norm, " << avg(norm) << ", " << quartile(norm, 0)
                  << ", " << quartile(norm, 1) << ", " << quartile(norm, 2)
                  << ", " << quartile(norm, 3) << ", " << quartile(norm, 4)
                  << std::endl;
    }
}
