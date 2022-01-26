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
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <tuple>


template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    ValueType value)
{
    auto res = gko::matrix::Dense<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType, int>(size, value));
    return res;
}


template <typename ValueType>
std::enable_if_t<!gko::is_complex_s<ValueType>::value,
                 std::unique_ptr<gko::matrix::Dense<ValueType>>>
create_matrix_sin(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
{
    auto h_res =
        gko::matrix::Dense<ValueType>::create(exec->get_master(), size);
    for (gko::size_type i = 0; i < size[0]; ++i) {
        for (gko::size_type j = 0; j < size[1]; ++j) {
            h_res->at(i, j) = std::sin(static_cast<ValueType>(i));
        }
    }
    auto res = gko::matrix::Dense<ValueType>::create(exec);
    h_res->move_to(res.get());
    return res;
}

// Note: complex values are assigned s[i, j] = {sin(2 * i), sin(2 * i + 1)}
template <typename ValueType>
std::enable_if_t<gko::is_complex_s<ValueType>::value,
                 std::unique_ptr<gko::matrix::Dense<ValueType>>>
create_matrix_sin(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
{
    using rc_vtype = gko::remove_complex<ValueType>;
    auto h_res =
        gko::matrix::Dense<ValueType>::create(exec->get_master(), size);
    for (gko::size_type i = 0; i < size[0]; ++i) {
        for (gko::size_type j = 0; j < size[1]; ++j) {
            h_res->at(i, j) =
                ValueType{std::sin(static_cast<rc_vtype>(2 * i)),
                          std::sin(static_cast<rc_vtype>(2 * i + 1))};
        }
    }
    auto res = gko::matrix::Dense<ValueType>::create(exec);
    h_res->move_to(res.get());
    return res;
}


template <typename ValueType>
void run_solver_test(std::shared_ptr<const gko::Executor> exec,
                     std::string mtx_path)
{
    using IndexType = int;
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using gmres = gko::solver::Gmres<ValueType>;
    using cb_gmres = gko::solver::CbGmres<ValueType>;
    const auto one = gko::initialize<vec>({1.0}, exec);
    const auto neg_one = gko::initialize<vec>({-1.0}, exec);

    auto A = share(gko::read<mtx>(std::ifstream(mtx_path), exec));
    const auto mtx_size = A->get_size();

    const auto vec_size = gko::dim<2>{mtx_size[0], 1};
    // Prepare initial x and the RHS
    auto x_init = create_matrix<ValueType>(exec, vec_size, 0);
    auto x = x_init->clone();

    // b = A * (x_sin / ||x_sin||)  with x_sin[i] = sin(i)
    auto b = gko::matrix::Dense<ValueType>::create(exec, vec_size);

    auto x_sin = create_matrix_sin<ValueType>(exec, vec_size);
    auto scalar = gko::matrix::Dense<RealValueType>::create(
        exec->get_master(), gko::dim<2>{1, vec_size[1]});
    x_sin->compute_norm2(scalar.get());
    for (gko::size_type i = 0; i < vec_size[1]; ++i) {
        scalar->at(0, i) = gko::one<RealValueType>() / scalar->at(0, i);
    }
    // normalize sin-vector
    if (gko::is_complex_s<ValueType>::value) {
        x_sin->scale(scalar->make_complex().get());
    } else {
        x_sin->scale(scalar.get());
    }
    A->apply(x_sin.get(), b.get());

    auto res = b->clone();
    auto res_norm = gko::matrix::Dense<RealValueType>::create(
        exec, gko::dim<2>(1, vec_size[1]));
    auto b_norm_host = gko::matrix::Dense<RealValueType>::create(
        exec->get_master(), gko::dim<2>{1, vec_size[1]});
    auto rel_res_norm_host = res_norm->clone(exec->get_master());


    // Create convergence logger to retrieve iteration count
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);

    const RealValueType reduction_factor{1e-12};
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(25000u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(reduction_factor)
                                   .on(exec));
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    auto cg_solver_gen =
        cg::build().with_criteria(iter_stop, tol_stop).on(exec);
    cg_solver_gen->add_logger(logger);
    auto cg_solver = cg_solver_gen->generate(A);

    auto gmres_solver_gen =
        cb_gmres::build()
            .with_criteria(iter_stop, tol_stop)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::keep)
            .on(exec);
    gmres_solver_gen->add_logger(logger);
    auto gmres_solver = gmres_solver_gen->generate(A);

    auto cb_gmres_solver_gen =
        cb_gmres::build()
            .with_criteria(iter_stop, tol_stop)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::reduce1)
            .on(exec);
    cb_gmres_solver_gen->add_logger(logger);
    auto cb_gmres_solver = cb_gmres_solver_gen->generate(A);

    exec->synchronize();

    using ar_element_t = std::tuple<std::unique_ptr<gko::LinOp>, std::string>;
    std::array<ar_element_t, 3> solvers{
        ar_element_t{std::move(cg_solver), "CG"},
        ar_element_t{std::move(gmres_solver), "GMRES"},
        ar_element_t{std::move(cb_gmres_solver), "CB-GMRES"}};

    for (auto&& el : solvers) {
        const auto& solver_name = std::get<1>(el);
        std::cout << "\nSolver " << solver_name << '\n';
        x->copy_from(gko::lend(x_init));
        res->copy_from(gko::lend(b));
        A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one),
                 gko::lend(res));
        res->compute_norm2(gko::lend(res_norm));
        std::cout << "Initial residual norm sqrt(r^t r):\n";
        gko::write(std::cout, gko::lend(res_norm));
        // std::cout << '\n';

        // Finally, solve the system. The solver, being a gko::LinOp, can be
        // applied to a right hand side, b to obtain the solution, x.
        auto tic = std::chrono::steady_clock::now();
        std::get<0>(el)->apply(lend(b), lend(x));
        exec->synchronize();
        auto toc = std::chrono::steady_clock::now();

        // Print the solution to the command line.

        b->compute_norm2(gko::lend(b_norm_host));
        res->copy_from(gko::lend(b));
        A->apply(lend(one), lend(x), lend(neg_one), lend(res));
        res->compute_norm2(lend(res_norm));
        rel_res_norm_host->copy_from(gko::lend(res_norm));
        for (int i = 0; i < vec_size[1]; ++i) {
            rel_res_norm_host->at(0, i) /= b_norm_host->at(0, i);
        }

        std::cout << "Final residual norm sqrt(r^T r):\n";
        write(std::cout, lend(res_norm));
        // std::cout << '\n';
        std::cout << "Relative residual norm:" << '\n';
        write(std::cout, gko::lend(rel_res_norm_host));
        std::cout << solver_name
                  << " iteration count: " << logger->get_num_iterations()
                  << '\n';
        std::cout << solver_name << " execution time: "
                  << std::chrono::duration<double>(toc - tic).count() * 1000
                  << " ms\n";
    }
}


int main(int argc, char* argv[])
{
    using ValueType = double;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc < 3 || (argc == 2 && (std::string(argv[1]) == "--help"))) {
        std::cerr << "Usage: " << argv[0] << " Matrix_path "
                  << " [executor] " << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 3 ? argv[2] : "reference";
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

    std::cout << "\nDouble precision:\n";
    run_solver_test<double>(exec, argv[1]);
    std::cout << "\n\nSingle precision:\n";
    run_solver_test<float>(exec, argv[1]);
}
