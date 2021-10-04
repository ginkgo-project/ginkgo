/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


// Some shortcuts
using ValueType = double;
using RealValueType = gko::remove_complex<ValueType>;
using IndexType = int;

using vec_type = gko::matrix::Dense<ValueType>;
using real_vec_type = gko::matrix::Dense<RealValueType>;
using mtx_type = gko::matrix::Csr<ValueType, IndexType>;
using solver_type = gko::solver::Bicgstab<ValueType>;


std::unique_ptr<mtx_type> read_mtx(std::shared_ptr<const gko::Executor> exec,
                                   std::string mat_path, int num_systems,
                                   int num_duplications);

std::unique_ptr<vec_type> read_vec(std::shared_ptr<const gko::Executor> exec,
                                   std::string mat_path, int num_systems,
                                   int num_duplications);


int main(int argc, char* argv[])
{
    // Print version information
    // std::cout << gko::version_info::get() << std::endl;

    if (argc < 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " [executor] [top-level batch directory] [number of systems]"
            << " (number of times to duplicate [1])" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argv[1];
    // Figure out where to run the code
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
    const std::string mat_path = argv[2];
    const int num_systems = std::atoi(argv[3]);
    const int num_duplications = argc >= 5 ? std::atoi(argv[4]) : 1;
    const bool benchmark = argc >= 6 ? std::atoi(argv[5]) : 0;

    // Read data
    std::shared_ptr<mtx_type> A =
        read_mtx(exec, mat_path, num_systems, num_duplications);
    auto b = read_vec(exec, mat_path, num_systems, num_duplications);
    std::cout << "System has " << A->get_size()[0] << " rows and "
              << A->get_num_stored_elements() << " non-zeros." << std::endl;

    // Initialize solution vector
    auto h_x = vec_type::create(exec->get_master(), b->get_size());
    for (size_t i = 0; i < h_x->get_size()[0]; i++) {
        for (size_t j = 0; j < h_x->get_size()[1]; j++) {
            h_x->at(i, j) = 0.0;
        }
    }
    auto x = vec_type::create(exec);
    x->copy_from(gko::lend(h_x));

    //// Generate incomplete factors using ParILU
    // auto par_ilu_fact =
    //    gko::factorization::ParIlu<ValueType, IndexType>::build()
    //        .with_iterations(10u)
    //        .on(exec);
    //// Generate concrete factorization for input matrix
    // auto par_ilu = par_ilu_fact->generate(A);

    //// Generate an ILU preconditioner factory by setting lower and upper
    //// triangular solver - in this case the exact triangular solves
    // auto ilu_pre_factory =
    //    gko::preconditioner::Ilu<
    //        gko::preconditioner::Isai<gko::preconditioner::isai_type::lower,
    //                                  ValueType, IndexType>,
    //        gko::preconditioner::Isai<gko::preconditioner::isai_type::upper,
    //                                  ValueType, IndexType>,
    //        false>::build()
    //        .on(exec);

    //// Use incomplete factors to generate ILU preconditioner
    // auto preconditioner = ilu_pre_factory->generate(gko::share(par_ilu));

    // Jacobi preconditioner
    using prec_type = gko::preconditioner::Jacobi<ValueType, IndexType>;
    auto jacobi_factory = prec_type::build().with_max_block_size(1u).on(exec);
    auto preconditioner = jacobi_factory->generate(A);

    // Use preconditioner inside GMRES solver factory
    // Generating a solver factory tied to a specific preconditioner makes sense
    // if there are several very similar systems to solve, and the same
    // solver+preconditioner combination is expected to be effective.
    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec);
    const RealValueType reduction_factor{1e-8};
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(reduction_factor)
                        .on(exec);
    std::shared_ptr<gko::log::Convergence<ValueType>> clog =
        gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(clog);
    tol_stop->add_logger(clog);

    auto solver_factory =
        solver_type::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .with_generated_preconditioner(gko::share(preconditioner))
            .on(exec);
    auto solver = solver_factory->generate(A);

    // Solve system
    auto start = std::chrono::steady_clock::now();
    solver->apply(gko::lend(b), gko::lend(x));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken for solve = " << elapsed.count() << " s"
              << std::endl;
    std::cout << "Number of iterations = " << clog->get_num_iterations()
              << std::endl;

    // Calculate residual
    auto one = gko::initialize<vec_type>({1.0}, exec);
    auto neg_one = gko::initialize<vec_type>({-1.0}, exec);
    auto res = gko::initialize<real_vec_type>({0.0}, exec->get_master());
    A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one), gko::lend(b));
    b->compute_norm2(gko::lend(res));

    std::cout << "Residual norm sqrt(r^T r):\n" << res->at(0, 0) << std::endl;

    return 0;
}

std::unique_ptr<mtx_type> read_mtx(std::shared_ptr<const gko::Executor> exec,
                                   std::string mat_path, const int num_systems,
                                   const int num_duplications)
{
    std::vector<std::unique_ptr<mtx_type>> batchentries;
    for (int i = 0; i < num_systems; ++i) {
        const std::string mat_str = "A.mtx";
        const std::string fbase = mat_path + "/" + std::to_string(i) + "/";
        const std::string fname = fbase + mat_str;
        std::ifstream mtx_fd(fname);
        auto data = gko::read_raw<ValueType>(mtx_fd);
        auto mat = mtx_type::create(exec);
        mat->read(data);
        batchentries.emplace_back(std::move(mat));
    }
    for (int id = 0; id < num_duplications - 1; id++) {
        for (int i = 0; i < num_systems; ++i) {
            auto mat = mtx_type::create(exec);
            mat->copy_from(batchentries[i].get());
            batchentries.emplace_back(std::move(mat));
        }
    }
    return gko::create_block_diagonal_matrix(exec, batchentries);
}

std::unique_ptr<vec_type> read_vec(std::shared_ptr<const gko::Executor> exec,
                                   std::string mat_path, const int num_systems,
                                   const int num_duplications)
{
    std::vector<std::unique_ptr<vec_type>> batchentries;
    for (int i = 0; i < num_systems; ++i) {
        const std::string b_str = "b.mtx";
        const std::string fbase = mat_path + "/" + std::to_string(i) + "/";
        const std::string fname = fbase + b_str;
        std::ifstream mtx_fd(fname);
        auto data = gko::read_raw<ValueType>(mtx_fd);
        auto mat = vec_type::create(exec);
        mat->read(data);
        batchentries.emplace_back(std::move(mat));
    }
    for (int id = 0; id < num_duplications - 1; id++) {
        for (int i = 0; i < num_systems; ++i) {
            auto mat = vec_type::create(exec);
            mat->copy_from(batchentries[i].get());
            batchentries.emplace_back(std::move(mat));
        }
    }
    return gko::concatenate_dense_matrices(exec, batchentries);
}
