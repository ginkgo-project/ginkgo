/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


// @sect3{Type aliases for convenience}
// Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
// with one column/one row. The advantage of this concept is that using
// multiple vectors is a now a natural extension of adding columns/rows are
// necessary.
using value_type = double;
using real_type = gko::remove_complex<value_type>;
using index_type = int;
using size_type = gko::size_type;
using vec_type = gko::matrix::BatchDense<value_type>;
using real_vec_type = gko::matrix::BatchDense<real_type>;
using mtx_type = gko::matrix::BatchTridiagonal<value_type>;
using batch_tridiag_solver = gko::solver::BatchTridiagonalSolver<value_type>;


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [problem_name] [num_duplications] "
                     "[num_WM_steps] [subwarp_size] [tridiag approach]"
                  << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for an gko::OmpExecutor, which uses OpenMP
    // multi-threading in most of its kernels,
    // a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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

    // @sect3{Read batch from files}
    // Name of the problem
    const std::string problem_name = argc >= 3 ? argv[2] : "gallery_lesp_100";

    // Number of times to duplicate whatever systems are read from files.
    const size_type num_duplications = argc >= 4 ? std::atoi(argv[3]) : 1;

    // Number of WM steps
    const int number_WM_steps = argc >= 5 ? std::atoi(argv[4]) : 2;

    // WM_pGE subwarp size
    const int subwarp_size = argc >= 6 ? std::atoi(argv[5]) : 16;

    // Approach
    enum gko::solver::batch_tridiag_solve_approach approach;
    const std::string approach_str = argc >= 7 ? argv[6] : "WM_pGE_app1";
    if (approach_str == std::string("WM_pGE_app1")) {
        approach = gko::solver::batch_tridiag_solve_approach::WM_pGE_app1;
    } else if (approach_str == std::string("WM_pGE_app2")) {
        approach = gko::solver::batch_tridiag_solve_approach::WM_pGE_app2;
    } else if (approach_str == std::string("vendor_provided")) {
        approach = gko::solver::batch_tridiag_solve_approach::vendor_provided;
    }


    const std::string mat_str = problem_name + ".mtx";
    const std::string fbase = "/home/hp/Desktop/Tridiagonal_test_matrices/";
    std::string fname = fbase + mat_str;
    std::cout << "\n\nfile to be read: " << fname << std::endl;
    std::ifstream mtx_fd(fname);

    auto data = std::vector<gko::matrix_data<value_type>>(1);
    data[0] = gko::read_raw<value_type>(mtx_fd);

    auto single_batch = mtx_type::create(exec);
    single_batch->read(data);

    // We can duplicate the batch a few times if we wish.
    std::shared_ptr<mtx_type> A =
        mtx_type::create(exec, num_duplications, single_batch.get());

    // Create RHS
    const auto nrows = A->get_size().at(0)[0];
    /*
    std::cout << "num_rows is: " << nrows << std::endl;

    auto A_host = gko::clone(exec->get_master(), A.get());
    for (int i = 0; i < nrows; i++) {
        std::cout << "\nsub_diag[" << i
                  << "]: " << A_host->get_const_sub_diagonal()[i];
    }
    for (int i = 0; i < nrows; i++) {
        std::cout << "\nmain_diag[" << i
                  << "]: " << A_host->get_const_main_diagonal()[i];
    }
    for (int i = 0; i < nrows; i++) {
        std::cout << "\nsuper_diag[" << i
                  << "]: " << A_host->get_const_super_diagonal()[i];
    }
    */

    const size_type num_total_systems = num_duplications;

    // std::cout << "\n\nNum total systems: " << num_total_systems << std::endl;

    auto host_b = vec_type::create(
        exec->get_master(),
        gko::batch_dim<2>(num_total_systems, gko::dim<2>(nrows, 1)));

    for (size_type isys = 0; isys < num_total_systems; isys++) {
        for (int irow = 0; irow < nrows; irow++) {
            host_b->at(isys, irow, 0) = static_cast<value_type>(rand());
        }
    }

    auto b = vec_type::create(exec);
    b->copy_from(host_b.get());

    auto x = vec_type::create(exec, b->get_size());

    // @sect3{Create the batch solver factory}
    // Create a batched solver factory with relevant parameters.
    auto solver_gen = batch_tridiag_solver::build()
                          .with_batch_tridiagonal_solution_approach(approach)
                          .with_num_WM_steps(number_WM_steps)
                          .with_WM_pGE_subwarp_size(subwarp_size)
                          .on(exec);

    // @sect3{Generate and solve}
    // Generate the batch solver from the batch matrix
    auto solver = solver_gen->generate(A);

    const int num_rounds = 1;
    const int leave_first = 0;

    for (int i = 0; i < leave_first; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < num_rounds; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1) /
        num_rounds;
    std::cout << "Entire solve took " << 1000 * time_span.count()
              << " milliseconds." << std::endl;

    // auto vec_b = b->unbatch(); TODO: //seg fault???
    // gko::write(std::ofstream(std::string("b.mtx")), vec_b[0].get());

    auto host_x = gko::clone(exec->get_master(), x.get());
    auto vec_x = host_x->unbatch();
    gko::write(std::cout, vec_x[0].get());

    return 0;
}
