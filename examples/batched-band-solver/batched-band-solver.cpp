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
using mtx_type = gko::matrix::BatchBand<value_type>;
using batch_csr = gko::matrix::BatchCsr<value_type>;
using batch_band_solver = gko::solver::BatchBandSolver<value_type>;


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [dir_path] [problem_name] [num_systems] "
                     "[num_duplications] "
                     "[band solver approach] [blocked solve panel size] "
                     "[file_timings]"
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

    // @sect3{Other optional command-line arguments}
    const std::string dir_name = argc >= 3 ? argv[2] : "data/";

    // Name of the problem
    const std::string problem_name =
        argc >= 4 ? argv[3] : "banded_N_100_KL_7_KU_8";

    // Number of linear systems to read from files.
    const size_type num_systems = argc >= 5 ? std::atoi(argv[4]) : 2;

    // Number of times to duplicate whatever systems are read from files.
    const size_type num_duplications = argc >= 6 ? std::atoi(argv[5]) : 3;

    // Approach
    enum gko::solver::batch_band_solve_approach approach;
    const std::string approach_str = argc >= 7 ? argv[6] : "unblocked";
    if (approach_str == std::string("unblocked")) {
        approach = gko::solver::batch_band_solve_approach::unblocked;
    } else if (approach_str == std::string("blocked")) {
        approach = gko::solver::batch_band_solve_approach::blocked;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    const int blocked_solve_panel_size = argc >= 8 ? std::atoi(argv[7]) : 2;

    const char* log_file = argc >= 9 ? argv[8] : "timings_file.txt";

    // @sect3{Read batch from files}
    auto data = std::vector<gko::matrix_data<value_type>>(num_systems);

    for (size_type i = 0; i < data.size(); ++i) {
        const std::string mat_str = "A.mtx";
        const std::string fbase =
            dir_name + problem_name + "/" + std::to_string(i) + "/";
        std::string fname = fbase + mat_str;
        std::cout << "\nReading file: " << fname << std::endl;
        std::ifstream mtx_fd(fname);
        data[i] = gko::read_raw<value_type>(mtx_fd);
    }

    auto single_batch = mtx_type::create(exec);
    single_batch->read_band_matrix(data);
    auto single_batch_csr = batch_csr::create(exec);
    single_batch_csr->read(data);

    // We can duplicate the batch a few times if we wish.
    std::shared_ptr<mtx_type> A =
        mtx_type::create(exec, num_duplications, single_batch.get());
    std::shared_ptr<batch_csr> A_csr =
        batch_csr::create(exec, num_duplications, single_batch_csr.get());

    // Create RHS
    const auto nrows = A->get_size().at(0)[0];

    const size_type num_total_systems = num_systems * num_duplications;

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
    auto x_dense_direct_solver = vec_type::create(exec, b->get_size());

    // @sect3{Create the batch solver factory}
    // Create a batched solver factory with relevant parameters.
    auto solver_gen =
        batch_band_solver::build()
            .with_batch_band_solution_approach(approach)
            .with_blocked_solve_panel_size(blocked_solve_panel_size)
            .on(exec);

    // @sect3{Generate and solve}
    // Generate the batch solver from the batch matrix
    auto solver = solver_gen->generate(A);

    auto solver_dense_direct_gen =
        gko::solver::BatchDirect<value_type>::build().on(exec);
    auto solver_dense_direct = solver_dense_direct_gen->generate(A_csr);

    const int num_rounds = 10;
    const int leave_first = 2;

    for (int i = 0; i < leave_first; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    exec->synchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_rounds; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    exec->synchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    double total_time_millisec =
        (double)(std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                       start))
            .count() /
        (double)1000;

    double av_time_millisec = total_time_millisec / num_rounds;

    std::cout << "\nThe entire solve took " << av_time_millisec
              << " milliseconds." << std::endl;


    {
        for (int i = 0; i < leave_first; i++) {
            // Solve the batch system
            solver_dense_direct->apply(lend(b), lend(x_dense_direct_solver));
        }

        exec->synchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_rounds; i++) {
            // Solve the batch system
            solver_dense_direct->apply(lend(b), lend(x_dense_direct_solver));
        }

        exec->synchronize();
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        double total_time_millisec =
            (double)(std::chrono::duration_cast<std::chrono::microseconds>(
                         stop - start))
                .count() /
            (double)1000;

        double av_time_millisec = total_time_millisec / num_rounds;

        std::cout << "\nThe entire dense direct solve took " << av_time_millisec
                  << " milliseconds." << std::endl;
    }

    auto x_host = gko::share(gko::clone(exec->get_master(), x.get()));
    auto x_dd_host = gko::share(gko::clone(exec->get_master(), x.get()));
    // for (int i = 0; i < num_total_systems; i++) {
    //     std::cout << "Batch idx: " << i << std::endl;
    //     for (int k = 0; k < nrows; k++) {
    //         std::cout << "x[" << k
    //                   << "]: " << x_host->get_const_values()[i * nrows + k]
    //                   << "   " << x_dd_host->get_const_values()[i * nrows +
    //                   k]
    //                   << std::endl;
    //     }
    // }

    std::ofstream timings_file;
    timings_file.open(log_file, std::ofstream::app);
    timings_file << num_total_systems << "  " << av_time_millisec << " \n";
    timings_file.close();

    return 0;
}
