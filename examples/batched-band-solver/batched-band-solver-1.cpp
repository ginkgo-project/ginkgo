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


// @sect3{'Application' structures and functions}
// Structure to simulate application data related to the linear systems
// to be solved.
//
// We use raw pointers below to demonstrate how to handle the situation when
// the application only gives us raw pointers. Ideally, one should use
// Ginkgo's gko::Array class here.
struct ApplSysData {
    // Number of small systems in the batch.
    size_type nsystems;
    // Number of rows in each system.
    int nrows;
    // Number of sub-diagonals in each system matrix.
    int num_sub_diags;
    // Number of  super-diagonals in each system matrix.
    int num_super_diags;
    // Column major batch band array (in the prescribed fashion)
    const value_type* batch_band_array;
    // RHS vectors for all systems in the batch, concatenated
    const value_type* all_rhs;
};

// Generates a batch of tridiagonal systems.
//
// @param nrows  Number of rows in each system.
// @param num_sub_diags Number of sub-diagonals in each system matrix
// @param num_super_diags  Number of  super-diagonals in each system matrix
// @param nsystems  Number of systems in the batch.
// @param exec  The device executor to use for the solver.
//   Normally, the application may not deal with Ginkgo executors, nor do we
//   need it to. Here, we use the executor for backend-independent device
//   memory allocation. The application, for example, might assume Hip
//   (for AMD GPUs) and use `hipMalloc` directly.
ApplSysData appl_generate_system(const int nrows, const int num_sub_diags,
                                 const int num_super_diags,
                                 const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec);

// Deallocate application data.
void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec);


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [N] [KL] [KU] [num_systems] "
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
    const int nrows = argc >= 3 ? std::atoi(argv[2]) : 15;

    const int KL = argc >= 4 ? std::atoi(argv[3]) : 3;

    const int KU = argc >= 5 ? std::atoi(argv[4]) : 4;

    // Number of linear systems
    const size_type num_systems = argc >= 6 ? std::atoi(argv[5]) : 2;

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

    // @sect3{Generate data}
    // The "application" generates the batch of linear systems on the device
    auto appl_sys = appl_generate_system(nrows, KL, KU, num_systems, exec);
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<>(num_systems, gko::dim<2>(nrows, nrows));
    auto batch_num_subdiags = gko::batch_stride(num_systems, KL);
    auto batch_num_superdiags = gko::batch_stride(num_systems, KU);
    auto batch_vec_size = gko::batch_dim<>(num_systems, gko::dim<2>(nrows, 1));
    // @sect3{Use of application-allocated memory}
    // We can either work on the existing memory allocated in the application,
    //  or we can copy it for the linear solve.
    //  Note: it is not possible to use data through a const pointer directly.
    //  Because our pointers are not const, we can just 'wrap' the given
    //  pointers into Ginkgo Array views so that we can create a Ginkgo matrix
    //  out of them.
    // Ginkgo expects the nonzero values for all the small matrices to be
    //  allocated contiguously, one matrix after the other.

    auto band_arr_vals_view = gko::array<value_type>::const_view(
        exec, num_systems * nrows * (2 * KL + KU + 1),
        appl_sys.batch_band_array);

    auto A = gko::share(mtx_type::create_const(
        exec, batch_mat_size, batch_num_subdiags, batch_num_superdiags,
        std::move(band_arr_vals_view)));

    std::vector<gko::matrix_data<value_type>> data;
    A->write(data);
    auto A_csr = gko::share(batch_csr::create(exec));
    A_csr->read(data);

    // @sect3{RHS and solution vectors}
    // batch_stride object specifies the access stride within the individual
    //  matrices (vectors) in the batch. In this case, we specify a stride of 1
    //  as the common value for all the matrices.
    auto batch_vec_stride = gko::batch_stride(num_systems, 1);
    // Create RHS, again reusing application allocation
    auto b_view = gko::array<value_type>::const_view(exec, num_systems * nrows,
                                                     appl_sys.all_rhs);
    auto b = vec_type::create_const(exec, batch_vec_size, std::move(b_view),
                                    batch_vec_stride);

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
    timings_file << num_systems << "  " << av_time_millisec << " \n";
    timings_file.close();

    return 0;
}

ApplSysData appl_generate_system(const int N, const int KL, const int KU,
                                 const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec)
{
    const int nrows_band_format = 2 * KL + KU + 1;
    const int ncols_band_format = N;
    std::ranlux48 rgen(15);
    std::normal_distribution<real_type> distb(0.5, 0.1);

    std::vector<value_type> batch_band_array(
        nrows_band_format * ncols_band_format * nsystems,
        gko::zero<value_type>());

    for (size_type isys = 0; isys < nsystems; isys++) {
        const auto batch_entry_offset =
            isys * ncols_band_format * nrows_band_format;

        for (int col_dense_layout = 0; col_dense_layout < N;
             col_dense_layout++) {
            const int row_dense_layout_start =
                std::max(int{0}, col_dense_layout - KU);
            const int row_dense_layout_end_inclusive =
                std::min(N - 1, col_dense_layout + KL);

            for (int row_dense_layout = row_dense_layout_start;
                 row_dense_layout <= row_dense_layout_end_inclusive;
                 row_dense_layout++) {
                const int row_band_layout =
                    KL + KU + row_dense_layout - col_dense_layout;
                const int col_band_layout = col_dense_layout;
                const value_type val = distb(rgen);
                batch_band_array[batch_entry_offset + row_band_layout +
                                 col_band_layout * nrows_band_format] = val;
            }
        }
    }

    std::vector<value_type> allb(N * nsystems);
    for (size_type isys = 0; isys < nsystems; isys++) {
        const value_type bval = distb(rgen);
        std::fill(allb.begin() + isys * N, allb.begin() + (isys + 1) * N, bval);
    }

    value_type* const batch_band_arr = exec->alloc<value_type>(
        nsystems * nrows_band_format * ncols_band_format);
    exec->copy_from(exec->get_master().get(),
                    nsystems * nrows_band_format * ncols_band_format,
                    batch_band_array.data(), batch_band_arr);

    value_type* const all_b = exec->alloc<value_type>(nsystems * N);
    exec->copy_from(exec->get_master().get(), nsystems * N, allb.data(), all_b);

    return {nsystems, N, KL, KU, batch_band_arr, all_b};
}

void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec)
{
    // In general, the application would control non-const pointers;
    //  the const casts below would not be needed.
    exec->free(const_cast<value_type*>(appl_data.batch_band_array));
    exec->free(const_cast<value_type*>(appl_data.all_rhs));
}
