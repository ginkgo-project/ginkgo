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


// @sect3{'Application' structures and functions}
// Structure to simulate application data related to the linear systems
// to be solved.
//
// We use raw pointers below to demonstrate how to handle the situation when
// the application only gives us raw pointers. Ideally, one should use
// Ginkgo's gko::Array class here.
struct ApplSysData {
    // Number of small systems in the batch.
    size_type num_systems;
    // Number of rows in each system.
    int num_rows;
    // Number of non-zeros in each system matrix.
    int nnz;
    const value_type* sup_diag;
    const value_type* main_diag;
    const value_type* sub_diag;
    const value_type* all_rhs;
};


// Generates a batch of tridiagonal systems.
//
// @param num_rows  Number of rows in each system.
// @param num_systems  Number of systems in the batch.
// @param exec  The device executor to use for the solver.
//   Normally, the application may not deal with Ginkgo executors, nor do we
//   need it to. Here, we use the executor for backend-independent device
//   memory allocation. The application, for example, might assume Hip
//   (for AMD GPUs) and use `hipMalloc` directly.
ApplSysData appl_generate_system(const int num_rows,
                                 const size_type num_systems,
                                 std::shared_ptr<gko::Executor> exec);

int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [dir_path] [problem_name] [num_systems] "
                     "[num_duplications] "
                     "[num_recursive_steps] [subwarp_size] [tridiag approach] "
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

    const size_type num_systems = argc >= 3 ? std::atoi(argv[2]) : 2;
    const int num_rows = argc >= 4 ? std::atoi(argv[3]) : 32;  // per system
    const std::string in_strat = argc >= 5 ? argv[4] : "recursive_app1";
    const bool print_time =
        argc >= 6 ? (std::string(argv[5]) == "time") : false;
    const bool print_residuals =
        argc >= 7 ? (std::string(argv[6]) == "residuals") : false;
    const int num_reps = argc >= 8 ? std::atoi(argv[7]) : 20;

    const int number_recursive_steps = argc >= 9 ? std::atoi(argv[8]) : 2;
    const int subwarp_size = argc >= 10 ? std::atoi(argv[9]) : 16;

    // Approach
    enum gko::solver::batch_tridiag_solve_approach approach;
    if (in_strat == std::string("recursive_app1")) {
        approach = gko::solver::batch_tridiag_solve_approach::recursive_app1;
    } else if (in_strat == std::string("recursive_app2")) {
        approach = gko::solver::batch_tridiag_solve_approach::recursive_app2;
    } else if (in_strat == std::string("vendor_provided")) {
        approach = gko::solver::batch_tridiag_solve_approach::vendor_provided;
    }

    // @sect3{Read batch from files}
    auto appl_sys = appl_generate_system(num_rows, num_systems, exec);
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<>(num_systems, gko::dim<2>(num_rows, num_rows));
    auto batch_vec_size =
        gko::batch_dim<>(num_systems, gko::dim<2>(num_rows, 1));

    auto sub_view = gko::array<value_type>::const_view(
        exec, num_systems * appl_sys.num_rows, appl_sys.sub_diag);
    auto main_view = gko::array<value_type>::const_view(
        exec, num_systems * appl_sys.num_rows, appl_sys.main_diag);
    auto sup_view = gko::array<value_type>::const_view(
        exec, num_systems * appl_sys.num_rows, appl_sys.sup_diag);
    auto A = gko::share(
        mtx_type::create_const(exec, batch_mat_size, std::move(sub_view),
                               std::move(main_view), std::move(sup_view)));

    // Create RHS
    auto batch_vec_stride = gko::batch_stride(num_systems, 1);
    // Create RHS, again reusing application allocation
    auto b_view = gko::array<value_type>::const_view(
        exec, num_systems * num_rows, appl_sys.all_rhs);
    auto b = vec_type::create_const(exec, batch_vec_size, std::move(b_view),
                                    batch_vec_stride);

    auto x = vec_type::create(exec, b->get_size());

    // @sect3{Create the batch solver factory}
    // Create a batched solver factory with relevant parameters.
    auto solver_gen = batch_tridiag_solver::build()
                          .with_batch_tridiagonal_solution_approach(approach)
                          .with_num_recursive_steps(number_recursive_steps)
                          .with_tile_size(subwarp_size)
                          .on(exec);

    // @sect3{Generate and solve}
    // Generate the batch solver from the batch matrix
    auto solver = solver_gen->generate(A);

    const int leave_first = 5;

    for (int i = 0; i < leave_first; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    exec->synchronize();
    solver->initialize_preprocess_time(0.0);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_reps; i++) {
        // Solve the batch system
        solver->apply(lend(b), lend(x));
    }

    exec->synchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    auto apply_time = duration.count();

    auto b_norm = gko::batch_initialize<real_vec_type>(num_systems, {0.0},
                                                       exec->get_master());
    b->compute_norm2(lend(b_norm));
    // we need constants on the device
    auto one = gko::batch_initialize<vec_type>(num_systems, {1.0}, exec);
    auto neg_one = gko::batch_initialize<vec_type>(num_systems, {-1.0}, exec);
    // allocate and compute the residual
    auto res = vec_type::create(exec, batch_vec_size);
    res->copy_from(lend(b));

    A->apply(lend(one), lend(x), lend(neg_one), lend(res));
    // allocate and compute residual norm
    auto res_norm = gko::batch_initialize<real_vec_type>(num_systems, {0.0},
                                                         exec->get_master());
    res->compute_norm2(lend(res_norm));
    if (print_residuals) {
        std::cout << "Residual norm sqrt(r^T r):\n";
        // "unbatch" converts a batch object into a vector of objects of the
        //   corresponding single type, eg. BatchDense --> vector<Dense>.
        auto unb_res = res_norm->unbatch();
        auto unb_bnorm = b_norm->unbatch();
        for (size_type i = 0; i < num_systems; ++i) {
            std::cout << " System no. " << i
                      << ": residual norm = " << unb_res[i]->at(0, 0)
                      << std::endl;
            const real_type relresnorm =
                unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
            if (!(relresnorm <= 1e-6)) {
                std::cout << "System " << i << " converged only to "
                          << relresnorm << " relative residual." << std::endl;
            }
        }
    }
    if (print_time) {
        std::cout << apply_time / num_reps << std::endl;
    } else {
        std::cout << "Solver type: " << in_strat
                  << "\nMatrix size: " << A->get_size().at(0)
                  << "\nNum batch entries: " << A->get_num_batch_entries()
                  << "\nEntire solve took: " << apply_time / num_reps
                  << " seconds." << std::endl;
        if (approach ==
            gko::solver::batch_tridiag_solve_approach::vendor_provided) {
            apply_time -= (solver->get_preprocess_time() * 1000);
            std::cout << "Preprocess time"
                      << solver->get_preprocess_time() * 1000
                      << "secs\nThe solve time (without pre-processing)"
                      << apply_time / num_reps << " secs " << std::endl;
        }
    }

    return 0;
}


ApplSysData appl_generate_system(const int num_rows,
                                 const size_type num_systems,
                                 std::shared_ptr<gko::Executor> exec)
{
    int nnz = num_rows * 3;
    std::ranlux48 rgen(15);
    std::normal_distribution<real_type> distb(0.5, 0.1);
    std::vector<value_type> sup_diag_values(num_rows * num_systems);
    std::vector<value_type> main_diag_values(num_rows * num_systems);
    std::vector<value_type> sub_diag_values(num_rows * num_systems);
    for (size_type sys = 0; sys < num_systems; ++sys) {
        for (size_type row = 0; row < num_rows; ++row) {
            if (row == 0) {
                sup_diag_values[sys * num_rows + row] = value_type{-2};
                main_diag_values[sys * num_rows + row] = value_type{4};
                sub_diag_values[sys * num_rows + row] = value_type{0};
            } else if (row == num_rows - 1) {
                sup_diag_values[sys * num_rows + row] = value_type{0};
                main_diag_values[sys * num_rows + row] = value_type{4};
                sub_diag_values[sys * num_rows + row] = value_type{-1};
            } else {
                sup_diag_values[sys * num_rows + row] = value_type{-2};
                main_diag_values[sys * num_rows + row] = value_type{4};
                sub_diag_values[sys * num_rows + row] = value_type{-1};
            }
        }
    }
    std::vector<value_type> allb(num_rows * num_systems);
    for (size_type isys = 0; isys < num_systems; isys++) {
        const value_type bval = distb(rgen);
        std::fill(allb.begin() + isys * num_rows,
                  allb.begin() + (isys + 1) * num_rows, bval);
    }

    value_type* const sup_diag =
        exec->alloc<value_type>(num_rows * num_systems);
    exec->copy_from(exec->get_master().get(),
                    static_cast<size_type>(num_rows * num_systems),
                    sup_diag_values.data(), sup_diag);
    value_type* const main_diag =
        exec->alloc<value_type>(num_rows * num_systems);
    exec->copy_from(exec->get_master().get(),
                    static_cast<size_type>(num_rows * num_systems),
                    main_diag_values.data(), main_diag);
    value_type* const sub_diag =
        exec->alloc<value_type>(num_rows * num_systems);
    exec->copy_from(exec->get_master().get(),
                    static_cast<size_type>(num_rows * num_systems),
                    sub_diag_values.data(), sub_diag);
    value_type* const all_b = exec->alloc<value_type>(num_systems * num_rows);
    exec->copy_from(exec->get_master().get(), num_systems * num_rows,
                    allb.data(), all_b);
    return {num_systems, num_rows, nnz, sup_diag, main_diag, sub_diag, all_b};
}

void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec)
{
    // In general, the application would control non-const pointers;
    //  the const casts below would not be needed.
    exec->free(const_cast<value_type*>(appl_data.sub_diag));
    exec->free(const_cast<value_type*>(appl_data.sup_diag));
    exec->free(const_cast<value_type*>(appl_data.main_diag));
    exec->free(const_cast<value_type*>(appl_data.all_rhs));
}
