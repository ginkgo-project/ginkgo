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
using mtx_type = gko::matrix::BatchCsr<value_type, index_type>;
// using mtx_type = gko::matrix::BatchEll<value_type, index_type>;
using bicgstab = gko::solver::BatchBicgstab<value_type>;
using cg = gko::solver::BatchCg<value_type>;
using gmres = gko::solver::BatchGmres<value_type>;
using richardson = gko::solver::BatchRichardson<value_type>;
using direct = gko::solver::BatchDirect<value_type>;


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
    if (!(std::string(argv[6]) == "time"))
        std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
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
    // Name of the problem, which is also the directory under which all
    //  matrices are stored.
    const std::string problem_descr_str = argc >= 3 ? argv[2] : "pores_1";
    // Number of linear systems to read from files.
    const size_type num_systems = argc >= 4 ? std::atoi(argv[3]) : 2;
    const size_type num_duplications = argc >= 5 ? std::atoi(argv[4]) : 2;
    const std::string in_solver = argc >= 6 ? argv[5] : "bicgstab";
    // Number of times to duplicate whatever systems are read from files.
    const bool print_time =
        argc >= 7 ? (std::string(argv[6]) == "time") : false;
    const bool print_residuals =
        argc >= 8 ? (std::string(argv[7]) == "residuals") : false;
    const int num_reps = argc >= 9 ? std::atoi(argv[8]) : 20;
    // Whether to enable diagonal scaling of the matrices before solving.
    //  The scaling vectors need to be available in a 'S.mtx' file.
    const std::string batch_scaling = argc >= 10 ? argv[9] : "none";
    const int gmres_restart = argc >= 11 ? std::atoi(argv[10]) : 10;
    auto data = std::vector<gko::matrix_data<value_type>>(num_systems);
    std::vector<gko::matrix_data<value_type>> bdata(num_systems);
    auto scale_data = std::vector<gko::matrix_data<value_type>>(num_systems);
    for (size_type i = 0; i < data.size(); ++i) {
        const std::string mat_str = "A.mtx";
        const std::string fbase =
            "data/" + problem_descr_str + "/" + std::to_string(i) + "/";
        std::string fname = fbase + mat_str;
        std::ifstream mtx_fd(fname);
        data[i] = gko::read_raw<value_type>(mtx_fd);
        // std::string b_str = "b.mtx";
        // std::string bfname =
        //     "data/" + problem_descr_str + "/" + std::to_string(i) + "/" +
        //     b_str;
        // std::ifstream b_fd(bfname);
        // bdata[i] = gko::read_raw<value_type>(b_fd);
        // If necessary, 'scaling vectors' can be provided to diagonal-scale
        //  a system from the left and the right. For this example, no
        //  scaling vectors are provided.
        if (batch_scaling == "explicit") {
            std::string scale_fname = fbase + "S.mtx";
            std::ifstream scale_fd(scale_fname);
            scale_data[i] = gko::read_raw<value_type>(scale_fd);
        }
    }
    auto single_batch = mtx_type::create(exec);
    single_batch->read(data);
    // We can duplicate the batch a few times if we wish.
    std::shared_ptr<mtx_type> A =
        mtx_type::create(exec, num_duplications, single_batch.get());
    // Create RHS

    const size_type num_total_systems = num_systems * num_duplications;
    const int num_rows = static_cast<int>(A->get_size().at(0)[0]);
    auto b = vec_type::create(exec);
    auto batch_vec_size =
        gko::batch_dim<>(num_total_systems, gko::dim<2>(num_rows, 1));
    auto batch_vec_stride = gko::batch_stride(num_total_systems, 1);
    auto host_b =
        vec_type::create(exec->get_master(), batch_vec_size, batch_vec_stride);
    for (size_type isys = 0; isys < num_total_systems; isys++) {
        for (int irow = 0; irow < num_rows; irow++) {
            host_b->at(isys, irow, 0) = gko::one<value_type>();
        }
    }
    b->copy_from(host_b.get());
    // Create initial guess as 0 and copy to device
    auto x = vec_type::create(exec);
    auto host_x = vec_type::create(exec->get_master(), b->get_size());
    // The number of rows in each system is taken as the 0th dimension of the
    //  size of the 0th system in the batch.
    for (size_type isys = 0; isys < num_total_systems; isys++) {
        for (int irow = 0; irow < num_rows; irow++) {
            host_x->at(isys, irow, 0) = gko::zero<value_type>();
        }
    }
    x->copy_from(host_x.get());

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-06};
    const int maxits = num_rows * 2;
    // Create a batched solver factory with relevant parameters.
    auto solver_gmres =
        gmres::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_restart(gmres_restart)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .on(exec);
    auto solver_cg =
        cg::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .on(exec);
    auto solver_bicgstab =
        bicgstab::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .on(exec);
    auto solver_richardson =
        richardson::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .on(exec);
    auto solver_direct = direct::build().on(exec);

    // @sect3{Batch logger}
    // Create a logger to obtain the iteration counts and "implicit" residual
    //  norms for every system after the solve.
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> logger =
        gko::log::BatchConvergence<value_type>::create(exec);

    // @sect3{Generate and solve}
    // Generate the batch solver from the batch matrix
    std::shared_ptr<gko::BatchLinOp> solver{};
    if (in_solver == "cg") {
        solver = solver_cg->generate(A);
    } else if (in_solver == "gmres") {
        solver = solver_gmres->generate(A);
    } else if (in_solver == "bicgstab") {
        solver = solver_bicgstab->generate(A);
    } else if (in_solver == "richardson") {
        solver = solver_richardson->generate(A);
    } else if (in_solver == "direct") {
        solver = solver_direct->generate(A);
    } else {
        throw "Not implemented";
    }

    // add the logger to the solver
    solver->add_logger(logger);
    auto x_clone = gko::clone(x);

    for (int i = 0; i < 3; ++i) {
        x_clone->copy_from(x.get());
        solver->apply(lend(b), lend(x_clone));
    }

    double apply_time = 0.0;
    for (int i = 0; i < num_reps; ++i) {
        x_clone->copy_from(x.get());
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
        solver->apply(lend(b), lend(x_clone));
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
        auto time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        apply_time += time_span.count();
    }
    x->copy_from(x_clone.get());
    // This is not necessary, but one might want to remove the logger before
    //  the next solve using the same solver object.
    solver->remove_logger(logger.get());

    // @sect3{Check result}
    // Compute norms of RHS and final residual to check the result
    auto b_norm = gko::batch_initialize<real_vec_type>(num_total_systems, {0.0},
                                                       exec->get_master());
    b->compute_norm2(lend(b_norm));
    // we need constants on the device
    auto one = gko::batch_initialize<vec_type>(num_total_systems, {1.0}, exec);
    auto neg_one =
        gko::batch_initialize<vec_type>(num_total_systems, {-1.0}, exec);
    // allocate and compute the residual
    auto res = vec_type::create(exec);
    res->copy_from(lend(b));
    A->apply(lend(one), lend(x), lend(neg_one), lend(res));
    // allocate and compute residual norm on the device
    auto res_norm = gko::batch_initialize<real_vec_type>(
        num_total_systems, {0.0}, exec->get_master());
    res->compute_norm2(lend(res_norm));

    if (print_residuals) {
        std::cout << "Residual norm sqrt(r^T r):\n";
        // "unbatch" converts a batch object into a vector of objects of the
        //   corresponding single type, eg. BatchDense --> vector<Dense>.
        auto unb_res = res_norm->unbatch();
        auto unb_bnorm = b_norm->unbatch();
        for (size_type i = 0; i < num_systems; ++i) {
            std::cout
                << " System no. " << i
                << ": residual norm = " << unb_res[i]->at(0, 0)
                << ", internal residual norm = "
                << ((in_solver == "direct")
                        ? 0
                        : logger->get_residual_norm()->at(i, 0, 0))
                << ", iterations = "
                << ((in_solver == "direct")
                        ? 0.0
                        : logger->get_num_iterations().get_const_data()[i])
                << std::endl;
            const real_type relresnorm =
                unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
            if (!(relresnorm <= reduction_factor)) {
                std::cout << "System " << i << " converged only to "
                          << relresnorm << " relative residual." << std::endl;
            }
        }
    }
    if (print_time) {
        std::cout << A->get_size().at(0)[0] << ","
                  << A->get_num_stored_elements() / num_total_systems << ","
                  << apply_time / num_reps << ","
                  << ((in_solver == "direct")
                          ? 0
                          : logger->get_num_iterations().get_const_data()[0])
                  << ","
                  << ((in_solver == "direct")
                          ? 0.0
                          : logger->get_residual_norm()->at(0, 0, 0))
                  << std::endl;
    } else {
        std::cout << "Solver type: " << in_solver
                  << "\nMatrix size: " << A->get_size().at(0)
                  << "\nNum batch entries: " << A->get_num_batch_entries()
                  << "\nEntire solve took: " << apply_time / num_reps
                  << " seconds." << std::endl;
    }

    return 0;
}
