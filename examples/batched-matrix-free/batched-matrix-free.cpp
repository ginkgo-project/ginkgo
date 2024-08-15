// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include "apply.hpp"


// @sect3{Type aliases for convenience}
// Use some shortcuts.
using value_type = double;
using real_type = gko::remove_complex<value_type>;
using index_type = int;
using size_type = gko::size_type;
using vec_type = gko::batch::MultiVector<value_type>;
using real_vec_type = gko::batch::MultiVector<real_type>;
using mtx_type = gko::batch::matrix::Csr<value_type, index_type>;
using ext_type = gko::batch::matrix::External<value_type>;
using cg = gko::batch::solver::Cg<value_type>;


// @sect3{Convenience functions}
// Unbatch batch items into distinct Ginkgo types
namespace detail {


template <typename InputType>
auto unbatch(const InputType* batch_object)
{
    auto unbatched_mats =
        std::vector<std::unique_ptr<typename InputType::unbatch_type>>{};
    for (size_type b = 0; b < batch_object->get_num_batch_items(); ++b) {
        unbatched_mats.emplace_back(
            batch_object->create_const_view_for_item(b)->clone());
    }
    return unbatched_mats;
}


}  // namespace detail


int main(int argc, char* argv[])
{
    // Print ginkgo version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [num_systems] [num_rows] [print_residuals] "
                     "[num_reps]"
                  << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for an gko::OmpExecutor, which uses OpenMP
    // multi-threading in most of its kernels,
    // a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor, gko::CudaExecutor, gko::HipExecutor,
    // gko::DpcppExecutor which runs the code on a NVIDIA, AMD and Intel GPUs,
    // respectively.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    size_type num_systems = argc >= 3 ? std::atoi(argv[2]) : 2;
    const int num_rows = argc >= 4 ? std::atoi(argv[3]) : 32;  // per system
    const int num_reps = 1;

    if (num_rows < 2) {
        std::cerr << "Expected a <num_rows> to be >= 2." << std::endl;
        std::exit(-1);
    }

    // @sect3{Generate data}
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, num_rows));
    auto batch_vec_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, 1));
    auto nnz = 3 * (num_rows - 2) + 4;

    gko::array<gko::size_type> payload(exec, {num_systems});
    auto A = gko::share(
        ext_type::create(exec, batch_mat_size,
                         {.cpu_apply = simple_apply_generic<value_type>,
                          .hip_apply = get_hip_simple_apply_ptr()},
                         {.cpu_apply = advanced_apply_generic<value_type>,
                          .hip_apply = get_hip_advanced_apply_ptr()},
                         payload.get_data()));

    auto A_mtx = gko::share(mtx_type::create(exec, batch_mat_size, nnz));
    auto create_batch = [batch_mat_size, num_rows, nnz](gko::size_type id) {
        gko::matrix_data<value_type, index_type> md(
            batch_mat_size.get_common_size());
        md.nonzeros.reserve(nnz);
        for (index_type i = 0; i < num_rows; ++i) {
            if (i > 0) {
                md.nonzeros.emplace_back(i, i - 1, -1);
            }
            md.nonzeros.emplace_back(
                i, i,
                2 + value_type(id) / batch_mat_size.get_num_batch_items());
            if (i < num_rows - 1) {
                md.nonzeros.emplace_back(i, i + 1, -1);
            }
        }
        return md;
    };
    for (gko::size_type id = 0; id < batch_mat_size.get_num_batch_items();
         ++id) {
        A_mtx->create_view_for_item(id)->read(create_batch(id));
    }

    // @sect3{RHS and solution vectors}
    auto b = vec_type::create(exec, batch_vec_size);
    b->fill(1.0);
    // Create initial guess as 0 and copy to device
    auto x = vec_type::create(exec, batch_vec_size);
    x->fill(0.0);

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-10};
    // Create a batched solver factory with relevant parameters.
    auto solver =
        cg::build()
            .with_max_iterations(500)
            .with_tolerance(reduction_factor)
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(exec)
            ->generate(A);

    // @sect3{Batch logger}
    // Create a logger to obtain the iteration counts and "implicit" residual
    //  norms for every system after the solve.
    std::shared_ptr<const gko::batch::log::BatchConvergence<value_type>>
        logger = gko::batch::log::BatchConvergence<value_type>::create();

    // @sect3{Generate and solve}
    // add the logger to the solver
    solver->add_logger(logger);

    solver->apply(b, x);

    // @sect3{Check result}
    // Compute norm of RHS on the device and automatically copy to host
    auto norm_dim = gko::batch_dim<2>(num_systems, gko::dim<2>(1, 1));
    auto host_b_norm = real_vec_type::create(exec->get_master(), norm_dim);
    host_b_norm->fill(0.0);

    b->compute_norm2(host_b_norm);
    // we need constants on the device
    auto one = vec_type::create(exec, norm_dim);
    one->fill(1.0);
    auto neg_one = vec_type::create(exec, norm_dim);
    neg_one->fill(-1.0);
    // allocate and compute the residual
    auto res = vec_type::create(exec, batch_vec_size);
    res->copy_from(b);
    // @todo: change this when the external apply becomes available
    A_mtx->apply(one, x, neg_one, res);
    // allocate and compute residual norm
    auto host_res_norm = real_vec_type::create(exec->get_master(), norm_dim);
    host_res_norm->fill(0.0);
    res->compute_norm2(host_res_norm);
    auto host_log_resid = gko::make_temporary_clone(
        exec->get_master(), &logger->get_residual_norm());
    auto host_log_iters = gko::make_temporary_clone(
        exec->get_master(), &logger->get_num_iterations());

    std::cout << "Solver type: "
              << "batch::bicgstab"
              << "\nMatrix size: " << A->get_common_size()
              << "\nNum batch entries: " << A->get_num_batch_items()
              << std::endl;

    std::cout << "Residual norm sqrt(r^T r):\n";
    // "unbatch" converts a batch object into a vector of objects of the
    // corresponding single type, eg. batch::matrix::Dense -->
    // std::vector<Dense>.
    auto unb_res = detail::unbatch(host_res_norm.get());
    auto unb_bnorm = detail::unbatch(host_b_norm.get());
    for (size_type i = 0; i < num_systems; ++i) {
        std::cout << " System no. " << i
                  << ": residual norm = " << unb_res[i]->at(0, 0)
                  << ", implicit residual norm = "
                  << host_log_resid->get_const_data()[i]
                  << ", iterations = " << host_log_iters->get_const_data()[i]
                  << std::endl;
        const real_type relresnorm =
            unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
        if (!(relresnorm <= reduction_factor)) {
            std::cout << "System " << i << " converged only to " << relresnorm
                      << " relative residual." << std::endl;
        }
    }

    return 0;
}
