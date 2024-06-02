// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


// @sect3{Type aliases for convenience}
// Use some shortcuts.
using value_type = double;
using real_type = gko::remove_complex<value_type>;
using index_type = int;
using size_type = gko::size_type;
using vec_type = gko::batch::MultiVector<value_type>;
using real_vec_type = gko::batch::MultiVector<real_type>;
using mtx_type = gko::batch::matrix::Csr<value_type, index_type>;
using bicgstab = gko::batch::solver::Bicgstab<value_type>;


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


// @sect3{'Application' structures and functions}
// Structure to simulate application data related to the linear systems
// to be solved.
//
// We use raw pointers below to demonstrate how to handle the situation when
// the application only gives us raw pointers. Ideally, one should use
// Ginkgo's gko::Array class here. In this example, we assume that the data is
// in a format that can directly be given to a batch::matrix::Csr object.
struct ApplSysData {
    // Number of small systems in the batch.
    size_type nsystems;
    // Number of rows in each system.
    int nrows;
    // Number of non-zeros in each system matrix.
    int nnz;
    // Row pointers for one matrix
    const index_type* row_ptrs;
    // Column indices of non-zeros for one matrix
    const index_type* col_idxs;
    // Nonzero values for all matrices in the batch, concatenated
    const value_type* all_values;
    // RHS vectors for all systems in the batch, concatenated
    const value_type* all_rhs;
};


/*
 * Generates a batch of tridiagonal systems.
 *
 * @param nrows  Number of rows in each system.
 * @param nsystems  Number of systems in the batch.
 * @param exec  The device executor to use for the solver.
 * @note  Normally, the application may not deal with Ginkgo executors, nor do
 * we need it to. Here, we use the executor for backend-independent device
 * memory allocation. The application, for example, might assume Hip (for AMD
 * GPUs) and use `hipMalloc` directly.
 */
ApplSysData appl_generate_system(const int nrows, const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec);

// Deallocate application data.
void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec);


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

    const size_type num_systems = argc >= 3 ? std::atoi(argv[2]) : 2;
    const int num_rows = argc >= 4 ? std::atoi(argv[3]) : 32;  // per system
    // Whether to print the residuals or not.
    const bool print_residuals =
        argc >= 5 ? (std::string(argv[4]) == "true") : false;
    // The number of repetitions for the timing.
    const int num_reps = argc >= 6 ? std::atoi(argv[5]) : 20;
    // @sect3{Generate data}
    // The "application" generates the batch of linear systems on the device
    auto appl_sys = appl_generate_system(num_rows, num_systems, exec);
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, num_rows));
    auto batch_vec_size =
        gko::batch_dim<2>(num_systems, gko::dim<2>(num_rows, 1));
    // @sect3{Use of application-allocated memory}
    // We can either work on the existing memory allocated in the application,
    // or we can copy it for the linear solve.
    // Ginkgo expects the nonzero values for all the small matrices to be
    // allocated contiguously, one matrix after the other.
    auto vals_view = gko::array<value_type>::const_view(
        exec, num_systems * appl_sys.nnz, appl_sys.all_values);
    auto rowptrs_view = gko::array<index_type>::const_view(exec, num_rows + 1,
                                                           appl_sys.row_ptrs);
    auto colidxs_view = gko::array<index_type>::const_view(exec, appl_sys.nnz,
                                                           appl_sys.col_idxs);
    auto A = gko::share(mtx_type::create_const(
        exec, batch_mat_size, std::move(vals_view), std::move(colidxs_view),
        std::move(rowptrs_view)));
    // @sect3{RHS and solution vectors}
    // batch_stride object specifies the access stride within the individual
    //  matrices (vectors) in the batch. In this case, we specify a stride of 1
    //  as the common value for all the matrices.
    // Create RHS, again reusing application allocation
    auto b_view = gko::array<value_type>::const_view(
        exec, num_systems * num_rows, appl_sys.all_rhs);
    auto b = vec_type::create_const(exec, batch_vec_size, std::move(b_view));
    // Create initial guess as 0 and copy to device
    auto x = vec_type::create(exec);
    auto host_x = vec_type::create(exec->get_master(), batch_vec_size);
    for (size_type isys = 0; isys < num_systems; isys++) {
        for (int irow = 0; irow < num_rows; irow++) {
            host_x->at(isys, irow, 0) = gko::zero<value_type>();
        }
    }
    x->copy_from(host_x.get());

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-10};
    // Create a batched solver factory with relevant parameters.
    auto solver =
        bicgstab::build()
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
    // Solve the batch system
    auto x_clone = gko::clone(x);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        x_clone->copy_from(x.get());
        solver->apply(b, x_clone);
    }

    double apply_time = 0.0;
    for (int i = 0; i < num_reps; ++i) {
        x_clone->copy_from(x.get());
        exec->synchronize();
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
        solver->apply(b, x_clone);
        exec->synchronize();
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
    A->apply(one, x, neg_one, res);
    // allocate and compute residual norm
    auto host_res_norm = real_vec_type::create(exec->get_master(), norm_dim);
    host_res_norm->fill(0.0);
    res->compute_norm2(host_res_norm);
    auto host_log_resid = gko::make_temporary_clone(
        exec->get_master(), &logger->get_residual_norm());
    auto host_log_iters = gko::make_temporary_clone(
        exec->get_master(), &logger->get_num_iterations());

    if (print_residuals) {
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
                      << ", iterations = "
                      << host_log_iters->get_const_data()[i] << std::endl;
            const real_type relresnorm =
                unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
            if (!(relresnorm <= reduction_factor)) {
                std::cout << "System " << i << " converged only to "
                          << relresnorm << " relative residual." << std::endl;
            }
        }
    }
    std::cout << "Solver type: "
              << "batch::bicgstab"
              << "\nMatrix size: " << A->get_common_size()
              << "\nNum batch entries: " << A->get_num_batch_items()
              << "\nEntire solve took: " << apply_time / num_reps << " seconds."
              << std::endl;

    // Ginkgo objects are cleaned up automatically; but the "application" still
    //  needs to clean up its data in this case.
    appl_clean_up(appl_sys, exec);
    return 0;
}


// Generate the matrix and the vectors. This function emulates the generation of
// the data from the application.
ApplSysData appl_generate_system(const int nrows, const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec)
{
    const int nnz = nrows * 3 - 2;
    std::default_random_engine rgen(15);
    std::normal_distribution<real_type> distb(0.5, 0.1);
    std::vector<real_type> spacings(nsystems * nrows);
    std::generate(spacings.begin(), spacings.end(),
                  [&]() { return distb(rgen); });

    std::vector<value_type> allvalues(nnz * nsystems);
    for (size_type isys = 0; isys < nsystems; isys++) {
        allvalues.at(isys * nnz) = 2.0 / spacings.at(isys * nrows);
        allvalues.at(isys * nnz + 1) = -1.0;
        for (int irow = 0; irow < nrows - 2; irow++) {
            allvalues.at(isys * nnz + 2 + irow * 3) = -1.0;
            allvalues.at(isys * nnz + 2 + irow * 3 + 1) =
                2.0 / spacings.at(isys * nrows + irow + 1);
            allvalues.at(isys * nnz + 2 + irow * 3 + 2) = -1.0;
        }
        allvalues.at(isys * nnz + 2 + (nrows - 2) * 3) = -1.0;
        allvalues.at(isys * nnz + 2 + (nrows - 2) * 3 + 1) =
            2.0 / spacings.at((isys + 1) * nrows - 1);
        assert(isys * nnz + 2 + (nrows - 2) * 3 + 2 == (isys + 1) * nnz);
    }

    std::vector<index_type> rowptrs(nrows + 1);
    rowptrs.at(0) = 0;
    rowptrs.at(1) = 2;
    for (int i = 2; i < nrows; i++) {
        rowptrs.at(i) = rowptrs.at(i - 1) + 3;
    }
    rowptrs.at(nrows) = rowptrs.at(nrows - 1) + 2;
    assert(rowptrs.at(nrows) == nnz);

    std::vector<index_type> colidxs(nnz);
    colidxs.at(0) = 0;
    colidxs.at(1) = 1;
    const int nnz_per_row = 3;
    for (int irow = 1; irow < nrows - 1; irow++) {
        colidxs.at(2 + (irow - 1) * nnz_per_row) = irow - 1;
        colidxs.at(2 + (irow - 1) * nnz_per_row + 1) = irow;
        colidxs.at(2 + (irow - 1) * nnz_per_row + 2) = irow + 1;
    }
    colidxs.at(2 + (nrows - 2) * nnz_per_row) = nrows - 2;
    colidxs.at(2 + (nrows - 2) * nnz_per_row + 1) = nrows - 1;
    assert(2 + (nrows - 2) * nnz_per_row + 1 == nnz - 1);

    std::vector<value_type> allb(nrows * nsystems);
    for (size_type isys = 0; isys < nsystems; isys++) {
        const value_type bval = distb(rgen);
        std::fill(allb.begin() + isys * nrows,
                  allb.begin() + (isys + 1) * nrows, bval);
    }

    index_type* const row_ptrs = exec->alloc<index_type>(nrows + 1);
    exec->copy_from(exec->get_master().get(), static_cast<size_type>(nrows + 1),
                    rowptrs.data(), row_ptrs);
    index_type* const col_idxs = exec->alloc<index_type>(nnz);
    exec->copy_from(exec->get_master().get(), static_cast<size_type>(nnz),
                    colidxs.data(), col_idxs);
    value_type* const all_values = exec->alloc<value_type>(nsystems * nnz);
    exec->copy_from(exec->get_master().get(), nsystems * nnz, allvalues.data(),
                    all_values);
    value_type* const all_b = exec->alloc<value_type>(nsystems * nrows);
    exec->copy_from(exec->get_master().get(), nsystems * nrows, allb.data(),
                    all_b);
    return {nsystems, nrows, nnz, row_ptrs, col_idxs, all_values, all_b};
}

void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec)
{
    // In general, the application would control non-const pointers;
    //  the const casts below would not be needed.
    exec->free(const_cast<index_type*>(appl_data.row_ptrs));
    exec->free(const_cast<index_type*>(appl_data.col_idxs));
    exec->free(const_cast<value_type*>(appl_data.all_values));
    exec->free(const_cast<value_type*>(appl_data.all_rhs));
}
