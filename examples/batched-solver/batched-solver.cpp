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

#include <fstream>
#include <iostream>
#include <map>
#include <random>
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
using solver_type = gko::solver::BatchBicgstab<value_type>;


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


// Generates a batch of tridiagonal systems.
//
// @param nrows  Number of rows in each system.
// @param nsystems  Number of systems in the batch.
// @param exec  The device executor to use for the solver.
//   Normally, the application may not deal with Ginkgo executors, nor do we
//   need it to. Here, we use the executor for backend-independent device
//   memory allocation. The application, for example, might assume Hip
//   (for AMD GPUs) and use `hipMalloc` directly.
ApplSysData appl_generate_system(const int nrows, const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec);

// Deallocate application data.
void appl_clean_up(ApplSysData& appl_data, std::shared_ptr<gko::Executor> exec);


int main(int argc, char* argv[])
{
    // Print the ginkgo version information.
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

    const size_type num_systems = argc >= 3 ? std::atoi(argv[2]) : 2;
    const int num_rows = 35;  // per system
    // @sect3{Generate data}
    // The "application" generates the batch of linear systems on the device
    auto appl_sys = appl_generate_system(num_rows, num_systems, exec);
    // Create batch_dim object to describe the dimensions of the batch matrix.
    auto batch_mat_size =
        gko::batch_dim<>(num_systems, gko::dim<2>(num_rows, num_rows));
    auto batch_vec_size =
        gko::batch_dim<>(num_systems, gko::dim<2>(num_rows, 1));
    // @sect3{Use of application-allocated memory}
    // We can either work on the existing memory allocated in the application,
    //  or we can copy it for the linear solve.
    //  Note: it is not possible to use data through a const pointer directly.
    //  Because our pointers are not const, we can just 'wrap' the given
    //  pointers into Ginkgo Array views so that we can create a Ginkgo matrix
    //  out of them.
    // Ginkgo expects the nonzero values for all the small matrices to be
    //  allocated contiguously, one matrix after the other.
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
    auto batch_vec_stride = gko::batch_stride(num_systems, 1);
    // Create RHS, again reusing application allocation
    auto b_view = gko::array<value_type>::const_view(
        exec, num_systems * num_rows, appl_sys.all_rhs);
    auto b = vec_type::create_const(exec, batch_vec_size, std::move(b_view),
                                    batch_vec_stride);
    // Create initial guess as 0 and copy to device
    auto x = vec_type::create(exec);
    auto host_x =
        vec_type::create(exec->get_master(), batch_vec_size, batch_vec_stride);
    for (size_type isys = 0; isys < num_systems; isys++) {
        for (int irow = 0; irow < num_rows; irow++) {
            host_x->at(isys, irow, 0) = gko::zero<value_type>();
        }
    }
    x->copy_from(host_x.get());

    // @sect3{Create the batch solver factory}
    const real_type reduction_factor{1e-6};
    // Create a batched solver factory with relevant parameters.
    auto solver_gen =
        solver_type::build()
            .with_default_max_iterations(500)
            .with_default_residual_tol(reduction_factor)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build().on(exec))
            .on(exec);

    // @sect3{Batch logger}
    // Create a logger to obtain the iteration counts and "implicit" residual
    //  norms for every system after the solve.
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> logger =
        gko::log::BatchConvergence<value_type>::create(exec);

    // @sect3{Generate and solve}
    // Generate the batch solver from the batch matrix
    auto solver = solver_gen->generate(A);
    // add the logger to the solver
    solver->add_logger(logger);
    // Solve the batch system
    solver->apply(lend(b), lend(x));
    // This is not necessary, but one might want to remove the logger before
    //  the next solve using the same solver object.
    solver->remove_logger(logger.get());

    // @sect3{Check result}
    // Compute norm of RHS on the device and automatically copy to host
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

    std::cout << "Residual norm sqrt(r^T r):\n";
    // "unbatch" converts a batch object into a vector of objects of the
    //   corresponding single type, eg. BatchDense --> vector<Dense>.
    auto unb_res = res_norm->unbatch();
    auto unb_bnorm = b_norm->unbatch();
    for (size_type i = 0; i < num_systems; ++i) {
        std::cout << " System no. " << i
                  << ": residual norm = " << unb_res[i]->at(0, 0)
                  << ", internal residual norm = "
                  << logger->get_residual_norm()->at(i, 0, 0)
                  << ", iterations = "
                  << logger->get_num_iterations().get_const_data()[i]
                  << std::endl;
        const real_type relresnorm =
            unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
        if (!(relresnorm <= reduction_factor)) {
            std::cout << "System " << i << " converged only to " << relresnorm
                      << " relative residual." << std::endl;
        }
    }

    // Ginkgo objects are cleaned up automatically; but the "application" still
    //  needs to clean up its data in this case.
    appl_clean_up(appl_sys, exec);
    return 0;
}

ApplSysData appl_generate_system(const int nrows, const size_type nsystems,
                                 std::shared_ptr<gko::Executor> exec)
{
    const int nnz = nrows * 3 - 2;
    std::ranlux48 rgen(15);
    std::normal_distribution<real_type> distb(0.5, 0.1);
    std::vector<real_type> spacings(nsystems * nrows);
    std::generate(spacings.begin(), spacings.end(),
                  [&]() { return distb(rgen); });

    std::vector<value_type> allvalues(nnz * nsystems);
    for (size_type isys = 0; isys < nsystems; isys++) {
        allvalues[isys * nnz] = 2.0 / spacings[isys * nrows];
        allvalues[isys * nnz + 1] = -1.0;
        for (int irow = 0; irow < nrows - 2; irow++) {
            allvalues[isys * nnz + 2 + irow * 3] = -1.0;
            allvalues[isys * nnz + 2 + irow * 3 + 1] =
                2.0 / spacings[isys * nrows + irow + 1];
            allvalues[isys * nnz + 2 + irow * 3 + 2] = -1.0;
        }
        allvalues[isys * nnz + 2 + (nrows - 2) * 3] = -1.0;
        allvalues[isys * nnz + 2 + (nrows - 2) * 3 + 1] =
            2.0 / spacings[(isys + 1) * nrows - 1];
        assert(isys * nnz + 2 + (nrows - 2) * 3 + 2 == (isys + 1) * nnz);
    }

    std::vector<index_type> rowptrs(nrows + 1);
    rowptrs[0] = 0;
    rowptrs[1] = 2;
    for (int i = 2; i < nrows; i++) {
        rowptrs[i] = rowptrs[i - 1] + 3;
    }
    rowptrs[nrows] = rowptrs[nrows - 1] + 2;
    assert(rowptrs[nrows] == nnz);

    std::vector<index_type> colidxs(nnz);
    colidxs[0] = 0;
    colidxs[1] = 1;
    const int nnz_per_row = 3;
    for (int irow = 1; irow < nrows - 1; irow++) {
        colidxs[2 + (irow - 1) * nnz_per_row] = irow - 1;
        colidxs[2 + (irow - 1) * nnz_per_row + 1] = irow;
        colidxs[2 + (irow - 1) * nnz_per_row + 2] = irow + 1;
    }
    colidxs[2 + (nrows - 2) * nnz_per_row] = nrows - 2;
    colidxs[2 + (nrows - 2) * nnz_per_row + 1] = nrows - 1;
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
