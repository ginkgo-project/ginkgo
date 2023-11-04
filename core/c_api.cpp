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

#include <cstring>  // std::strcpy
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "../include/ginkgo/c_api.h"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/version.hpp>

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include <ginkgo/core/solver/cg.hpp>

/* ----------------------------------------------------------------------
 * C memory management
 * ---------------------------------------------------------------------- */
void c_char_ptr_free(char* ptr) { delete[] ptr; }


/* ----------------------------------------------------------------------
 * Library functions for other types in GINKGO
 * ---------------------------------------------------------------------- */

gko_dim2_st ginkgo_dim2_create(size_t rows, size_t cols)
{
    return gko_dim2_st{rows, cols};
}

size_t ginkgo_dim2_rows_get(gko_dim2_st dim) { return dim.rows; }

size_t ginkgo_dim2_cols_get(gko_dim2_st dim) { return dim.cols; }

/* ----------------------------------------------------------------------
 * Library functions for creating executors in GINKGO
 * ---------------------------------------------------------------------- */

struct gko_executor_st {
    std::shared_ptr<gko::Executor> shared_ptr;
};

void ginkgo_executor_delete(gko_executor exec_st_ptr) { delete exec_st_ptr; }

//-----------------------------------------------------------------------

gko_executor ginkgo_executor_omp_create()
{
    return new gko_executor_st{gko::OmpExecutor::create()};
}

// gko_executor ginkgo_executor_cuda_create()
// {
//     return new gko_executor_st{gko::CudaExecutor::create()};
// }

// gko_executor ginkgo_executor_hip_create()
// {
//     return new gko_executor_st{gko::HipExecutor::create()};
// }

// gko_executor ginkgo_executor_dpcpp_create()
// {
//     return new gko_executor_st{gko::DpcppExecutor::create()};
// }

gko_executor ginkgo_executor_reference_create()
{
    return new gko_executor_st{gko::ReferenceExecutor::create()};
}


/* ----------------------------------------------------------------------
 * Library functions for creating arrays and array operations in GINKGO
 * ---------------------------------------------------------------------- */
DEFINE_ARRAY_OVERLOAD(short, short, i16);
DEFINE_ARRAY_OVERLOAD(int, int, i32);
DEFINE_ARRAY_OVERLOAD(long long, long long, i64);
DEFINE_ARRAY_OVERLOAD(float, float, f32);
DEFINE_ARRAY_OVERLOAD(double, double, f64);
// DEFINE_ARRAY_OVERLOAD(float _Complex, std::complex<float>, cf32);
// DEFINE_ARRAY_OVERLOAD(double _Complex, std::complex<double>, cf64);


/* ----------------------------------------------------------------------
 * Library functions for creating matrices and matrix operations in GINKGO
 * ---------------------------------------------------------------------- */
DEFINE_DENSE_OVERLOAD(float, float, f32);
DEFINE_DENSE_OVERLOAD(double, double, f64);
// DEFINE_DENSE_OVERLOAD(float _Complex, std::complex<float>, cf32);
// DEFINE_DENSE_OVERLOAD(double _Complex, std::complex<double>, cf64);
// DEFINE_DENSE_OVERLOAD(short, short, i16);
// DEFINE_DENSE_OVERLOAD(int, int, i32);
// DEFINE_DENSE_OVERLOAD(long long, long long, i64);


struct gko_matrix_csr_f32_i32_st {
    std::shared_ptr<gko::matrix::Csr<float, int>> mat;
};

gko_matrix_csr_f32_i32 ginkgo_matrix_csr_f32_i32_create(gko_executor exec,
                                                        gko_dim2_st size,
                                                        size_t nnz)
{
    return new gko_matrix_csr_f32_i32_st{gko::matrix::Csr<float, int>::create(
        (*exec).shared_ptr, gko::dim<2>{size.rows, size.cols}, nnz)};
}

void ginkgo_matrix_csr_f32_i32_delete(gko_matrix_csr_f32_i32 mat_st_ptr)
{
    delete mat_st_ptr;
}

gko_matrix_csr_f32_i32 ginkgo_matrix_csr_f32_i32_read(const char* str_ptr,
                                                      gko_executor exec)
{
    std::string filename(str_ptr);
    std::ifstream ifs(filename, std::ifstream::in);

    return new gko_matrix_csr_f32_i32_st{
        gko::read<gko::matrix::Csr<float, int>>(std::move(ifs),
                                                (*exec).shared_ptr)};
}

size_t ginkgo_matrix_csr_f32_i32_get_num_stored_elements(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_num_stored_elements();
}

size_t ginkgo_matrix_csr_f32_i32_get_num_srow_elements(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_num_srow_elements();
}


gko_dim2_st ginkgo_matrix_csr_f32_i32_get_size(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    auto dim = (*mat_st_ptr).mat->get_size();
    return gko_dim2_st{dim[0], dim[1]};
}

// TODO: check how to do this!
const float* ginkgo_matrix_csr_f32_i32_get_const_values(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_const_values();
}

const int* ginkgo_matrix_csr_f32_i32_get_const_col_idxs(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_const_col_idxs();
}

const int* ginkgo_matrix_csr_f32_i32_get_const_row_ptrs(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_const_row_ptrs();
}

const int* ginkgo_matrix_csr_f32_i32_get_const_srow(
    gko_matrix_csr_f32_i32 mat_st_ptr)
{
    return (*mat_st_ptr).mat->get_const_srow();
}

// float ginkgo_matrix_csr_f32_i64_at(gko_matrix_csr_f32_i64 mat_st_ptr, size_t
// row, size_t col)
// {
//     return (*mat_st_ptr).mat->at(row, col);
// }


void ginkgo_matrix_csr_f32_i32_apply(gko_matrix_csr_f32_i32 mat_st_ptr,
                                     gko_matrix_dense_f32 alpha,
                                     gko_matrix_dense_f32 x,
                                     gko_matrix_dense_f32 beta,
                                     gko_matrix_dense_f32 y)
{
    mat_st_ptr->mat->apply(alpha->mat, x->mat, beta->mat, y->mat);
}

/* ----------------------------------------------------------------------
 * Library functions for BLAS linop in GINKGO
 * ---------------------------------------------------------------------- */
// function spmm!(A::SparseMatrixCsr{Tv, Ti}, α::Dense{Tv}, x::Dense{Tv},
// β::Dense{Tv}, y::Dense{Tv}) where {Tv, Ti}

// end


/* ----------------------------------------------------------------------
 * Library functions for iterative solvers in GINKGO
 * ---------------------------------------------------------------------- */
void ginkgo_solver_cg_solve(gko_executor exec_st_ptr,
                            gko_matrix_csr_f32_i32 A_st_ptr,
                            gko_matrix_dense_f32 b_st_ptr,
                            gko_matrix_dense_f32 x_st_ptr, int maxiter,
                            double reduction)
{
    auto solver_gen =
        gko::solver::Cg<float>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(maxiter),
                gko::stop::ResidualNorm<float>::build().with_reduction_factor(
                    reduction))
            .on(exec_st_ptr->shared_ptr);

    auto solver = solver_gen->generate(A_st_ptr->mat);

    solver->apply(b_st_ptr->mat, x_st_ptr->mat);
}

/* ----------------------------------------------------------------------
 * Library functions for retrieving configuration information in GINKGO
 * ---------------------------------------------------------------------- */

void ginkgo_version_get()
{
    std::cout << gko::version_info::get() << std::endl;
}