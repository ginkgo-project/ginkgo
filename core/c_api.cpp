// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Copyright (c) 2017-2023, the Ginkgo authors
#include <../include/ginkgo/c_api.h>
#include <cstring>  // std::strcpy
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/version.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

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

// TODO: currently use default device id 0
gko_executor ginkgo_executor_cuda_create()
{
    return new gko_executor_st{
        gko::CudaExecutor::create(0, gko::OmpExecutor::create())};
}

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
DEFINE_ARRAY_OVERLOAD(short, short, i16)
DEFINE_ARRAY_OVERLOAD(int, int, i32)
DEFINE_ARRAY_OVERLOAD(__int64_t, std::int64_t, i64)
DEFINE_ARRAY_OVERLOAD(float, float, f32)
DEFINE_ARRAY_OVERLOAD(double, double, f64)
// DEFINE_ARRAY_OVERLOAD(float _Complex, std::complex<float>, cf32)
// DEFINE_ARRAY_OVERLOAD(double _Complex, std::complex<double>, cf64)


/* ----------------------------------------------------------------------
 * Library functions for creating matrices and matrix operations in GINKGO
 * ---------------------------------------------------------------------- */
DEFINE_DENSE_OVERLOAD(float, float, f32)
DEFINE_DENSE_OVERLOAD(double, double, f64)
// DEFINE_DENSE_OVERLOAD(float _Complex, std::complex<float>, cf32)
// DEFINE_DENSE_OVERLOAD(double _Complex, std::complex<double>, cf64)
// DEFINE_DENSE_OVERLOAD(short, short, i16)
// DEFINE_DENSE_OVERLOAD(int, int, i32)
// DEFINE_DENSE_OVERLOAD(__int64_t, __int64_t, i64)

DEFINE_CSR_OVERLOAD(float, int, float, int, f32_i32, f32)
DEFINE_CSR_OVERLOAD(float, __int64_t, float, std::int64_t, f32_i64, f32)
DEFINE_CSR_OVERLOAD(double, int, double, int, f64_i32, f64)
DEFINE_CSR_OVERLOAD(double, __int64_t, double, std::int64_t, f64_i64, f64)
// DEFINE_CSR_OVERLOAD(double, __int16_t, double, std::int16_t, f64_i16, f64)
// DEFINE_CSR_OVERLOAD(float _Complex, int, std::complex<float>, int, cf32_i32,
// cf32); DEFINE_CSR_OVERLOAD(double _Complex, int, std::complex<double>, int,
// cf64_i32, cf64)

/* ----------------------------------------------------------------------
 * Library functions for BLAS linop in GINKGO
 * ---------------------------------------------------------------------- */
// function spmm!(A::SparseMatrixCsr{Tv, Ti}, α::Dense{Tv}, x::Dense{Tv},
// β::Dense{Tv}, y::Dense{Tv}) where {Tv, Ti}

// end


/* ----------------------------------------------------------------------
 * Library functions for iterative solvers in GINKGO
 * ---------------------------------------------------------------------- */
// void ginkgo_solver_cg_solve_f32_i32(gko_executor exec_st_ptr,
//                             gko_matrix_csr_f32_i32 A_st_ptr,
//                             gko_matrix_dense_f32 b_st_ptr,
//                             gko_matrix_dense_f32 x_st_ptr, int maxiter,
//                             double reduction)
// {
//     auto solver_gen =
//         gko::solver::Cg<float>::build()
//             .with_criteria(
//                 gko::stop::Iteration::build().with_max_iters(maxiter),
//                 gko::stop::ResidualNorm<float>::build().with_reduction_factor(
//                     reduction))
//             .on(exec_st_ptr->shared_ptr);

//     auto solver = solver_gen->generate(A_st_ptr->mat);

//     solver->apply(b_st_ptr->mat, x_st_ptr->mat);
// }


/* ----------------------------------------------------------------------
 * Library functions for retrieving configuration information in GINKGO
 * ---------------------------------------------------------------------- */

void ginkgo_version_get()
{
    std::cout << gko::version_info::get() << std::endl;
}
