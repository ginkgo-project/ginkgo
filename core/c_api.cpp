// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/c_api.h>
#include <cstring>
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
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

/* ----------------------------------------------------------------------
 * C memory management
 * ---------------------------------------------------------------------- */
void c_char_ptr_free(char* ptr) { delete[] ptr; }

/* ----------------------------------------------------------------------
 * Library functions for retrieving configuration information in GINKGO
 * ---------------------------------------------------------------------- */
void ginkgo_version_get()
{
    std::cout << gko::version_info::get() << std::endl;
}

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
 * Library functions for executors (Creation, Getters) in GINKGO
 * ---------------------------------------------------------------------- */
struct gko_executor_st {
    std::shared_ptr<gko::Executor> shared_ptr;
};

void ginkgo_executor_delete(gko_executor exec_st_ptr) { delete exec_st_ptr; }

gko_executor ginkgo_executor_get_master(gko_executor exec_st_ptr)
{
    return new gko_executor_st{exec_st_ptr->shared_ptr->get_master()};
}

bool ginkgo_executor_memory_accessible(gko_executor exec_st_ptr,
                                       gko_executor other_exec_st_ptr)
{
    return exec_st_ptr->shared_ptr->memory_accessible(
        other_exec_st_ptr->shared_ptr);
}

void ginkgo_executor_synchronize(gko_executor exec_st_ptr)
{
    exec_st_ptr->shared_ptr->synchronize();
}

//---------------------------- CPU -----------------------------
gko_executor ginkgo_executor_omp_create()
{
    return new gko_executor_st{gko::OmpExecutor::create()};
}

gko_executor ginkgo_executor_reference_create()
{
    return new gko_executor_st{gko::ReferenceExecutor::create()};
}

size_t ginkgo_executor_cpu_get_num_cores(gko_executor exec_st_ptr)
{
    if (auto omp_exec = std::dynamic_pointer_cast<gko::OmpExecutor>(
            exec_st_ptr->shared_ptr)) {
        return omp_exec->get_num_cores();
    } else if (auto ref_exec =
                   std::dynamic_pointer_cast<gko::ReferenceExecutor>(
                       exec_st_ptr->shared_ptr)) {
        return ref_exec->get_num_cores();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_cpu_get_num_threads_per_core(gko_executor exec_st_ptr)
{
    if (auto omp_exec = std::dynamic_pointer_cast<gko::OmpExecutor>(
            exec_st_ptr->shared_ptr)) {
        return omp_exec->get_num_threads_per_core();
    } else if (auto ref_exec =
                   std::dynamic_pointer_cast<gko::ReferenceExecutor>(
                       exec_st_ptr->shared_ptr)) {
        return ref_exec->get_num_threads_per_core();
    } else {
        return 0;
    }
}

//---------------------------- GPU -----------------------------
size_t ginkgo_executor_gpu_get_device_id(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_device_id();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_device_id();
    } else if (auto dpcpp_exec = std::dynamic_pointer_cast<gko::DpcppExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return dpcpp_exec->get_device_id();
    } else {
        return 0;
    }
}

// CUDA/HIP
gko_executor ginkgo_executor_cuda_create(size_t device_id,
                                         gko_executor exec_st_ptr)
{
    if (auto omp_exec = std::dynamic_pointer_cast<gko::OmpExecutor>(
            exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::CudaExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else if (auto reference_exec =
                   std::dynamic_pointer_cast<gko::ReferenceExecutor>(
                       exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::CudaExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else {
        return nullptr;
    }
}

size_t ginkgo_executor_cuda_get_num_devices()
{
    return gko::CudaExecutor::get_num_devices();
}

gko_executor ginkgo_executor_hip_create(size_t device_id,
                                        gko_executor exec_st_ptr)
{
    if (auto omp_exec = std::dynamic_pointer_cast<gko::OmpExecutor>(
            exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::HipExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else if (auto reference_exec =
                   std::dynamic_pointer_cast<gko::ReferenceExecutor>(
                       exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::HipExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else {
        return nullptr;
    }
}

size_t ginkgo_executor_hip_get_num_devices()
{
    return gko::HipExecutor::get_num_devices();
}

size_t ginkgo_executor_gpu_thread_get_num_multiprocessor(
    gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_num_multiprocessor();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_num_multiprocessor();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_num_warps_per_sm(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_num_warps_per_sm();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_num_warps_per_sm();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_num_warps(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_num_warps();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_num_warps();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_warp_size(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_warp_size();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_warp_size();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_major_version(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_major_version();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_major_version();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_minor_version(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_minor_version();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_minor_version();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_thread_get_closest_numa(gko_executor exec_st_ptr)
{
    if (auto cuda_exec = std::dynamic_pointer_cast<gko::CudaExecutor>(
            exec_st_ptr->shared_ptr)) {
        return cuda_exec->get_closest_numa();
    } else if (auto hip_exec = std::dynamic_pointer_cast<gko::HipExecutor>(
                   exec_st_ptr->shared_ptr)) {
        return hip_exec->get_closest_numa();
    } else {
        return 0;
    }
}

// DPCPP/SYCL
gko_executor ginkgo_executor_dpcpp_create(size_t device_id,
                                          gko_executor exec_st_ptr)
{
    if (auto omp_exec = std::dynamic_pointer_cast<gko::OmpExecutor>(
            exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::DpcppExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else if (auto reference_exec =
                   std::dynamic_pointer_cast<gko::ReferenceExecutor>(
                       exec_st_ptr->shared_ptr)) {
        return new gko_executor_st{
            gko::DpcppExecutor::create(device_id, exec_st_ptr->shared_ptr)};
    } else {
        return nullptr;
    }
}

size_t ginkgo_executor_dpcpp_get_num_devices()
{
    return gko::DpcppExecutor::get_num_devices("gpu");
}

size_t ginkgo_executor_gpu_item_get_max_subgroup_size(gko_executor exec_st_ptr)
{
    if (auto dpcpp_exec = std::dynamic_pointer_cast<gko::DpcppExecutor>(
            exec_st_ptr->shared_ptr)) {
        return dpcpp_exec->get_max_subgroup_size();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_item_get_max_workgroup_size(gko_executor exec_st_ptr)
{
    if (auto dpcpp_exec = std::dynamic_pointer_cast<gko::DpcppExecutor>(
            exec_st_ptr->shared_ptr)) {
        return dpcpp_exec->get_max_workgroup_size();
    } else {
        return 0;
    }
}

size_t ginkgo_executor_gpu_item_get_num_computing_units(
    gko_executor exec_st_ptr)
{
    if (auto dpcpp_exec = std::dynamic_pointer_cast<gko::DpcppExecutor>(
            exec_st_ptr->shared_ptr)) {
        return dpcpp_exec->get_num_computing_units();
    } else {
        return 0;
    }
}

/* ----------------------------------------------------------------------
 * Library functions for creating arrays and array operations in GINKGO
 * ---------------------------------------------------------------------- */
DEFINE_ARRAY_OVERLOAD(int16_t, int16_t, i16)
DEFINE_ARRAY_OVERLOAD(int, int, i32)
DEFINE_ARRAY_OVERLOAD(int64_t, std::int64_t, i64)
DEFINE_ARRAY_OVERLOAD(float, float, f32)
DEFINE_ARRAY_OVERLOAD(double, double, f64)

/* ----------------------------------------------------------------------
 * Library functions for creating matrices and matrix operations in GINKGO
 * ---------------------------------------------------------------------- */
DEFINE_DENSE_OVERLOAD(float, float, f32)
DEFINE_DENSE_OVERLOAD(double, double, f64)

DEFINE_CSR_OVERLOAD(float, int, float, int, f32_i32, f32)
DEFINE_CSR_OVERLOAD(float, int64_t, float, std::int64_t, f32_i64, f32)
DEFINE_CSR_OVERLOAD(double, int, double, int, f64_i32, f64)
DEFINE_CSR_OVERLOAD(double, int64_t, double, std::int64_t, f64_i64, f64)

/* ----------------------------------------------------------------------
 * Library functions for deferred factory parameters in GINKGO
 * ---------------------------------------------------------------------- */
struct gko_deferred_factory_parameter_st {
    gko::deferred_factory_parameter<const gko::LinOpFactory> parameter;
};

void ginkgo_deferred_factory_parameter_delete(
    gko_deferred_factory_parameter dfp_st_ptr)
{
    delete dfp_st_ptr;
}

//-------------------- Preconditioner -----------------------------
gko_deferred_factory_parameter ginkgo_preconditioner_none_create()
{
    return new gko_deferred_factory_parameter_st{};
}

gko_deferred_factory_parameter ginkgo_preconditioner_jacobi_f64_i32_create(
    int blocksize)
{
    return new gko_deferred_factory_parameter_st{
        gko::preconditioner::Jacobi<double, int>::build().with_max_block_size(
            static_cast<gko::uint32>(blocksize))};
}

gko_deferred_factory_parameter ginkgo_preconditioner_ilu_f64_i32_create(
    gko_deferred_factory_parameter dfp_st_ptr)
{
    // Generate an ILU preconditioner factory by setting lower and upper
    // triangular solver - in this case the exact triangular solves
    return new gko_deferred_factory_parameter_st{
        gko::preconditioner::Ilu<gko::solver::LowerTrs<double, int>,
                                 gko::solver::UpperTrs<double, int>,
                                 false>::build()
            .with_factorization(dfp_st_ptr->parameter)};
}

//-------------------- Factorization ------------------------------
gko_deferred_factory_parameter ginkgo_factorization_parilu_f64_i32_create(
    int iteration, bool skip_sorting)
{
    // Generate factors using ParILU
    return new gko_deferred_factory_parameter_st{
        gko::factorization::ParIlu<double, int>::build()
            .with_iterations(static_cast<gko::uint32>(iteration))
            .with_skip_sorting(skip_sorting)};
}

/* ----------------------------------------------------------------------
 * Library functions for LinOp objects in GINKGO
 * ---------------------------------------------------------------------- */
struct gko_linop_st {
    std::shared_ptr<gko::LinOp> shared_ptr;
};

void ginkgo_linop_delete(gko_linop linop_st_ptr) { delete linop_st_ptr; }

void ginkgo_linop_apply(gko_linop A_st_ptr, gko_linop b_st_ptr,
                        gko_linop x_st_ptr)
{
    (A_st_ptr->shared_ptr)->apply(b_st_ptr->shared_ptr, x_st_ptr->shared_ptr);
}

//-------------------- Iterative solvers -----------------------------
gko_linop ginkgo_linop_cg_preconditioned_f64_create(
    gko_executor exec_st_ptr, gko_linop A_st_ptr,
    gko_deferred_factory_parameter dfp_st_ptr, double reduction, int maxiter)
{
    return new gko_linop_st{
        gko::solver::Cg<double>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(
                    static_cast<gko::uint32>(maxiter)),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    reduction))
            .with_preconditioner(dfp_st_ptr->parameter)
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}

gko_linop ginkgo_linop_gmres_preconditioned_f64_create(
    gko_executor exec_st_ptr, gko_linop A_st_ptr,
    gko_deferred_factory_parameter dfp_st_ptr, double reduction, int maxiter)
{
    return new gko_linop_st{
        gko::solver::Gmres<double>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(
                    static_cast<gko::uint32>(maxiter)),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    reduction))
            .with_preconditioner(dfp_st_ptr->parameter)
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}

//-------------------- Direct solvers -----------------------------
gko_linop ginkgo_linop_spd_direct_f64_i64_create(gko_executor exec_st_ptr,
                                                 gko_linop A_st_ptr)
{
    return new gko_linop_st{
        gko::experimental::solver::Direct<double, long>::build()
            .with_factorization(
                gko::experimental::factorization::Cholesky<double,
                                                           long>::build())
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}

gko_linop ginkgo_linop_lu_direct_f64_i64_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr)
{
    return new gko_linop_st{
        gko::experimental::solver::Direct<double, long>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<double, long>::build())
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}

gko_linop ginkgo_linop_lu_direct_f64_i32_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr)
{
    return new gko_linop_st{
        gko::experimental::solver::Direct<double, int>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<double, int>::build())
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}

gko_linop ginkgo_linop_lu_direct_f32_i32_create(gko_executor exec_st_ptr,
                                                gko_linop A_st_ptr)
{
    return new gko_linop_st{
        gko::experimental::solver::Direct<float, int>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<float, int>::build())
            .on(exec_st_ptr->shared_ptr)
            ->generate(A_st_ptr->shared_ptr)};
}
