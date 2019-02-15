/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres_kernels.hpp"


#include <iomanip>
#include <iostream>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace gmres {


constexpr int default_block_size = 512;


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type krylov_dim, const ValueType *__restrict__ b,
    ValueType *__restrict__ b_norm, ValueType *__restrict__ residual,
    ValueType *__restrict__ givens_sin, ValueType *__restrict__ givens_cos,
    stopping_status *__restrict__ stop_status)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    if (global_id < num_cols) {
        stop_status[global_id].reset();
    }

    if (global_id < num_rows * stride) {
        residual[global_id] = b[global_id];
    }

    if (global_id < krylov_dim * num_cols) {
        givens_sin[global_id] = zero<ValueType>();
        givens_cos[global_id] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize_1(
    std::shared_ptr<const CudaExecutor> exec, const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *b_norm, matrix::Dense<ValueType> *residual,
    matrix::Dense<ValueType> *givens_sin, matrix::Dense<ValueType> *givens_cos,
    Array<stopping_status> *stop_status, const size_type krylov_dim)
{
    const dim3 grid_dim(
        ceildiv(b->get_size()[0], default_block_size) * b->get_stride(), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    b->compute_norm2(b_norm);
    initialize_1_kernel<block_size><<<grid_dim, block_dim>>>(
        b->get_size()[0], b->get_size()[1], b->get_stride(), krylov_dim,
        as_cuda_type(b->get_const_values()), as_cuda_type(b_norm->get_values()),
        as_cuda_type(residual->get_values()),
        as_cuda_type(givens_sin->get_values()),
        as_cuda_type(givens_cos->get_values()),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_1_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    if (global_id < num_rows * (krylov_dim + 1) * num_cols) {
        krylov_bases[global_id] = 0;
    }

    if (global_id < (krylov_dim + 1) * num_cols) {
        residual_norm_collection[global_id] = 0;
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void initialize_2_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ residual,
    const ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases,
    size_type *__restrict__ final_iter_nums)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row = ceildiv(global_id + 1, num_cols) - 1;
    const auto column = global_id - row * num_cols;

    if (global_id < num_cols) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (global_id < num_rows * num_cols) {
        krylov_bases[row * stride + column] =
            residual[global_id] / residual_norm[column];
    }
}


template <typename ValueType>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, const size_type krylov_dim)
{
    const dim3 grid_dim(
        ceildiv(krylov_bases->get_size()[0], default_block_size) *
            krylov_bases->get_stride(),
        1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_2_1_kernel<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1], krylov_dim,
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(krylov_bases->get_values()));
    residual->compute_norm2(residual_norm);
    initialize_2_2_kernel<block_size><<<grid_dim, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        krylov_bases->get_stride(), as_cuda_type(residual->get_const_values()),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_type(krylov_bases->get_values()),
        as_cuda_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


__global__
    __launch_bounds__(default_block_size) void increase_final_iteration_numbers_kernel(
        size_type *__restrict__ final_iter_nums,
        const stopping_status *__restrict__ stop_status, size_type total_number)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < total_number) {
        final_iter_nums[tidx] += (1 - stop_status[tidx].has_stopped());
    }
}

template <typename ValueType>
void print_matrix(std::string name, const matrix::Dense<ValueType> *d_mtx)
{}

template <>
void print_matrix(std::string name, const matrix::Dense<double> *d_mtx)
{
    using ValueType = double;
    auto mtx =
        matrix::Dense<ValueType>::create(d_mtx->get_executor()->get_master());
    mtx->copy_from(d_mtx);

    const auto stride = mtx->get_stride();
    const auto dim = mtx->get_size();

    const auto dim0 = dim[0];
    const auto dim1 = dim[1];
    std::cout << name << "  dim = " << dim[0] << " x " << dim[1]
              << ", st = " << stride << "  ";
    std::cout << (d_mtx->get_executor() == d_mtx->get_executor()->get_master()
                      ? "ref"
                      : "cuda")
              << std::endl;
    for (auto i = 0; i < 20; ++i) {
        std::cout << '-';
    }
    std::cout << std::endl;

    for (size_type i = 0; i < dim[0]; ++i) {
        for (size_type j = 0; j < stride; ++j) {
            if (j == dim[1]) {
                std::cout << "| ";
            }
            const auto val = mtx->get_const_values()[i * stride + j];
            if (val == zero<ValueType>())
                std::cout << "0 ";
            else
                std::cout << val << ' ';
        }
        std::cout << std::endl;
    }

    for (auto i = 0; i < 20; ++i) {
        std::cout << '-';
    }
    std::cout << std::endl;
}


template <typename ValueType>
bool are_same_mtx(const gko::matrix::Dense<ValueType> *d_mtx1,
                  const gko::matrix::Dense<ValueType> *d_mtx2,
                  double error = 1e-12)
{
    auto mtx1 = gko::matrix::Dense<ValueType>::create(
        d_mtx1->get_executor()->get_master());
    mtx1->copy_from(d_mtx1);
    auto mtx2 = gko::matrix::Dense<ValueType>::create(
        d_mtx2->get_executor()->get_master());
    mtx2->copy_from(d_mtx2);
    auto get_error = [](const ValueType &v1, const ValueType &v2) {
        return std::abs((v1 - v2) / std::max(v1, v2));
    };
    auto size = mtx1->get_size();
    if (size != mtx2->get_size()) {
        std::cerr << "Mismatching sizes!!!\n";
        return false;
    }
    for (int j = 0; j < size[1]; ++j) {
        for (int i = 0; i < size[0]; ++i) {
            if (get_error(mtx1->at(i, j), mtx2->at(i, j)) > error) {
                std::cerr << "Problem at component (" << i << "," << j
                          << "): " << mtx1->at(i, j) << " != " << mtx2->at(i, j)
                          << " !!!\n";
                return false;
            }
            // std::cout << "All good for (" << i << "," << j << "): " <<
            // x->at(i,j) << " == " << x_host->at(i,j) << "\n";
        }
    }
    return true;
}


template <typename ValueType>
void compare_mtx(std::string name, const gko::matrix::Dense<ValueType> *d_mtx1,
                 const gko::matrix::Dense<ValueType> *d_mtx2,
                 double error = 1e-14)
{}


template <>
void compare_mtx(std::string name, const gko::matrix::Dense<double> *d_mtx1,
                 const gko::matrix::Dense<double> *d_mtx2, double error)
{
    if (!are_same_mtx(d_mtx1, d_mtx2, error)) {
        print_matrix(name, d_mtx1);
        print_matrix(name, d_mtx2);
    }
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2
template <int block_size = default_block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_hessenberg_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, const ValueType *__restrict__ krylov_bases,
    size_type stride_krylov, ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto global_id = blockIdx.x * blockDim.x + tidx;
    const auto col_idx = global_id / block_size;

    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    __shared__ UninitializedArray<ValueType, block_size> reduction_helper_array;
    ValueType *__restrict__ reduction_helper = reduction_helper_array;


    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        ValueType local_res{};
        const auto krylov_col = k * num_cols + col_idx;

        for (size_type i = tidx; i < num_rows; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto krylov_idx = i * stride_krylov + krylov_col;

            local_res +=
                next_krylov_basis[next_krylov_idx] * krylov_bases[krylov_idx];
        }

        reduction_helper[tidx] = local_res;

        // Perform thread block reduction. Result is in reduction_helper[0]
        reduce(group::this_thread_block(), reduction_helper,
               [](const ValueType &a, const ValueType &b) { return a + b; });

        if (tidx == 0) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;
            hessenberg_iter[hessenberg_idx] = reduction_helper[0];
        }
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in total
template <int block_size = default_block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto krylov_idx =
            row_idx * stride_krylov + k * num_cols + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] * krylov_bases[krylov_idx];
    }
}


// Must be called with at least `num_cols` threads in total
template <int block_size = default_block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_hessenberg_2_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        ValueType next_hessenberg = zero<ValueType>();
        for (size_type i = 0; i < num_rows; ++i) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto next_krylov_value = next_krylov_basis[next_krylov_idx];
            next_hessenberg += next_krylov_value * next_krylov_value;
        }
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
            sqrt(next_hessenberg);
    }
}


// Must be called with at least `num_rows * stride_next_krylov` threads in total
template <typename ValueType>
__global__
    __launch_bounds__(default_block_size) void update_krylov_next_krylov_kernel(
        size_type iter, size_type num_rows, size_type num_cols,
        ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
        ValueType *__restrict__ krylov_bases, size_type stride_krylov,
        const ValueType *__restrict__ hessenberg_iter,
        size_type stride_hessenberg,
        const stopping_status *__restrict__ stop_status)
{
    const auto global_id = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        const auto krylov_idx =
            row_idx * stride_krylov + num_cols * (iter + 1) + col_idx;

        const auto next_krylov_value =
            next_krylov_basis[next_krylov_idx] / hessenberg;

        next_krylov_basis[next_krylov_idx] = next_krylov_value;
        krylov_bases[krylov_idx] = next_krylov_value;
    }
}

template <typename ValueType>
void finish_arnoldi_cpu(matrix::Dense<ValueType> *next_krylov_basis,
                        matrix::Dense<ValueType> *krylov_bases,
                        matrix::Dense<ValueType> *hessenberg_iter,
                        const size_type iter)
{
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        for (size_type k = 0; k < iter + 1; ++k) {
            hessenberg_iter->at(k, i) = 0;
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                hessenberg_iter->at(k, i) +=
                    next_krylov_basis->at(j, i) *
                    krylov_bases->at(j,
                                     next_krylov_basis->get_size()[1] * k + i);
            }
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                next_krylov_basis->at(j, i) -=
                    hessenberg_iter->at(k, i) *
                    krylov_bases->at(j,
                                     next_krylov_basis->get_size()[1] * k + i);
            }
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end

        //*
        hessenberg_iter->at(iter + 1, i) = 0;
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            hessenberg_iter->at(iter + 1, i) +=
                next_krylov_basis->at(j, i) * next_krylov_basis->at(j, i);
        }
        hessenberg_iter->at(iter + 1, i) =
            sqrt(hessenberg_iter->at(iter + 1, i));
        // hessenberg(iter, iter + 1) = norm(next_krylov_basis)

        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            next_krylov_basis->at(j, i) /= hessenberg_iter->at(iter + 1, i);
            krylov_bases->at(j, next_krylov_basis->get_size()[1] * (iter + 1) +
                                    i) = next_krylov_basis->at(j, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // End of arnoldi
        //*/
    }
}

template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Dense<ValueType> *next_krylov_basis,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter,
                    const size_type iter, const stopping_status *stop_status)
{
    // Store cout state to restore it later.
    std::ios oldCoutState(nullptr);
    oldCoutState.copyfmt(std::cout);

    std::cout << std::setprecision(15);
    std::cout << std::scientific;

    const auto stride_next_krylov = next_krylov_basis->get_stride();
    const auto stride_krylov = krylov_bases->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    const auto dim_size = next_krylov_basis->get_size();

    /*
        auto h_nkb = matrix::Dense<ValueType>::create(exec->get_master());
        h_nkb->copy_from(next_krylov_basis);
        auto h_kb = matrix::Dense<ValueType>::create(exec->get_master());
        h_kb->copy_from(krylov_bases);
        auto h_hi = matrix::Dense<ValueType>::create(exec->get_master());
        h_hi->copy_from(hessenberg_iter);

        finish_arnoldi_cpu(h_nkb.get(), h_kb.get(), h_hi.get(), iter);
    //*/

    for (size_type k = 0; k < iter + 1; ++k) {
        update_hessenberg_kernel<<<ceildiv((iter + 1) * stride_hessenberg,
                                           default_block_size),
                                   default_block_size>>>(
            k, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_const_values()),
            stride_next_krylov, as_cuda_type(krylov_bases->get_const_values()),
            stride_krylov, as_cuda_type(hessenberg_iter->get_values()),
            stride_hessenberg, as_cuda_type(stop_status));

        update_next_krylov_kernel<<<ceildiv(dim_size[0] * stride_next_krylov,
                                            default_block_size),
                                    default_block_size>>>(
            k, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_values()), stride_next_krylov,
            as_cuda_type(krylov_bases->get_const_values()), stride_krylov,
            as_cuda_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_cuda_type(stop_status));
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end


    update_hessenberg_2_kernel<<<ceildiv(dim_size[1], default_block_size),
                                 default_block_size>>>(
        iter, dim_size[0], dim_size[1],
        as_cuda_type(next_krylov_basis->get_const_values()), stride_next_krylov,
        as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
        as_cuda_type(stop_status));

    update_krylov_next_krylov_kernel<<<ceildiv(dim_size[0] * stride_next_krylov,
                                               default_block_size),
                                       default_block_size>>>(
        iter, dim_size[0], dim_size[1],
        as_cuda_type(next_krylov_basis->get_values()), stride_next_krylov,
        as_cuda_type(krylov_bases->get_values()), stride_krylov,
        as_cuda_type(hessenberg_iter->get_const_values()), stride_hessenberg,
        as_cuda_type(stop_status));
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // End of arnoldi

    /*
    constexpr double accuracy = 1e-13;
    compare_mtx(std::string("next_krylov INTERNAL ") + std::to_string(iter),
                next_krylov_basis, h_nkb.get(), accuracy);
    compare_mtx(std::string("krylov INTERNAL ") + std::to_string(iter),
                krylov_bases, h_kb.get(), accuracy);
    compare_mtx(std::string("hessenberg INTERNAL ") + std::to_string(iter),
                hessenberg_iter, h_hi.get(), accuracy);
    next_krylov_basis->copy_from(h_nkb.get());
    krylov_bases->copy_from(h_kb.get());
    hessenberg_iter->copy_from(h_hi.get());
    //*/


    // Restore cout settings
    std::cout.copyfmt(oldCoutState);
}


template <typename ValueType>
__device__ void calculate_sin_and_cos(
    const size_type num_cols, const ValueType *hessenberg_iter,
    size_type stride_hessenberg, ValueType *givens_sin, size_type stride_sin,
    ValueType *givens_cos, size_type stride_cos, const size_type iter)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    if (hessenberg_iter[iter * stride_hessenberg + local_id] ==
        zero<ValueType>()) {
        givens_cos[iter * stride_cos + local_id] = zero<ValueType>();
        givens_sin[iter * stride_sin + local_id] = one<ValueType>();
    } else {
        auto hypotenuse = sqrt(
            hessenberg_iter[iter * stride_hessenberg + local_id] *
                hessenberg_iter[iter * stride_hessenberg + local_id] +
            hessenberg_iter[(iter + 1) * stride_hessenberg + local_id] *
                hessenberg_iter[(iter + 1) * stride_hessenberg + local_id]);
        givens_cos[iter * stride_cos + local_id] =
            abs(hessenberg_iter[iter * stride_hessenberg + local_id]) /
            hypotenuse;
        givens_sin[iter * stride_sin + local_id] =
            givens_cos[iter * stride_cos + local_id] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + local_id] /
            hessenberg_iter[iter * stride_hessenberg + local_id];
    }
}


template <typename ValueType>
__device__ void calculate_residual_norm(
    const size_type num_cols, const ValueType *givens_sin, size_type stride_sin,
    const ValueType *givens_cos, size_type stride_cos, ValueType *residual_norm,
    ValueType *residual_norm_collection,
    size_type stride_residual_norm_collection, const ValueType *b_norm,
    const size_type iter)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             local_id] =
        -givens_sin[iter * stride_sin + local_id] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 local_id];
    residual_norm_collection[iter * stride_residual_norm_collection +
                             local_id] =
        givens_cos[iter * stride_cos + local_id] *
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 local_id];
    residual_norm[local_id] =
        abs(residual_norm_collection[(iter + 1) *
                                         stride_residual_norm_collection +
                                     local_id]) /
        b_norm[local_id];
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void givens_rotation_kernel(
    const size_type num_rows, const size_type num_cols,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    ValueType *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ b_norm, const size_type iter,
    const stopping_status *__restrict__ stop_status)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();
    __shared__ UninitializedArray<ValueType, block_size> tmp;

    if (local_id >= num_cols || stop_status[local_id].has_stopped()) return;

    for (size_type i = 0; i < iter; ++i) {
        tmp[local_id] =
            givens_cos[i * stride_cos + local_id] *
                hessenberg_iter[i * stride_hessenberg + local_id] +
            givens_sin[i * stride_sin + local_id] *
                hessenberg_iter[(i + 1) * stride_hessenberg + local_id];
        __syncthreads();
        hessenberg_iter[(i + 1) * stride_hessenberg + local_id] =
            givens_cos[i * stride_cos + local_id] *
                hessenberg_iter[(i + 1) * stride_hessenberg + local_id] -
            givens_sin[i * stride_sin + local_id] *
                hessenberg_iter[i * stride_hessenberg + local_id];
        hessenberg_iter[i * stride_hessenberg + local_id] = tmp[local_id];
        __syncthreads();
    }
    // for j in 1:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    calculate_sin_and_cos(num_cols, hessenberg_iter, stride_hessenberg,
                          givens_sin, stride_sin, givens_cos, stride_cos, iter);
    // Calculate sin and cos

    hessenberg_iter[iter * stride_hessenberg + local_id] =
        givens_cos[iter * stride_cos + local_id] *
            hessenberg_iter[iter * stride_hessenberg + local_id] +
        givens_sin[iter * stride_sin + local_id] *
            hessenberg_iter[(iter + 1) * stride_hessenberg + local_id];
    hessenberg_iter[(iter + 1) * stride_hessenberg + local_id] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter)
    // hessenberg(iter+1) = 0

    calculate_residual_norm(num_cols, givens_sin, stride_sin, givens_cos,
                            stride_cos, residual_norm, residual_norm_collection,
                            stride_residual_norm_collection, b_norm, iter);
    // Calculate residual norm
}


template <typename ValueType>
void givens_rotation(std::shared_ptr<const CudaExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<ValueType> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     const matrix::Dense<ValueType> *b_norm,
                     const size_type iter,
                     const Array<stopping_status> *stop_status)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         block_size / cuda_config::warp_size};

    givens_rotation_kernel<block_size><<<1, block_dim>>>(
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1],
        as_cuda_type(hessenberg_iter->get_values()),
        hessenberg_iter->get_stride(), as_cuda_type(givens_sin->get_values()),
        givens_sin->get_stride(), as_cuda_type(givens_cos->get_values()),
        givens_cos->get_stride(), as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(b_norm->get_const_values()), iter,
        as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, const size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
    increase_final_iteration_numbers_kernel<<<
        ceildiv(final_iter_nums->get_num_elems(), default_block_size),
        default_block_size>>>(as_cuda_type(final_iter_nums->get_data()),
                              as_cuda_type(stop_status->get_const_data()),
                              final_iter_nums->get_num_elems());
    finish_arnoldi(exec, next_krylov_basis, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, b_norm, iter,
                    stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ hessenberg, size_type stride_hessenberg,
    ValueType *__restrict__ y, size_type stride_y,
    const size_type *__restrict__ final_iter_nums)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    if (local_id >= num_rhs) return;

    for (int i = final_iter_nums[local_id] - 1; i >= 0; --i) {
        auto temp =
            residual_norm_collection[i * stride_residual_norm_collection +
                                     local_id];
        for (size_type j = i + 1; j < final_iter_nums[local_id]; ++j) {
            temp -= hessenberg[i * stride_hessenberg + j * num_rhs + local_id] *
                    y[j * stride_y + local_id];
        }

        y[i * stride_y + local_id] =
            temp / hessenberg[i * stride_hessenberg + i * num_rhs + local_id];
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void calculate_Qy_kernel(
    const size_type num_rows, const size_type num_cols, const size_type num_rhs,
    const ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ before_preconditioner,
    const size_type *__restrict__ final_iter_nums)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_rhs;
    const auto col_id = global_id % num_rhs;

    if (row_id < num_rows && col_id < num_cols) {
        before_preconditioner[global_id] = zero<ValueType>();
        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            before_preconditioner[global_id] +=
                krylov_bases[row_id * stride_krylov + j * num_rhs + col_id] *
                y[j * stride_y + col_id];
        }
    }
}


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const Array<size_type> *final_iter_nums)
{
    // TODO: tune this parameter
    // TODO: when number of right hand side is larger than block_size
    constexpr auto block_size = default_block_size;
    const dim3 block_dim{cuda_config::warp_size, 1,
                         ceildiv(block_size, cuda_config::warp_size)};

    solve_upper_triangular_kernel<block_size><<<1, block_dim>>>(
        hessenberg->get_size()[1], residual_norm_collection->get_size()[1],
        as_cuda_type(residual_norm_collection->get_const_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(hessenberg->get_const_values()), hessenberg->get_stride(),
        as_cuda_type(y->get_values()), y->get_stride(),
        as_cuda_type(final_iter_nums->get_const_data()));
}


template <typename ValueType>
void solve_x(std::shared_ptr<const CudaExecutor> exec,
             const matrix::Dense<ValueType> *krylov_bases,
             const matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
             const Array<size_type> *final_iter_nums,
             const LinOp *preconditioner)
{
    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
    const dim3 block_dim{cuda_config::warp_size, 1,
                         ceildiv(block_size, cuda_config::warp_size)};

    calculate_Qy_kernel<block_size><<<grid_dim, block_dim>>>(
        before_preconditioner->get_size()[0], krylov_bases->get_size()[1],
        before_preconditioner->get_size()[1],
        as_cuda_type(krylov_bases->get_const_values()),
        krylov_bases->get_stride(), as_cuda_type(y->get_const_values()),
        y->get_stride(),
        as_cuda_type(before_preconditioner.get()->get_values()),
        as_cuda_type(final_iter_nums->get_const_data()));

    preconditioner->apply(before_preconditioner.get(),
                          after_preconditioner.get());

    auto one_op =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    x->add_scaled(one_op.get(), after_preconditioner.get());
    // Solve x
    // x = x + preconditioner_ * krylov_bases * y
}


template <typename ValueType>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums,
            const LinOp *preconditioner)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    solve_x(exec, krylov_bases, y, x, final_iter_nums, preconditioner);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
