/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#include "core/matrix/coo_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "gpu/base/cusparse_bindings.hpp"
#include "gpu/base/math.hpp"
#include "gpu/base/types.hpp"
#include "gpu/components/shuffle.cuh"
#include "gpu/components/synchronization.cuh"

namespace gko {
namespace kernels {
namespace gpu {
namespace coo {


constexpr int default_block_size = 512;


namespace {


__forceinline__ __device__ static float atomic_add(float *addr, float val)
{
    return atomicAdd(addr, val);
}


#if (defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))
__forceinline__ __device__ static double atomic_add(double *addr, double val)
{
    double old = *addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(atomicCAS(
            (unsigned long long int *)addr, __double_as_longlong(assumed),
            __double_as_longlong(val + assumed)));
    } while (assumed != old);

    return old;
}
#else
__forceinline__ __device__ static double atomic_add(double *addr, double val)
{
    return atomicAdd(addr, val);
}
#endif


__forceinline__ __device__ static cuDoubleComplex atomic_add(
    cuDoubleComplex *address, cuDoubleComplex val)
{
    // Seperate to real part and imag part
    // real part
    atomic_add(&(address->x), val.x);
    // imag part
    atomic_add(&(address->y), val.y);
    return *address;
}


__forceinline__ __device__ static cuComplex atomic_add(cuComplex *address,
                                                       cuComplex val)
{
    // Seperate to real part and imag part
    // real part
    atomic_add(&(address->x), val.x);
    // imag part
    atomic_add(&(address->y), val.y);
    return *address;
}


__forceinline__ __device__ static thrust::complex<float> atomic_add(
    thrust::complex<float> *address, thrust::complex<float> val)
{
    cuComplex *cuaddr = reinterpret_cast<cuComplex *>(address);
    cuComplex *cuval = reinterpret_cast<cuComplex *>(&val);
    atomic_add(cuaddr, *cuval);
    return *address;
}


__forceinline__ __device__ static thrust::complex<double> atomic_add(
    thrust::complex<double> *address, thrust::complex<double> val)
{
    cuDoubleComplex *cuaddr = reinterpret_cast<cuDoubleComplex *>(address);
    cuDoubleComplex *cuval = reinterpret_cast<cuDoubleComplex *>(&val);
    atomic_add(cuaddr, *cuval);
    return *address;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ void segment_scan(IndexType *ind, ValueType *val,
                                             bool *head)
{
    ValueType add_val;
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        const IndexType add_ind = warp::shuffle_up(*ind, i);
        add_val = zero<ValueType>();
        if (threadIdx.x >= i && add_ind == *ind) {
            add_val = *val;
            if (i == 1) {
                *head = false;
            }
        }
        add_val = warp::shuffle_down(add_val, i);
        if (threadIdx.x < 32 - i) {
            *val += add_val;
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(128) void spmv_kernel(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    size_type num = (nnz > start) * ceildiv(nnz - start, 32);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * 32;
    IndexType ind = ind_start;
    bool atomichead = true;
    IndexType temp_row = (num > 0) ? row[ind] : 0;
    IndexType next_row;
    for (; ind < ind_end; ind += 32) {
        temp_val += (ind > nnz) ? zero<ValueType>() : val[ind] * b[col[ind]];
        next_row = (ind + 32 > nnz) ? row[nnz - 1] : row[ind + 32];
        // segmented scan
        const bool is_scan = temp_row != next_row;
        if (warp::any(is_scan)) {
            atomichead = true;
            segment_scan(&temp_row, &temp_val, &atomichead);
            if (atomichead) {
                atomic_add(&(c[temp_row]), temp_val);
            }
            temp_val = 0;
        }
        temp_row = next_row;
    }
    if (num > 0) {
        ind = ind_start + (num - 1) * 32;
        temp_val += (ind > nnz) ? zero<ValueType>() : val[ind] * b[col[ind]];
        // segmented scan
        atomichead = true;
        segment_scan(&temp_row, &temp_val, &atomichead);
        if (atomichead) {
            atomic_add(&(c[temp_row]), temp_val);
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void set_zero(
    const size_type nnz, ValueType *__restrict__ val)
{
    const auto ind =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ind < nnz) {
        val[ind] = zero<ValueType>();
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const GpuExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    int multiple = 8;
    auto nnz = a->get_num_stored_elements();
    const dim3 grid(ceildiv(nnz, default_block_size));
    const dim3 block(default_block_size);
    set_zero<<<grid, block>>>(c->get_num_stored_elements(),
                              as_cuda_type(c->get_values()));
    if (nnz >= 2000000) {
        multiple = 128;
    } else if (nnz >= 200000) {
        multiple = 32;
    }
    const int warps_per_block = 4;
    // TODO: get config from GPUExecutor
    int config = 112;
    int nwarps = config * multiple;
    if (nwarps > ceildiv(nnz, 32)) {
        nwarps = ceildiv(nnz, 32);
    }
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * 32);
        const dim3 coo_block(32, warps_per_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_per_block));
        spmv_kernel<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()), as_cuda_type(c->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(128) void advanced_spmv_kernel(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const ValueType *__restrict__ b, ValueType *__restrict__ c)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    size_type num = (nnz > start) * ceildiv(nnz - start, 32);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * 32;
    IndexType ind = ind_start;
    bool atomichead = true;
    IndexType temp_row = (num > 0) ? row[ind] : 0;
    IndexType next_row;
    const auto alpha_val = alpha[0];
    for (; ind < ind_end; ind += 32) {
        temp_val += (ind > nnz) ? zero<ValueType>() : val[ind] * b[col[ind]];
        next_row = (ind + 32 > nnz) ? row[nnz - 1] : row[ind + 32];
        // segmented scan
        const bool is_scan = temp_row != next_row;
        if (warp::any(is_scan)) {
            atomichead = true;
            segment_scan(&temp_row, &temp_val, &atomichead);
            if (atomichead) {
                atomic_add(&(c[temp_row]), alpha_val * temp_val);
            }
            temp_val = 0;
        }
        temp_row = next_row;
    }
    if (num > 0) {
        ind = ind_start + (num - 1) * 32;
        temp_val += (ind > nnz) ? zero<ValueType>() : val[ind] * b[col[ind]];
        // segmented scan
        atomichead = true;
        segment_scan(&temp_row, &temp_val, &atomichead);
        if (atomichead) {
            atomic_add(&(c[temp_row]), alpha_val * temp_val);
        }
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const GpuExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    int multiple = 8;
    auto nnz = a->get_num_stored_elements();
    if (nnz >= 2000000) {
        multiple = 128;
    } else if (nnz >= 200000) {
        multiple = 32;
    }
    const int warps_per_block = 4;
    // TODO: get config from GPUExecutor
    int config = 112;
    int nwarps = config * multiple;
    if (nwarps > ceildiv(nnz, 32)) {
        nwarps = ceildiv(nnz, 32);
    }
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * 32);
        const dim3 coo_block(32, warps_per_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_per_block));
        advanced_spmv_kernel<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()), as_cuda_type(c->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const GpuExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs,
                              size_type length) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_ROW_IDXS_TO_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const GpuExecutor> exec,
               matrix::Coo<ValueType, IndexType> *trans,
               const matrix::Coo<ValueType, IndexType> *orig) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *trans,
                    const matrix::Coo<ValueType, IndexType> *orig)
    NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const GpuExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Coo<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
