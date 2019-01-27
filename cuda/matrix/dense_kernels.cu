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

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace dense {


constexpr auto default_block_size = 512;


template <typename ValueType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_size()[1],
                     c->get_size()[0], a->get_size()[1], &alpha,
                     b->get_const_values(), b->get_stride(),
                     a->get_const_values(), a->get_stride(), &beta,
                     c->get_values(), c->get_stride());
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    } else {
        NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        cublas::gemm(exec->get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                     c->get_size()[1], c->get_size()[0], a->get_size()[1],
                     alpha->get_const_values(), b->get_const_values(),
                     b->get_stride(), a->get_const_values(), a->get_stride(),
                     beta->get_const_values(), c->get_values(),
                     c->get_stride());
    } else {
        NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void scale(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, ValueType *__restrict__ x,
    size_type stride_x)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows) {
        x[row_id * stride_x + col_id] =
            alpha[alpha_id] == zero<ValueType>()
                ? zero<ValueType>()
                : x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}


}  // namespace kernel


template <typename ValueType>
void scale(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::scal(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_values(),
                     x->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{cuda_config::warp_size, 1,
                             block_size / cuda_config::warp_size};
        kernel::scale<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_values()), x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void add_scaled(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ x,
    size_type stride_x, ValueType *__restrict__ y, size_type stride_y)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows && alpha[alpha_id] != zero<ValueType>()) {
        y[row_id * stride_y + col_id] +=
            x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}


}  // namespace kernel


template <typename ValueType>
void add_scaled(std::shared_ptr<const CudaExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::axpy(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_const_values(),
                     x->get_stride(), y->get_values(), y->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{cuda_config::warp_size, 1,
                             block_size / cuda_config::warp_size};
        kernel::add_scaled<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_const_values()), x->get_stride(),
            as_cuda_type(y->get_values()), y->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


namespace kernel {


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void compute_partial_dot(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ work)
{
    constexpr auto warps_per_block = block_size / cuda_config::warp_size;

    const auto num_blocks = gridDim.x;
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size, warps_per_block>();

    auto tmp = zero<ValueType>();
    for (auto i = global_id; i < num_rows; i += block_size * num_blocks) {
        tmp += x[i * stride_x] * y[i * stride_y];
    }
    __shared__ UninitializedArray<ValueType, block_size> tmp_work;
    tmp_work[local_id] = tmp;

    reduce(group::this_thread_block(), static_cast<ValueType *>(tmp_work),
           [](const ValueType &x, const ValueType &y) { return x + y; });

    if (local_id == 0) {
        work[thread::get_block_id()] = tmp_work[0];
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void finalize_dot_computation(
    size_type size, const ValueType *work, ValueType *result)
{
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();

    ValueType tmp = zero<ValueType>();
    for (auto i = local_id; i < size; i += block_size) {
        tmp += work[i];
    }
    __shared__ UninitializedArray<ValueType, block_size> tmp_work;
    tmp_work[local_id] = tmp;

    reduce(group::this_thread_block(), static_cast<ValueType *>(tmp_work),
           [](const ValueType &x, const ValueType &y) { return x + y; });

    if (local_id == 0) {
        *result = tmp_work[0];
    }
}


}  // namespace kernel


template <typename ValueType>
void compute_dot(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    if (cublas::is_supported<ValueType>::value) {
        // TODO: write a custom kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            cublas::dot(exec->get_cublas_handle(), x->get_size()[0],
                        x->get_const_values() + col, x->get_stride(),
                        y->get_const_values() + col, y->get_stride(),
                        result->get_values() + col);
        }
    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr auto work_per_thread = 32;
        constexpr auto block_size = 1024;

        constexpr auto work_per_block = work_per_thread * block_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{cuda_config::warp_size, 1,
                             block_size / cuda_config::warp_size};
        Array<ValueType> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            kernel::compute_partial_dot<block_size><<<grid_dim, block_dim>>>(
                x->get_size()[0], as_cuda_type(x->get_const_values() + col),
                x->get_stride(), as_cuda_type(y->get_const_values() + col),
                y->get_stride(), as_cuda_type(work.get_data()));
            kernel::finalize_dot_computation<block_size><<<1, block_dim>>>(
                grid_dim.x, as_cuda_type(work.get_const_data()),
                as_cuda_type(result->get_values() + col));
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


namespace kernel {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void compute_sqrt(
    size_type num_cols, ValueType *__restrict__ work)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tidx < num_cols) {
        work[tidx] = sqrt(abs(work[tidx]));
    }
}


}  // namespace kernel


template <typename ValueType>
void compute_norm2(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<ValueType> *result)
{
    if (cublas::is_supported<ValueType>::value) {
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            cublas::norm2(exec->get_cublas_handle(), x->get_size()[0],
                          x->get_const_values() + col, x->get_stride(),
                          result->get_values() + col);
        }
    } else {
        compute_dot(exec, x, x, result);
        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(result->get_size()[1], block_size.x), 1,
                             1);
        kernel::compute_sqrt<<<grid_size, block_size, 0, 0>>>(
            result->get_size()[1], as_cuda_type(result->get_values()));
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);

namespace kernel {

__global__ __launch_bounds__(cuda_config::warp_size) void start_nnz_sum(
    size_type num_rows, size_type work_size,
    size_type *__restrict__ nnz_per_row, size_type *__restrict__ add_values)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        auto start_row = tidx * work_size;
        auto nnz_up_to_here = nnz_per_row[start_row];
        for (auto i = start_row + 1; i < min(start_row + work_size, num_rows);
             i++) {
            nnz_up_to_here += nnz_per_row[i];
            nnz_per_row[i] = nnz_up_to_here;
        }
        add_values[tidx] = nnz_up_to_here;
    }
}

__global__ __launch_bounds__(default_block_size) void finalize_nnz_sum(
    size_type num_rows, size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ add_values)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx + blockDim.x < num_rows) {
        size_type add_value = 0;
        for (auto i = 0; i <= blockIdx.x; i++) {
            add_value += add_values[i];
        }
        nnz_per_row[tidx + blockDim.x] += add_value;
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_coo(
    size_type num_rows, size_type num_cols, size_type stride,
    const size_type *__restrict__ row_ptrs,
    const ValueType *__restrict__ source, IndexType *__restrict__ row_idxs,
    IndexType *__restrict__ col_idxs, ValueType *__restrict__ values)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        size_type write_to = (tidx == 0) ? 0 : row_ptrs[tidx - 1];

        for (auto i = 0; i < num_cols; i++) {
            if (source[stride * tidx + i] != zero<ValueType>()) {
                values[write_to] = source[stride * tidx + i];
                col_idxs[write_to] = i;
                row_idxs[write_to] = tidx;
                write_to++;
            }
        }
    }
}

}  // namespace kernel

template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    auto nnz_per_row = Array<size_type>(exec);
    nnz_per_row.resize_and_reset(num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    const size_type grid_dim = ceildiv(num_rows, default_block_size);
    auto add_values = Array<size_type>(exec);
    add_values.resize_and_reset(grid_dim);

    kernel::start_nnz_sum<<<ceildiv(grid_dim, cuda_config::warp_size),
                            cuda_config::warp_size>>>(
        num_rows, default_block_size, as_cuda_type(nnz_per_row.get_data()),
        as_cuda_type(add_values.get_data()));

    kernel::finalize_nnz_sum<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(nnz_per_row.get_data()),
        as_cuda_type(add_values.get_data()));

    kernel::fill_in_coo<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(source->get_const_values()), as_cuda_type(row_idxs),
        as_cuda_type(col_idxs), as_cuda_type(values));

    nnz_per_row.clear();
    add_values.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);

namespace kernel {

__global__ __launch_bounds__(cuda_config::warp_size) void start_row_ptr_sum(
    size_type num_rows, size_type work_size,
    size_type *__restrict__ nnz_per_row)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        auto start_row = tidx * work_size;
        auto nnz_up_to_here = nnz_per_row[start_row];
        for (auto i = start_row + 1; i < min(start_row + work_size, num_rows);
             i++) {
            nnz_up_to_here += nnz_per_row[i];
            nnz_per_row[i] = nnz_up_to_here;
        }
    }
}

template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_row_ptrs(
    size_type num_rows, size_type *__restrict__ nnz_per_row,
    IndexType *__restrict__ row_ptrs)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx == 0) {
        row_ptrs[0] = 0;
    }

    if (tidx < num_rows) {
        IndexType result_entry = nnz_per_row[tidx];
        for (auto i = 0; i < blockIdx.x; i++) {
            result_entry += nnz_per_row[(i + 1) * blockDim.x - 1];
        }
        row_ptrs[tidx + 1] = result_entry;
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ source, IndexType *__restrict__ row_ptrs,
    IndexType *__restrict__ col_idxs, ValueType *__restrict__ values)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        auto write_to = row_ptrs[tidx];
        for (auto i = 0; i < num_cols; i++) {
            if (source[stride * tidx + i] != zero<ValueType>()) {
                values[write_to] = source[stride * tidx + i];
                col_idxs[write_to] = i;
                write_to++;
            }
        }
    }
}

}  // namespace kernel

template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    auto nnz_per_row = Array<size_type>(exec);
    nnz_per_row.resize_and_reset(num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    kernel::start_row_ptr_sum<<<ceildiv(grid_dim, cuda_config::warp_size),
                                cuda_config::warp_size>>>(
        num_rows, default_block_size, as_cuda_type(nnz_per_row.get_data()));

    kernel::fill_row_ptrs<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(nnz_per_row.get_data()), as_cuda_type(row_ptrs));

    kernel::fill_in_csr<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(source->get_const_values()),
        as_cuda_type(row_ptrs), as_cuda_type(col_idxs), as_cuda_type(values));

    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_csr(std::shared_ptr<const CudaExecutor> exec,
                 matrix::Csr<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL);

namespace kernel {

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_ell(
    size_type num_rows, size_type num_cols, size_type source_stride,
    const ValueType *__restrict__ source, size_type max_nnz_per_row,
    size_type result_stride, IndexType *__restrict__ col_ptrs,
    ValueType *__restrict__ values)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        IndexType col_idx = 0;
        for (auto col = 0; col < num_cols; col++) {
            if (source[tidx * source_stride + col] != zero<ValueType>()) {
                col_ptrs[col_idx * result_stride + tidx] = col;
                values[col_idx * result_stride + tidx] =
                    source[tidx * source_stride + col];
                col_idx++;
            }
        }
        for (auto j = col_idx; j < max_nnz_per_row; j++) {
            col_ptrs[j * result_stride + tidx] = 0;
            values[j * result_stride + tidx] = zero<ValueType>();
        }
    } else if (tidx < result_stride) {
        for (auto j = 0; j < max_nnz_per_row; j++) {
            col_ptrs[j * result_stride + tidx] = 0;
            values[j * result_stride + tidx] = zero<ValueType>();
        }
    }
}

}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Ell<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();

    auto col_ptrs = result->get_col_idxs();
    auto values = result->get_values();

    auto source_stride = source->get_stride();
    auto result_stride = result->get_stride();

    auto grid_dim = ceildiv(result_stride, default_block_size);
    kernel::fill_in_ell<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, source_stride,
        as_cuda_type(source->get_const_values()), max_nnz_per_row,
        result_stride, as_cuda_type(col_ptrs), as_cuda_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_ell(std::shared_ptr<const CudaExecutor> exec,
                 matrix::Ell<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const CudaExecutor> exec,
                       matrix::Hybrid<ValueType, IndexType> *result,
                       const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_hybrid(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Hybrid<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Sellp<ValueType, IndexType> *result,
                      const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_sellp(std::shared_ptr<const CudaExecutor> exec,
                   matrix::Sellp<ValueType, IndexType> *result,
                   const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_SELLP_KERNEL);

namespace kernel {

__global__ __launch_bounds__(default_block_size) void reduce_nnz(
    size_type size, const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ result)
{
    extern __shared__ size_type block_sum[];
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    auto thread_sum = 0;
    for (auto i = tidx; i < size; i += blockDim.x * gridDim.x) {
        thread_sum += nnz_per_row[i];
    }
    block_sum[threadIdx.x] = thread_sum;

    __syncthreads();

    auto start_step = 1;
    while (blockDim.x > 2 * start_step) {
        start_step *= 2;
    }

    for (auto i = start_step; i >= 1; i >>= 1) {
        if (threadIdx.x < i && threadIdx.x + i < blockDim.x) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_sum[0];
    }
}

}  // namespace kernel


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    const auto n = ceildiv(num_rows, default_block_size);
    const size_type grid_dim =
        (n <= default_block_size) ? n : default_block_size;

    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_nnz<<<grid_dim, default_block_size,
                         default_block_size * sizeof(size_type)>>>(
        num_rows, as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(block_results.get_data()));

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_nnz<<<1, default_block_size,
                         default_block_size * sizeof(size_type)>>>(
        grid_dim, as_cuda_type(block_results.get_const_data()),
        as_cuda_type(d_result.get_data()));

    exec->get_master()->copy_from(exec.get(), 1, d_result.get_const_data(),
                                  result);
    d_result.clear();
    block_results.clear();
    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);

namespace kernel {

__global__ __launch_bounds__(default_block_size) void reduce_max_nnz(
    size_type size, const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ result)
{
    extern __shared__ size_type block_max[];
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    size_type thread_max = 0;
    for (auto i = tidx; i < size; i += blockDim.x * gridDim.x) {
        thread_max = max(thread_max, nnz_per_row[i]);
    }
    block_max[threadIdx.x] = thread_max;

    __syncthreads();

    auto start_step = 1;
    while (blockDim.x > 2 * start_step) {
        start_step *= 2;
    }

    for (auto i = start_step; i >= 1; i >>= 1) {
        if (threadIdx.x < i && threadIdx.x + i < blockDim.x) {
            block_max[threadIdx.x] =
                max(block_max[threadIdx.x + i], block_max[threadIdx.x]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_max[0];
    }
}

}  // namespace kernel

template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const CudaExecutor> exec,
                               const matrix::Dense<ValueType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    const auto n = ceildiv(num_rows, default_block_size);
    const size_type grid_dim =
        (n <= default_block_size) ? n : default_block_size;

    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_max_nnz<<<grid_dim, default_block_size,
                             default_block_size * sizeof(size_type)>>>(
        num_rows, as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(block_results.get_data()));

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_max_nnz<<<1, default_block_size,
                             default_block_size * sizeof(size_type)>>>(
        grid_dim, as_cuda_type(block_results.get_const_data()),
        as_cuda_type(d_result.get_data()));

    exec->get_master()->copy_from(exec.get(), 1, d_result.get_const_data(),
                                  result);
    d_result.clear();
    block_results.clear();
    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);

namespace kernel {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_row(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ work, size_type *__restrict__ result)
{
    __shared__ size_type part_results[default_block_size];
    const auto warp_size = cuda_config::warp_size;
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto global_warpid = tidx / warp_size;
    const auto local_warpid = threadIdx.x / warp_size;

    if (global_warpid < num_rows) {
        part_results[threadIdx.x] = 0;
        for (auto i = threadIdx.x % warp_size; i < num_cols; i += warp_size) {
            if (work[stride * global_warpid + i] != zero<ValueType>()) {
                part_results[threadIdx.x] += 1;
            }
        }
        __syncthreads();
        if (threadIdx.x % warp_size == 0) {
            result[global_warpid] = 0;
            for (auto i = 0; i < warp_size; i++) {
                result[global_warpid] +=
                    part_results[local_warpid * warp_size + i];
            }
        }
    }
}


}  // namespace kernel

template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result)
{
    const dim3 block_size(default_block_size, 1, 1);
    auto rows_per_block = ceildiv(default_block_size, cuda_config::warp_size);
    const size_t grid_x = ceildiv(source->get_size()[0], rows_per_block);
    const dim3 grid_size(grid_x, 1, 1);
    kernel::count_nnz_per_row<<<grid_size, block_size>>>(
        source->get_size()[0], source->get_size()[1], source->get_stride(),
        as_cuda_type(source->get_const_values()),
        as_cuda_type(result->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::Dense<ValueType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig->get_size()[0],
                     orig->get_size()[1], &alpha, orig->get_const_values(),
                     orig->get_stride(), &beta,
                     static_cast<ValueType *>(nullptr), trans->get_size()[1],
                     trans->get_values(), trans->get_stride());

        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    } else {
        NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)

{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::geam(handle, CUBLAS_OP_C, CUBLAS_OP_N, orig->get_size()[0],
                     orig->get_size()[1], &alpha, orig->get_const_values(),
                     orig->get_stride(), &beta,
                     static_cast<ValueType *>(nullptr), trans->get_size()[1],
                     trans->get_values(), trans->get_stride());

        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    } else {
        NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
