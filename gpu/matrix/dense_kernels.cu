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

#include "core/matrix/dense_kernels.hpp"


#include "core/base/math.hpp"
#include "gpu/base/cublas_bindings.hpp"
#include "gpu/components/reduction.cuh"
#include "gpu/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const GpuExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_size().num_cols,
                     c->get_size().num_rows, a->get_size().num_cols, &alpha,
                     b->get_const_values(), b->get_stride(),
                     a->get_const_values(), a->get_stride(), &beta,
                     c->get_values(), c->get_stride());
        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const GpuExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_size().num_cols,
                     c->get_size().num_rows, a->get_size().num_cols,
                     alpha->get_const_values(), b->get_const_values(),
                     b->get_stride(), a->get_const_values(), a->get_stride(),
                     beta->get_const_values(), c->get_values(),
                     c->get_stride());
        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const GpuExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        if (alpha->get_size().num_cols == 1) {
            cublas::scal(handle, x->get_num_stored_elements(),
                         alpha->get_const_values(), x->get_values(), 1);
        } else {
            // TODO: write a custom kernel which does this more efficiently
            for (size_type col = 0; col < x->get_size().num_cols; ++col) {
                cublas::scal(handle, x->get_size().num_rows,
                             alpha->get_const_values() + col,
                             x->get_values() + col, x->get_stride());
            }
        }
        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const GpuExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        // TODO: write a custom kernel which does this more efficiently
        if (alpha->get_size().num_cols == 1) {
            // cannot write as single kernel call, x and y can have different
            // strides
            for (size_type col = 0; col < x->get_size().num_cols; ++col) {
                cublas::axpy(handle, x->get_size().num_rows,
                             alpha->get_const_values(),
                             x->get_const_values() + col, x->get_stride(),
                             y->get_values() + col, y->get_stride());
            }
        } else {
            for (size_type col = 0; col < x->get_size().num_cols; ++col) {
                cublas::axpy(handle, x->get_size().num_rows,
                             alpha->get_const_values() + col,
                             x->get_const_values() + col, x->get_stride(),
                             y->get_values() + col, y->get_stride());
            }
        }
        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
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
    const auto num_blocks = gridDim.x;
    const auto local_id = thread::get_local_thread_id<cuda_config::warp_size>();
    const auto global_id =
        thread::get_thread_id<cuda_config::warp_size,
                              block_size / cuda_config::warp_size>();

    auto tmp = zero<ValueType>();
    for (auto i = global_id; i < num_rows; i += block_size * num_blocks) {
        tmp += x[i * stride_x] * y[i * stride_y];
    }
    __shared__ UninitializedArray<ValueType, block_size> tmp_work;
    tmp_work[local_id] = tmp;

    block::reduce<block_size, cuda_config::warp_size>(
        static_cast<ValueType *>(tmp_work),
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

    block::reduce<block_size, cuda_config::warp_size>(
        static_cast<ValueType *>(tmp_work),
        [](const ValueType &x, const ValueType &y) { return x + y; });

    if (local_id == 0) {
        *result = tmp_work[0];
    }
}


}  // namespace kernel


template <typename ValueType>
void compute_dot(std::shared_ptr<const GpuExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        // TODO: write a custom kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size().num_cols; ++col) {
            cublas::dot(handle, x->get_size().num_rows,
                        x->get_const_values() + col, x->get_stride(),
                        y->get_const_values() + col, y->get_stride(),
                        result->get_values() + col);
        }
        cublas::destroy(handle);
    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr auto work_per_thread = 32;
        constexpr auto block_size = 1024;

        constexpr auto work_per_block = work_per_thread * block_size;
        const auto grid_size = ceildiv(x->get_size().num_rows, work_per_block);
        Array<ValueType> work(exec, grid_size);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size().num_cols; ++col) {
            kernel::compute_partial_dot<block_size><<<grid_size, block_size>>>(
                x->get_size().num_rows,
                as_cuda_type(x->get_const_values() + col), x->get_stride(),
                as_cuda_type(y->get_const_values() + col), y->get_stride(),
                as_cuda_type(work.get_data()));
            kernel::finalize_dot_computation<block_size><<<1, block_size>>>(
                grid_size, as_cuda_type(work.get_const_data()),
                as_cuda_type(result->get_values() + col));
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_csr(std::shared_ptr<const GpuExecutor> exec,
                 matrix::Csr<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Ell<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_ell(std::shared_ptr<const GpuExecutor> exec,
                 matrix::Ell<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_ELL_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const GpuExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    size_type *result) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nonzeros_per_row(std::shared_ptr<const GpuExecutor> exec,
                                    const matrix::Dense<ValueType> *source,
                                    size_type *result) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const GpuExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     orig->get_size().num_rows, orig->get_size().num_cols,
                     &alpha, orig->get_const_values(), orig->get_stride(),
                     &beta, static_cast<ValueType *>(nullptr),
                     trans->get_size().num_cols, trans->get_values(),
                     trans->get_stride());

        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)

{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = cublas::init();
        ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        auto alpha = one<ValueType>();
        auto beta = zero<ValueType>();
        cublas::geam(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                     orig->get_size().num_rows, orig->get_size().num_cols,
                     &alpha, orig->get_const_values(), orig->get_stride(),
                     &beta, static_cast<ValueType *>(nullptr),
                     trans->get_size().num_cols, trans->get_values(),
                     trans->get_stride());

        cublas::destroy(handle);
    } else {
        NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
