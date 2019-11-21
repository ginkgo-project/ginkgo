/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


constexpr auto default_block_size = 512;


#include "common/matrix/dense_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_size()[1],
                         c->get_size()[0], a->get_size()[1], &alpha,
                         b->get_const_values(), b->get_stride(),
                         a->get_const_values(), a->get_stride(), &beta,
                         c->get_values(), c->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
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
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


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
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        kernel::scale<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_values()), x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


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
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        kernel::add_scaled<block_size><<<grid_dim, block_dim>>>(
            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(x->get_const_values()), x->get_stride(),
            as_cuda_type(y->get_values()), y->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


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
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
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

    auto nnz_prefix_sum = Array<size_type>(exec, num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_prefix_sum);

    const size_type grid_dim = ceildiv(num_rows, default_block_size);
    auto add_values = Array<size_type>(exec, grid_dim);

    start_prefix_sum<default_block_size><<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(nnz_prefix_sum.get_data()),
        as_cuda_type(add_values.get_data()));

    finalize_prefix_sum<default_block_size><<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(nnz_prefix_sum.get_data()),
        as_cuda_type(add_values.get_data()));

    kernel::fill_in_coo<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, stride,
        as_cuda_type(nnz_prefix_sum.get_const_data()),
        as_cuda_type(source->get_const_values()), as_cuda_type(row_idxs),
        as_cuda_type(col_idxs), as_cuda_type(values));

    nnz_prefix_sum.clear();
    add_values.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


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

    const auto rows_per_block = ceildiv(default_block_size, config::warp_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    kernel::count_nnz_per_row<<<grid_dim_nnz, default_block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(source->get_const_values()),
        as_cuda_type(row_ptrs));

    size_type grid_dim = ceildiv(num_rows + 1, default_block_size);
    auto add_values = Array<IndexType>(exec, grid_dim);

    start_prefix_sum<default_block_size>
        <<<grid_dim, default_block_size>>>(num_rows + 1, as_cuda_type(row_ptrs),
                                           as_cuda_type(add_values.get_data()));

    finalize_prefix_sum<default_block_size><<<grid_dim, default_block_size>>>(
        num_rows + 1, as_cuda_type(row_ptrs),
        as_cuda_type(add_values.get_const_data()));

    kernel::fill_in_csr<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(source->get_const_values()),
        as_cuda_type(row_ptrs), as_cuda_type(col_idxs), as_cuda_type(values));

    add_values.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


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
void convert_to_hybrid(std::shared_ptr<const CudaExecutor> exec,
                       matrix::Hybrid<ValueType, IndexType> *result,
                       const matrix::Dense<ValueType> *source)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Sellp<ValueType, IndexType> *result,
                      const matrix::Dense<ValueType> *source)
{
    const auto stride = source->get_stride();
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();

    const auto slice_size = (result->get_slice_size() == 0)
                                ? matrix::default_slice_size
                                : result->get_slice_size();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? matrix::default_stride_factor
                                   : result->get_stride_factor();
    const int slice_num = ceildiv(num_rows, slice_size);

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    auto grid_dim = slice_num;

    kernel::calculate_slice_lengths<<<grid_dim, config::warp_size>>>(
        num_rows, slice_size, slice_num, stride_factor,
        as_cuda_type(nnz_per_row.get_const_data()), as_cuda_type(slice_lengths),
        as_cuda_type(slice_sets));

    auto add_values =
        Array<size_type>(exec, ceildiv(slice_num + 1, default_block_size));
    grid_dim = ceildiv(slice_num + 1, default_block_size);

    start_prefix_sum<default_block_size><<<grid_dim, default_block_size>>>(
        slice_num + 1, as_cuda_type(slice_sets),
        as_cuda_type(add_values.get_data()));

    finalize_prefix_sum<default_block_size><<<grid_dim, default_block_size>>>(
        slice_num + 1, as_cuda_type(slice_sets),
        as_cuda_type(add_values.get_const_data()));

    grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_sellp<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, slice_size, stride,
        as_cuda_type(source->get_const_values()), as_cuda_type(slice_lengths),
        as_cuda_type(slice_sets), as_cuda_type(col_idxs), as_cuda_type(vals));

    add_values.clear();
    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const CudaExecutor> exec,
                             matrix::SparsityCsr<ValueType, IndexType> *result,
                             const matrix::Dense<ValueType> *source)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


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

    exec->get_master()->get_mem_space()->copy_from(
        exec->get_mem_space().get(), 1, d_result.get_const_data(), result);
    d_result.clear();
    block_results.clear();
    nnz_per_row.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result)
{
    const dim3 block_size(default_block_size, 1, 1);
    auto rows_per_block = ceildiv(default_block_size, config::warp_size);
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
                          size_type slice_size)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto slice_num = ceildiv(num_rows, slice_size);

    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);

    auto grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);

    kernel::reduce_max_nnz_per_slice<<<grid_dim, default_block_size>>>(
        num_rows, slice_size, stride_factor,
        as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(max_nnz_per_slice.get_data()));

    grid_dim = ceildiv(slice_num, default_block_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_total_cols<<<grid_dim, default_block_size,
                                default_block_size * sizeof(size_type)>>>(
        slice_num, as_cuda_type(max_nnz_per_slice.get_const_data()),
        as_cuda_type(block_results.get_data()));

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_total_cols<<<1, default_block_size,
                                default_block_size * sizeof(size_type)>>>(
        grid_dim, as_cuda_type(block_results.get_const_data()),
        as_cuda_type(d_result.get_data()));

    exec->get_master()->get_mem_space()->copy_from(
        exec->get_mem_space().get(), 1, d_result.get_const_data(), result);

    block_results.clear();
    nnz_per_row.clear();
    max_nnz_per_slice.clear();
    d_result.clear();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::geam(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, orig->get_size()[0],
                orig->get_size()[1], &alpha, orig->get_const_values(),
                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
                trans->get_size()[1], trans->get_values(), trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
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
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::geam(
                handle, CUBLAS_OP_C, CUBLAS_OP_N, orig->get_size()[0],
                orig->get_size()[1], &alpha, orig->get_const_values(),
                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
                trans->get_size()[1], trans->get_values(), trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const CudaExecutor> exec,
                 const Array<IndexType> *permutation_indices,
                 matrix::Dense<ValueType> *row_permuted,
                 const matrix::Dense<ValueType> *orig) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const CudaExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    matrix::Dense<ValueType> *column_permuted,
                    const matrix::Dense<ValueType> *orig) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const CudaExecutor> exec,
                         const Array<IndexType> *permutation_indices,
                         matrix::Dense<ValueType> *row_permuted,
                         const matrix::Dense<ValueType> *orig)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const CudaExecutor> exec,
                            const Array<IndexType> *permutation_indices,
                            matrix::Dense<ValueType> *column_permuted,
                            const matrix::Dense<ValueType> *orig)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_INVERSE_COLUMN_PERMUTE_KERNEL);


}  // namespace dense
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
