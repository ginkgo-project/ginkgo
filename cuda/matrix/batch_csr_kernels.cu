/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {


constexpr int default_block_size = 512;
constexpr int sm_multiplier = 4;
// constexpr int warps_in_block = 4;
// constexpr int spmv_block_size = warps_in_block * config::warp_size;
// constexpr int wsize = config::warp_size;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/matrix/batch_csr_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType> *const a,
          const matrix::BatchDense<ValueType> *const b,
          matrix::BatchDense<ValueType> *const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    spmv<<<num_blocks, default_block_size>>>(a_ub, b_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::BatchDense<ValueType> *const alpha,
                   const matrix::BatchCsr<ValueType, IndexType> *const a,
                   const matrix::BatchDense<ValueType> *const b,
                   const matrix::BatchDense<ValueType> *const beta,
                   matrix::BatchDense<ValueType> *const c)
{
    const auto num_blocks = exec->get_num_multiprocessor() * sm_multiplier;
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto c_ub = get_batch_struct(c);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    advanced_spmv<<<num_blocks, default_block_size>>>(alpha_ub, a_ub, b_ub,
                                                      beta_ub, c_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto grid_dim = ceildiv(num_rows, default_block_size);
//
//    kernel::convert_row_ptrs_to_idxs<<<grid_dim, default_block_size>>>(
//        num_rows, as_cuda_type(ptrs), as_cuda_type(idxs));
//}


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType> *source,
                      matrix::BatchDense<ValueType> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto num_rows = result->get_size()[0];
//    const auto num_cols = result->get_size()[1];
//    const auto stride = result->get_stride();
//    const auto row_ptrs = source->get_const_row_ptrs();
//    const auto col_idxs = source->get_const_col_idxs();
//    const auto vals = source->get_const_values();
//
//    const dim3 block_size(config::warp_size,
//                          config::max_block_size / config::warp_size, 1);
//    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
//                             ceildiv(num_rows, block_size.y), 1);
//    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
//        num_rows, num_cols, stride, as_cuda_type(result->get_values()));
//
//    auto grid_dim = ceildiv(num_rows, default_block_size);
//    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
//        num_rows, as_cuda_type(row_ptrs), as_cuda_type(col_idxs),
//        as_cuda_type(vals), stride, as_cuda_type(result->get_values()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//
//    if (num_rows == 0) {
//        *result = 0;
//        return;
//    }
//
//    const auto slice_num = ceildiv(num_rows, slice_size);
//    const auto row_ptrs = source->get_const_row_ptrs();
//
//    auto nnz_per_row = Array<size_type>(exec, num_rows);
//    auto grid_dim = ceildiv(num_rows, default_block_size);
//
//    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
//        num_rows, as_cuda_type(row_ptrs),
//        as_cuda_type(nnz_per_row.get_data()));
//
//    grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);
//    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);
//
//    kernel::reduce_max_nnz_per_slice<<<grid_dim, default_block_size>>>(
//        num_rows, slice_size, stride_factor,
//        as_cuda_type(nnz_per_row.get_const_data()),
//        as_cuda_type(max_nnz_per_slice.get_data()));
//
//    grid_dim = ceildiv(slice_num, default_block_size);
//    auto block_results = Array<size_type>(exec, grid_dim);
//
//    kernel::reduce_total_cols<<<grid_dim, default_block_size>>>(
//        slice_num, as_cuda_type(max_nnz_per_slice.get_const_data()),
//        as_cuda_type(block_results.get_data()));
//
//    auto d_result = Array<size_type>(exec, 1);
//
//    kernel::reduce_total_cols<<<1, default_block_size>>>(
//        grid_dim, as_cuda_type(block_results.get_const_data()),
//        as_cuda_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType> *orig,
               matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (cusparse::is_supported<ValueType, IndexType>::value) {
//#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
//        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
//        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
//
//        cusparse::transpose(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues,
//            idxBase);
//#else  // CUDA_VERSION >= 11000
//        cudaDataType_t cu_value =
//            gko::kernels::cuda::cuda_data_type<ValueType>();
//        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
//        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
//        cusparseBatchCsr2CscAlg_t alg = CUSPARSE_BATCH_CSR2CSC_ALG1;
//        size_type buffer_size = 0;
//        cusparse::transpose_buffersize(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value,
//            copyValues, idxBase, alg, &buffer_size);
//        Array<char> buffer_array(exec, buffer_size);
//        auto buffer = buffer_array.get_data();
//        cusparse::transpose(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value,
//            copyValues, idxBase, alg, buffer);
//#endif
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType> *orig,
                    matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (cusparse::is_supported<ValueType, IndexType>::value) {
//        const dim3 block_size(default_block_size, 1, 1);
//        const dim3 grid_size(
//            ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);
//
//#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
//        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
//        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
//
//        cusparse::transpose(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues,
//            idxBase);
//#else  // CUDA_VERSION >= 11000
//        cudaDataType_t cu_value =
//            gko::kernels::cuda::cuda_data_type<ValueType>();
//        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
//        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
//        cusparseBatchCsr2CscAlg_t alg = CUSPARSE_BATCH_CSR2CSC_ALG1;
//        size_type buffer_size = 0;
//        cusparse::transpose_buffersize(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value,
//            copyValues, idxBase, alg, &buffer_size);
//        Array<char> buffer_array(exec, buffer_size);
//        auto buffer = buffer_array.get_data();
//        cusparse::transpose(
//            exec->get_cusparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value,
//            copyValues, idxBase, alg, buffer);
//#endif
//
//        conjugate_kernel<<<grid_size, block_size, 0, 0>>>(
//            trans->get_num_stored_elements(),
//            as_cuda_type(trans->get_values()));
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    size_type *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//
//    auto nnz_per_row = Array<size_type>(exec, num_rows);
//    auto block_results = Array<size_type>(exec, default_block_size);
//    auto d_result = Array<size_type>(exec, 1);
//
//    const auto grid_dim = ceildiv(num_rows, default_block_size);
//    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
//        num_rows, as_cuda_type(source->get_const_row_ptrs()),
//        as_cuda_type(nnz_per_row.get_data()));
//
//    const auto n = ceildiv(num_rows, default_block_size);
//    const auto reduce_dim = n <= default_block_size ? n : default_block_size;
//    kernel::reduce_max_nnz<<<reduce_dim, default_block_size>>>(
//        num_rows, as_cuda_type(nnz_per_row.get_const_data()),
//        as_cuda_type(block_results.get_data()));
//
//    kernel::reduce_max_nnz<<<1, default_block_size>>>(
//        reduce_dim, as_cuda_type(block_results.get_const_data()),
//        as_cuda_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    Array<size_type> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//    auto row_ptrs = source->get_const_row_ptrs();
//    auto grid_dim = ceildiv(num_rows, default_block_size);
//
//    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
//        num_rows, as_cuda_type(row_ptrs), as_cuda_type(result->get_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const CudaExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (cusparse::is_supported<ValueType, IndexType>::value) {
//        auto handle = exec->get_cusparse_handle();
//        auto descr = cusparse::create_mat_descr();
//        auto m = IndexType(to_sort->get_size()[0]);
//        auto n = IndexType(to_sort->get_size()[1]);
//        auto nnz = IndexType(to_sort->get_num_stored_elements());
//        auto row_ptrs = to_sort->get_const_row_ptrs();
//        auto col_idxs = to_sort->get_col_idxs();
//        auto vals = to_sort->get_values();
//
//        // copy values
//        Array<ValueType> tmp_vals_array(exec, nnz);
//        exec->copy(nnz, vals, tmp_vals_array.get_data());
//        auto tmp_vals = tmp_vals_array.get_const_data();
//
//        // init identity permutation
//        Array<IndexType> permutation_array(exec, nnz);
//        auto permutation = permutation_array.get_data();
//        cusparse::create_identity_permutation(handle, nnz, permutation);
//
//        // allocate buffer
//        size_type buffer_size{};
//        cusparse::batch_csrsort_buffer_size(handle, m, n, nnz, row_ptrs,
//        col_idxs,
//                                      buffer_size);
//        Array<char> buffer_array{exec, buffer_size};
//        auto buffer = buffer_array.get_data();
//
//        // sort column indices
//        cusparse::batch_csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
//                          permutation, buffer);
//
//        // sort values
//#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
//        cusparse::gather(handle, nnz, tmp_vals, vals, permutation);
//#else  // CUDA_VERSION >= 11000
//        auto val_vec = cusparse::create_spvec(nnz, nnz, permutation, vals);
//        auto tmp_vec =
//            cusparse::create_dnvec(nnz, const_cast<ValueType *>(tmp_vals));
//        cusparse::gather(handle, tmp_vec, val_vec);
//#endif
//
//        cusparse::destroy(descr);
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *to_check,
    bool *is_sorted) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    *is_sorted = true;
//    auto cpu_array = Array<bool>::view(exec->get_master(), 1, is_sorted);
//    auto gpu_array = Array<bool>{exec, cpu_array};
//    auto block_size = default_block_size;
//    auto num_rows = static_cast<IndexType>(to_check->get_size()[0]);
//    auto num_blocks = ceildiv(num_rows, block_size);
//    kernel::check_unsorted<<<num_blocks, block_size>>>(
//        to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
//        num_rows, gpu_array.get_data());
//    cpu_array = gpu_array;
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace batch_csr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
