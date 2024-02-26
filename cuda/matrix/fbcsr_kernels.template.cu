// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>


#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/base/array_access.hpp"
#include "core/base/block_sizes.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/cusparse_block_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * @brief The fixed-size block compressed sparse row matrix format namespace.
 *
 * @ingroup fbcsr
 */
namespace fbcsr {


constexpr int default_block_size{512};


#include "common/cuda_hip/matrix/csr_common.hpp.inc"
#include "common/cuda_hip/matrix/fbcsr_kernels.hpp.inc"


namespace {

template <typename ValueType>
void dense_transpose(std::shared_ptr<const CudaExecutor> exec,
                     const size_type nrows, const size_type ncols,
                     const size_type orig_stride, const ValueType* const orig,
                     const size_type trans_stride, ValueType* const trans)
{
    if (nrows == 0) {
        return;
    }
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nrows, ncols, &alpha,
                         orig, orig_stride, &beta, trans, trans_stride, trans,
                         trans_stride);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Fbcsr<ValueType, IndexType>* const a,
          const matrix::Dense<ValueType>* const b,
          matrix::Dense<ValueType>* const c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
        return;
    }
    if (b->get_size()[0] == 0 || a->get_num_stored_blocks() == 0) {
        // empty input: fill output with zero
        dense::fill(exec, c, zero<ValueType>());
        return;
    }
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);
        const auto alpha = one<ValueType>();
        const auto beta = zero<ValueType>();
        auto descr = cusparse::create_mat_descr();
        const auto row_ptrs = a->get_const_row_ptrs();
        const auto col_idxs = a->get_const_col_idxs();
        const auto values = a->get_const_values();
        const int bs = a->get_block_size();
        const IndexType mb = a->get_num_block_rows();
        const IndexType nb = a->get_num_block_cols();
        const auto nnzb = static_cast<IndexType>(a->get_num_stored_blocks());
        const auto nrhs = static_cast<IndexType>(b->get_size()[1]);
        const auto nrows = a->get_size()[0];
        const auto ncols = a->get_size()[1];
        const auto in_stride = b->get_stride();
        const auto out_stride = c->get_stride();
        if (nrhs == 1 && in_stride == 1 && out_stride == 1) {
            cusparse::bsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                            nnzb, &alpha, descr, values, row_ptrs, col_idxs, bs,
                            b->get_const_values(), &beta, c->get_values());
        } else {
            const auto trans_stride = nrows;
            auto trans_c = array<ValueType>(exec, nrows * nrhs);
            cusparse::bsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE, mb, nrhs, nb, nnzb,
                            &alpha, descr, values, row_ptrs, col_idxs, bs,
                            b->get_const_values(), in_stride, &beta,
                            trans_c.get_data(), trans_stride);
            dense_transpose(exec, nrhs, nrows, trans_stride, trans_c.get_data(),
                            out_stride, c->get_values());
        }
        cusparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType>* const alpha,
                   const matrix::Fbcsr<ValueType, IndexType>* const a,
                   const matrix::Dense<ValueType>* const b,
                   const matrix::Dense<ValueType>* const beta,
                   matrix::Dense<ValueType>* const c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
        return;
    }
    if (b->get_size()[0] == 0 || a->get_num_stored_blocks() == 0) {
        // empty input: scale output
        dense::scale(exec, beta, c);
        return;
    }
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        const auto alphp = alpha->get_const_values();
        const auto betap = beta->get_const_values();
        auto descr = cusparse::create_mat_descr();
        const auto row_ptrs = a->get_const_row_ptrs();
        const auto col_idxs = a->get_const_col_idxs();
        const auto values = a->get_const_values();
        const int bs = a->get_block_size();
        const IndexType mb = a->get_num_block_rows();
        const IndexType nb = a->get_num_block_cols();
        const auto nnzb = static_cast<IndexType>(a->get_num_stored_blocks());
        const auto nrhs = static_cast<IndexType>(b->get_size()[1]);
        const auto nrows = a->get_size()[0];
        const auto ncols = a->get_size()[1];
        const auto in_stride = b->get_stride();
        const auto out_stride = c->get_stride();
        if (nrhs == 1 && in_stride == 1 && out_stride == 1) {
            cusparse::bsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                            nnzb, alphp, descr, values, row_ptrs, col_idxs, bs,
                            b->get_const_values(), betap, c->get_values());
        } else {
            const auto trans_stride = nrows;
            auto trans_c = array<ValueType>(exec, nrows * nrhs);
            dense_transpose(exec, nrows, nrhs, out_stride, c->get_values(),
                            trans_stride, trans_c.get_data());
            cusparse::bsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE, mb, nrhs, nb, nnzb,
                            alphp, descr, values, row_ptrs, col_idxs, bs,
                            b->get_const_values(), in_stride, betap,
                            trans_c.get_data(), trans_stride);
            dense_transpose(exec, nrhs, nrows, trans_stride, trans_c.get_data(),
                            out_stride, c->get_values());
        }
        cusparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


namespace {


template <int mat_blk_sz, typename ValueType, typename IndexType>
void transpose_blocks_impl(syn::value_list<int, mat_blk_sz>,
                           std::shared_ptr<const DefaultExecutor> exec,
                           matrix::Fbcsr<ValueType, IndexType>* const mat)
{
    constexpr int subwarp_size = config::warp_size;
    const auto nbnz = mat->get_num_stored_blocks();
    const auto numthreads = nbnz * subwarp_size;
    const auto block_size = default_block_size;
    const auto grid_dim = ceildiv(numthreads, block_size);
    if (grid_dim > 0) {
        kernel::transpose_blocks<mat_blk_sz, subwarp_size>
            <<<grid_dim, block_size, 0, exec->get_stream()>>>(
                nbnz, mat->get_values());
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_transpose_blocks,
                                    transpose_blocks_impl);


}  // namespace


template <typename ValueType, typename IndexType>
void transpose(const std::shared_ptr<const CudaExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* const orig,
               matrix::Fbcsr<ValueType, IndexType>* const trans)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        const int bs = orig->get_block_size();
        const IndexType nnzb =
            static_cast<IndexType>(orig->get_num_stored_blocks());
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        const IndexType buffer_size = cusparse::bsr_transpose_buffersize(
            exec->get_cusparse_handle(), orig->get_num_block_rows(),
            orig->get_num_block_cols(), nnzb, orig->get_const_values(),
            orig->get_const_row_ptrs(), orig->get_const_col_idxs(), bs, bs);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        cusparse::bsr_transpose(
            exec->get_cusparse_handle(), orig->get_num_block_rows(),
            orig->get_num_block_cols(), nnzb, orig->get_const_values(),
            orig->get_const_row_ptrs(), orig->get_const_col_idxs(), bs, bs,
            trans->get_values(), trans->get_col_idxs(), trans->get_row_ptrs(),
            copyValues, idxBase, buffer);

        // transpose blocks
        select_transpose_blocks(
            fixedblock::compiled_kernels(),
            [bs](int compiled_block_size) { return bs == compiled_block_size; },
            syn::value_list<int>(), syn::type_list<>(), exec, trans);
    } else {
        fallback_transpose(exec, orig, trans);
    }
}


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* orig,
                    matrix::Fbcsr<ValueType, IndexType>* trans)
{
    const int grid_size =
        ceildiv(trans->get_num_stored_elements(), default_block_size);
    transpose(exec, orig, trans);
    if (grid_size > 0 && is_complex<ValueType>()) {
        kernel::
            conjugate<<<grid_size, default_block_size, 0, exec->get_stream()>>>(
                trans->get_num_stored_elements(),
                as_device_type(trans->get_values()));
    }
}


}  // namespace fbcsr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
