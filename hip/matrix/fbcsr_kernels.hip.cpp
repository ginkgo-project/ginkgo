/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>


#include <hip/hip_runtime.h>
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
#include "core/base/block_sizes.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/hipsparse_block_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {


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
void dense_transpose(std::shared_ptr<const HipExecutor> exec,
                     const size_type nrows, const size_type ncols,
                     const size_type orig_stride, const ValueType* const orig,
                     const size_type trans_stride, ValueType* const trans)
{
    if (nrows == 0) {
        return;
    }
    if (hipblas::is_supported<ValueType>::value) {
        auto handle = exec->get_hipblas_handle();
        {
            hipblas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            hipblas::geam(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, nrows, ncols,
                          &alpha, orig, orig_stride, &beta, trans, trans_stride,
                          trans, trans_stride);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
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
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_hipsparse_handle();
        hipsparse::pointer_mode_guard pm_guard(handle);
        const auto alpha = one<ValueType>();
        const auto beta = zero<ValueType>();
        auto descr = hipsparse::create_mat_descr();
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
            hipsparse::bsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                             nnzb, &alpha, descr, values, row_ptrs, col_idxs,
                             bs, b->get_const_values(), &beta, c->get_values());
        } else {
            const auto trans_stride = nrows;
            auto trans_c = array<ValueType>(exec, nrows * nrhs);
            hipsparse::bsrmm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             HIPSPARSE_OPERATION_TRANSPOSE, mb, nrhs, nb, nnzb,
                             &alpha, descr, values, row_ptrs, col_idxs, bs,
                             b->get_const_values(), in_stride, &beta,
                             trans_c.get_data(), trans_stride);
            dense_transpose(exec, nrhs, nrows, trans_stride, trans_c.get_data(),
                            out_stride, c->get_values());
        }
        hipsparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
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
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_hipsparse_handle();
        const auto alphp = alpha->get_const_values();
        const auto betap = beta->get_const_values();
        auto descr = hipsparse::create_mat_descr();
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
            hipsparse::bsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, mb, nb,
                             nnzb, alphp, descr, values, row_ptrs, col_idxs, bs,
                             b->get_const_values(), betap, c->get_values());
        } else {
            const auto trans_stride = nrows;
            auto trans_c = array<ValueType>(exec, nrows * nrhs);
            dense_transpose(exec, nrows, nrhs, out_stride, c->get_values(),
                            trans_stride, trans_c.get_data());
            hipsparse::bsrmm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             HIPSPARSE_OPERATION_TRANSPOSE, mb, nrhs, nb, nnzb,
                             alphp, descr, values, row_ptrs, col_idxs, bs,
                             b->get_const_values(), in_stride, betap,
                             trans_c.get_data(), trans_stride);
            dense_transpose(exec, nrhs, nrows, trans_stride, trans_c.get_data(),
                            out_stride, c->get_values());
        }
        hipsparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Fbcsr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    constexpr auto warps_per_block = default_block_size / config::warp_size;
    const auto num_blocks =
        ceildiv(source->get_num_block_rows(), warps_per_block);
    if (num_blocks > 0) {
        kernel::fill_in_dense<<<num_blocks, default_block_size>>>(
            source->get_const_row_ptrs(), source->get_const_col_idxs(),
            as_hip_type(source->get_const_values()),
            as_hip_type(result->get_values()), result->get_stride(),
            source->get_num_block_rows(), source->get_block_size());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const HipExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* const source,
                    matrix::Csr<ValueType, IndexType>* const result)
{
    constexpr auto warps_per_block = default_block_size / config::warp_size;
    const auto num_blocks =
        ceildiv(source->get_num_block_rows(), warps_per_block);
    if (num_blocks > 0) {
        kernel::convert_to_csr<<<num_blocks, default_block_size>>>(
            source->get_const_row_ptrs(), source->get_const_col_idxs(),
            as_hip_type(source->get_const_values()), result->get_row_ptrs(),
            result->get_col_idxs(), as_hip_type(result->get_values()),
            source->get_num_block_rows(), source->get_block_size());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(const std::shared_ptr<const HipExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* const orig,
               matrix::Fbcsr<ValueType, IndexType>* const trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* orig,
                    matrix::Fbcsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType>* const to_check,
    bool* const is_sorted)
{
    *is_sorted = true;
    auto gpu_array = array<bool>(exec, 1);
    // need to initialize the GPU value to true
    exec->copy_from(exec->get_master().get(), 1, is_sorted,
                    gpu_array.get_data());
    auto block_size = default_block_size;
    const auto num_brows =
        static_cast<IndexType>(to_check->get_num_block_rows());
    const auto num_blocks = ceildiv(num_brows, block_size);
    if (num_blocks > 0) {
        kernel::check_unsorted<<<num_blocks, block_size>>>(
            to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
            num_brows, gpu_array.get_data());
    }
    *is_sorted = exec->copy_val_to_host(gpu_array.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void sort_by_column_index(const std::shared_ptr<const HipExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType>* const to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const HipExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
