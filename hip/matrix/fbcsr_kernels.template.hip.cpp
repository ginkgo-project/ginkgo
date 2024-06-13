// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include "core/base/array_access.hpp"
#include "core/base/block_sizes.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/hipsparse_block_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/thrust.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
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


template <typename ValueType, typename IndexType>
void transpose(const std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* const input,
               matrix::Fbcsr<ValueType, IndexType>* const output)
{
    fallback_transpose(exec, input, output);
}


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
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
}  // namespace hip
}  // namespace kernels
}  // namespace gko
