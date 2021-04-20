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


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int wsize = config::warp_size;
constexpr int classical_overweight = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


namespace host_kernel {


template <int items_per_thread, typename ValueType, typename IndexType>
void merge_path_spmv(
    syn::value_list<int, items_per_thread>,
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *a,
    const matrix::BatchDense<ValueType> *b, matrix::BatchDense<ValueType> *c,
    const matrix::BatchDense<ValueType> *alpha = nullptr,
    const matrix::BatchDense<ValueType> *beta = nullptr) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
//    const IndexType grid_num =
//        ceildiv(total, spmv_block_size * items_per_thread);
//    const dim3 grid(grid_num);
//    const dim3 block(spmv_block_size);
//    Array<IndexType> row_out(exec, grid_num);
//    Array<ValueType> val_out(exec, grid_num);
//
//    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
//        if (alpha == nullptr && beta == nullptr) {
//            const auto b_vals = b->get_const_values() + column_id;
//            auto c_vals = c->get_values() + column_id;
//            hipLaunchKernelGGL(
//                HIP_KERNEL_NAME(
//                    kernel::abstract_merge_path_spmv<items_per_thread>),
//                dim3(grid), dim3(block), 0, 0,
//                static_cast<IndexType>(a->get_size()[0]),
//                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//                as_hip_type(a->get_const_row_ptrs()),
//                as_hip_type(a->get_const_srow()), as_hip_type(b_vals),
//                b->get_stride(), as_hip_type(c_vals), c->get_stride(),
//                as_hip_type(row_out.get_data()),
//                as_hip_type(val_out.get_data()));
//            hipLaunchKernelGGL(kernel::abstract_reduce, dim3(1),
//                               dim3(spmv_block_size), 0, 0, grid_num,
//                               as_hip_type(val_out.get_data()),
//                               as_hip_type(row_out.get_data()),
//                               as_hip_type(c_vals), c->get_stride());
//
//        } else if (alpha != nullptr && beta != nullptr) {
//            const auto b_vals = b->get_const_values() + column_id;
//            auto c_vals = c->get_values() + column_id;
//            hipLaunchKernelGGL(
//                HIP_KERNEL_NAME(
//                    kernel::abstract_merge_path_spmv<items_per_thread>),
//                dim3(grid), dim3(block), 0, 0,
//                static_cast<IndexType>(a->get_size()[0]),
//                as_hip_type(alpha->get_const_values()),
//                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//                as_hip_type(a->get_const_row_ptrs()),
//                as_hip_type(a->get_const_srow()), as_hip_type(b_vals),
//                b->get_stride(), as_hip_type(beta->get_const_values()),
//                as_hip_type(c_vals), c->get_stride(),
//                as_hip_type(row_out.get_data()),
//                as_hip_type(val_out.get_data()));
//            hipLaunchKernelGGL(kernel::abstract_reduce, dim3(1),
//                               dim3(spmv_block_size), 0, 0, grid_num,
//                               as_hip_type(val_out.get_data()),
//                               as_hip_type(row_out.get_data()),
//                               as_hip_type(alpha->get_const_values()),
//                               as_hip_type(c_vals), c->get_stride());
//        } else {
//            GKO_KERNEL_NOT_FOUND;
//        }
//    }
//}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const HipExecutor> exec)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//#if GINKGO_HIP_PLATFORM_NVCC
//
//
//    const int version =
//        (exec->get_major_version() << 4) + exec->get_minor_version();
//    // The num_item is decided to make the occupancy 100%
//    // TODO: Extend this list when new GPU is released
//    //       Tune this parameter
//    // 128 threads/block the number of items per threads
//    // 3.0 3.5: 6
//    // 3.7: 14
//    // 5.0, 5.3, 6.0, 6.2: 8
//    // 5.2, 6.1, 7.0: 12
//    int num_item = 6;
//    switch (version) {
//    case 0x50:
//    case 0x53:
//    case 0x60:
//    case 0x62:
//        num_item = 8;
//        break;
//    case 0x52:
//    case 0x61:
//    case 0x70:
//        num_item = 12;
//        break;
//    case 0x37:
//        num_item = 14;
//    }
//
//
//#else
//
//
//    // HIP uses the minimal num_item to make the code work correctly.
//    // TODO: this parameter should be tuned.
//    int num_item = 6;
//
//
//#endif  // GINKGO_HIP_PLATFORM_NVCC
//
//
//    // Ensure that the following is satisfied:
//    // sizeof(IndexType) + sizeof(ValueType)
//    // <= items_per_thread * sizeof(IndexType)
//    constexpr int minimal_num =
//        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
//    int items_per_thread = num_item * 4 / sizeof(IndexType);
//    return std::max(minimal_num, items_per_thread);
//}


template <int subwarp_size, typename ValueType, typename IndexType>
void classical_spmv(
    syn::value_list<int, subwarp_size>, std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *a,
    const matrix::BatchDense<ValueType> *b, matrix::BatchDense<ValueType> *c,
    const matrix::BatchDense<ValueType> *alpha = nullptr,
    const matrix::BatchDense<ValueType> *beta = nullptr) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto nwarps = exec->get_num_warps_per_sm() *
//                        exec->get_num_multiprocessor() * classical_overweight;
//    const auto gridx =
//        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
//                 int64(nwarps / warps_in_block));
//    const dim3 grid(gridx, b->get_size()[1]);
//    const dim3 block(spmv_block_size);
//
//    if (alpha == nullptr && beta == nullptr) {
//        hipLaunchKernelGGL(
//            HIP_KERNEL_NAME(kernel::abstract_classical_spmv<subwarp_size>),
//            dim3(grid), dim3(block), 0, 0, a->get_size()[0],
//            as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//            as_hip_type(a->get_const_row_ptrs()),
//            as_hip_type(b->get_const_values()), b->get_stride(),
//            as_hip_type(c->get_values()), c->get_stride());
//
//    } else if (alpha != nullptr && beta != nullptr) {
//        hipLaunchKernelGGL(
//            HIP_KERNEL_NAME(kernel::abstract_classical_spmv<subwarp_size>),
//            dim3(grid), dim3(block), 0, 0, a->get_size()[0],
//            as_hip_type(alpha->get_const_values()),
//            as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//            as_hip_type(a->get_const_row_ptrs()),
//            as_hip_type(b->get_const_values()), b->get_stride(),
//            as_hip_type(beta->get_const_values()),
//            as_hip_type(c->get_values()), c->get_stride());
//    } else {
//        GKO_KERNEL_NOT_FOUND;
//    }
//}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType> *a,
          const matrix::BatchDense<ValueType> *b,
          matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (a->get_strategy()->get_name() == "load_balance") {
//        components::fill_array(exec, c->get_values(),
//                               c->get_num_stored_elements(),
//                               zero<ValueType>());
//        const IndexType nwarps = a->get_num_srow_elements();
//        if (nwarps > 0) {
//            const dim3 batch_csr_block(config::warp_size, warps_in_block, 1);
//            const dim3 batch_csr_grid(ceildiv(nwarps, warps_in_block),
//                                b->get_size()[1]);
//            hipLaunchKernelGGL(
//                kernel::abstract_spmv, dim3(batch_csr_grid),
//                dim3(batch_csr_block), 0, 0, nwarps,
//                static_cast<IndexType>(a->get_size()[0]),
//                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//                as_hip_type(a->get_const_row_ptrs()),
//                as_hip_type(a->get_const_srow()),
//                as_hip_type(b->get_const_values()),
//                as_hip_type(b->get_stride()), as_hip_type(c->get_values()),
//                as_hip_type(c->get_stride()));
//        } else {
//            GKO_NOT_SUPPORTED(nwarps);
//        }
//    } else if (a->get_strategy()->get_name() == "merge_path") {
//        int items_per_thread =
//            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
//        host_kernel::select_merge_path_spmv(
//            compiled_kernels(),
//            [&items_per_thread](int compiled_info) {
//                return items_per_thread == compiled_info;
//            },
//            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
//    } else if (a->get_strategy()->get_name() == "classical") {
//        IndexType max_length_per_row = 0;
//        using Tbatch_csr = matrix::BatchCsr<ValueType, IndexType>;
//        if (auto strategy =
//                std::dynamic_pointer_cast<const typename
//                Tbatch_csr::classical>(
//                    a->get_strategy())) {
//            max_length_per_row = strategy->get_max_length_per_row();
//        } else if (auto strategy = std::dynamic_pointer_cast<
//                       const typename
//                       Tbatch_csr::automatical>(a->get_strategy())) {
//            max_length_per_row = strategy->get_max_length_per_row();
//        } else {
//            GKO_NOT_SUPPORTED(a->get_strategy());
//        }
//        host_kernel::select_classical_spmv(
//            classical_kernels(),
//            [&max_length_per_row](int compiled_info) {
//                return max_length_per_row >= compiled_info;
//            },
//            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
//    } else if (a->get_strategy()->get_name() == "sparselib" ||
//               a->get_strategy()->get_name() == "cusparse") {
//        if (hipsparse::is_supported<ValueType, IndexType>::value) {
//            // TODO: add implementation for int64 and multiple RHS
//            auto handle = exec->get_hipsparse_handle();
//            auto descr = hipsparse::create_mat_descr();
//            {
//                hipsparse::pointer_mode_guard pm_guard(handle);
//                auto row_ptrs = a->get_const_row_ptrs();
//                auto col_idxs = a->get_const_col_idxs();
//                auto alpha = one<ValueType>();
//                auto beta = zero<ValueType>();
//                if (b->get_stride() != 1 || c->get_stride() != 1) {
//                    GKO_NOT_IMPLEMENTED;
//                }
//                hipsparse::spmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
//                                a->get_size()[0], a->get_size()[1],
//                                a->get_num_stored_elements(), &alpha, descr,
//                                a->get_const_values(), row_ptrs, col_idxs,
//                                b->get_const_values(), &beta,
//                                c->get_values());
//            }
//            hipsparse::destroy(descr);
//        } else {
//            GKO_NOT_IMPLEMENTED;
//        }
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::BatchDense<ValueType> *alpha,
                   const matrix::BatchCsr<ValueType, IndexType> *a,
                   const matrix::BatchDense<ValueType> *b,
                   const matrix::BatchDense<ValueType> *beta,
                   matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (a->get_strategy()->get_name() == "load_balance") {
//        dense::scale(exec, beta, c);
//
//        const IndexType nwarps = a->get_num_srow_elements();
//
//        if (nwarps > 0) {
//            const dim3 batch_csr_block(config::warp_size, warps_in_block, 1);
//            const dim3 batch_csr_grid(ceildiv(nwarps, warps_in_block),
//                                b->get_size()[1]);
//            hipLaunchKernelGGL(
//                kernel::abstract_spmv, dim3(batch_csr_grid),
//                dim3(batch_csr_block), 0, 0, nwarps,
//                static_cast<IndexType>(a->get_size()[0]),
//                as_hip_type(alpha->get_const_values()),
//                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
//                as_hip_type(a->get_const_row_ptrs()),
//                as_hip_type(a->get_const_srow()),
//                as_hip_type(b->get_const_values()),
//                as_hip_type(b->get_stride()), as_hip_type(c->get_values()),
//                as_hip_type(c->get_stride()));
//        } else {
//            GKO_NOT_SUPPORTED(nwarps);
//        }
//    } else if (a->get_strategy()->get_name() == "sparselib" ||
//               a->get_strategy()->get_name() == "cusparse") {
//        if (hipsparse::is_supported<ValueType, IndexType>::value) {
//            // TODO: add implementation for int64 and multiple RHS
//            auto descr = hipsparse::create_mat_descr();
//
//            auto row_ptrs = a->get_const_row_ptrs();
//            auto col_idxs = a->get_const_col_idxs();
//
//            if (b->get_stride() != 1 || c->get_stride() != 1)
//                GKO_NOT_IMPLEMENTED;
//
//            hipsparse::spmv(exec->get_hipsparse_handle(),
//                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
//                            a->get_size()[0], a->get_size()[1],
//                            a->get_num_stored_elements(),
//                            alpha->get_const_values(), descr,
//                            a->get_const_values(), row_ptrs, col_idxs,
//                            b->get_const_values(), beta->get_const_values(),
//                            c->get_values());
//
//            hipsparse::destroy(descr);
//        } else {
//            GKO_NOT_IMPLEMENTED;
//        }
//    } else if (a->get_strategy()->get_name() == "classical") {
//        IndexType max_length_per_row = 0;
//        using Tbatch_csr = matrix::BatchCsr<ValueType, IndexType>;
//        if (auto strategy =
//                std::dynamic_pointer_cast<const typename
//                Tbatch_csr::classical>(
//                    a->get_strategy())) {
//            max_length_per_row = strategy->get_max_length_per_row();
//        } else if (auto strategy = std::dynamic_pointer_cast<
//                       const typename
//                       Tbatch_csr::automatical>(a->get_strategy())) {
//            max_length_per_row = strategy->get_max_length_per_row();
//        } else {
//            GKO_NOT_SUPPORTED(a->get_strategy());
//        }
//        host_kernel::select_classical_spmv(
//            classical_kernels(),
//            [&max_length_per_row](int compiled_info) {
//                return max_length_per_row >= compiled_info;
//            },
//            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
//            beta);
//    } else if (a->get_strategy()->get_name() == "merge_path") {
//        int items_per_thread =
//            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
//        host_kernel::select_merge_path_spmv(
//            compiled_kernels(),
//            [&items_per_thread](int compiled_info) {
//                return items_per_thread == compiled_info;
//            },
//            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
//            beta);
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto grid_dim = ceildiv(num_rows, default_block_size);
//
//    hipLaunchKernelGGL(kernel::convert_row_ptrs_to_idxs, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, num_rows,
//                       as_hip_type(ptrs), as_hip_type(idxs));
//}


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
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
//    hipLaunchKernelGGL(kernel::initialize_zero_dense, dim3(init_grid_dim),
//                       dim3(block_size), 0, 0, num_rows, num_cols, stride,
//                       as_hip_type(result->get_values()));
//
//    auto grid_dim = ceildiv(num_rows, default_block_size);
//    hipLaunchKernelGGL(
//        kernel::fill_in_dense, dim3(grid_dim), dim3(default_block_size), 0, 0,
//        num_rows, as_hip_type(row_ptrs), as_hip_type(col_idxs),
//        as_hip_type(vals), stride, as_hip_type(result->get_values()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const HipExecutor> exec,
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
//    hipLaunchKernelGGL(kernel::calculate_nnz_per_row, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, num_rows,
//                       as_hip_type(row_ptrs),
//                       as_hip_type(nnz_per_row.get_data()));
//
//    grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);
//    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);
//
//    hipLaunchKernelGGL(kernel::reduce_max_nnz_per_slice, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, num_rows, slice_size,
//                       stride_factor,
//                       as_hip_type(nnz_per_row.get_const_data()),
//                       as_hip_type(max_nnz_per_slice.get_data()));
//
//    grid_dim = ceildiv(slice_num, default_block_size);
//    auto block_results = Array<size_type>(exec, grid_dim);
//
//    hipLaunchKernelGGL(kernel::reduce_total_cols, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, slice_num,
//                       as_hip_type(max_nnz_per_slice.get_const_data()),
//                       as_hip_type(block_results.get_data()));
//
//    auto d_result = Array<size_type>(exec, 1);
//
//    hipLaunchKernelGGL(kernel::reduce_total_cols, dim3(1),
//                       dim3(default_block_size), 0, 0, grid_dim,
//                       as_hip_type(block_results.get_const_data()),
//                       as_hip_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType> *orig,
               matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (hipsparse::is_supported<ValueType, IndexType>::value) {
//        hipsparseAction_t copyValues = HIPSPARSE_ACTION_NUMERIC;
//        hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;
//
//        hipsparse::transpose(
//            exec->get_hipsparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues,
//            idxBase);
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType> *orig,
                    matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (hipsparse::is_supported<ValueType, IndexType>::value) {
//        const dim3 block_size(default_block_size, 1, 1);
//        const dim3 grid_size(
//            ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);
//
//        hipsparseAction_t copyValues = HIPSPARSE_ACTION_NUMERIC;
//        hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;
//
//        hipsparse::transpose(
//            exec->get_hipsparse_handle(), orig->get_size()[0],
//            orig->get_size()[1], orig->get_num_stored_elements(),
//            orig->get_const_values(), orig->get_const_row_ptrs(),
//            orig->get_const_col_idxs(), trans->get_values(),
//            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues,
//            idxBase);
//
//        hipLaunchKernelGGL(conjugate_kernel, dim3(grid_size),
//        dim3(block_size),
//                           0, 0, trans->get_num_stored_elements(),
//                           as_hip_type(trans->get_values()));
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const HipExecutor> exec,
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
//    hipLaunchKernelGGL(kernel::calculate_nnz_per_row, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, num_rows,
//                       as_hip_type(source->get_const_row_ptrs()),
//                       as_hip_type(nnz_per_row.get_data()));
//
//    const auto n = ceildiv(num_rows, default_block_size);
//    const auto reduce_dim = n <= default_block_size ? n : default_block_size;
//    hipLaunchKernelGGL(kernel::reduce_max_nnz, dim3(reduce_dim),
//                       dim3(default_block_size), 0, 0, num_rows,
//                       as_hip_type(nnz_per_row.get_const_data()),
//                       as_hip_type(block_results.get_data()));
//
//    hipLaunchKernelGGL(kernel::reduce_max_nnz, dim3(1),
//                       dim3(default_block_size), 0, 0, reduce_dim,
//                       as_hip_type(block_results.get_const_data()),
//                       as_hip_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    Array<size_type> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//    auto row_ptrs = source->get_const_row_ptrs();
//    auto grid_dim = ceildiv(num_rows, default_block_size);
//
//    hipLaunchKernelGGL(kernel::calculate_nnz_per_row, dim3(grid_dim),
//                       dim3(default_block_size), 0, 0, num_rows,
//                       as_hip_type(row_ptrs),
//                       as_hip_type(result->get_data()));
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const HipExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    if (hipsparse::is_supported<ValueType, IndexType>::value) {
//        auto handle = exec->get_hipsparse_handle();
//        auto descr = hipsparse::create_mat_descr();
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
//        hipsparse::create_identity_permutation(handle, nnz, permutation);
//
//        // allocate buffer
//        size_type buffer_size{};
//        hipsparse::batch_csrsort_buffer_size(handle, m, n, nnz, row_ptrs,
//        col_idxs,
//                                       buffer_size);
//        Array<char> buffer_array{exec, buffer_size};
//        auto buffer = buffer_array.get_data();
//
//        // sort column indices
//        hipsparse::batch_csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
//                           permutation, buffer);
//
//        // sort values
//        hipsparse::gather(handle, nnz, tmp_vals, vals, permutation);
//
//        hipsparse::destroy(descr);
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const HipExecutor> exec,
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
//    hipLaunchKernelGGL(
//        HIP_KERNEL_NAME(kernel::check_unsorted), dim3(num_blocks),
//        dim3(block_size), 0, 0, to_check->get_const_row_ptrs(),
//        to_check->get_const_col_idxs(), num_rows, gpu_array.get_data());
//    cpu_array = gpu_array;
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const HipExecutor> exec,
                 const matrix::BatchDense<ValueType> *left_scale,
                 const matrix::BatchDense<ValueType> *right_scale,
                 matrix::BatchCsr<ValueType, IndexType> *scaled)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BATCH_CSR_SCALE);


}  // namespace batch_csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
