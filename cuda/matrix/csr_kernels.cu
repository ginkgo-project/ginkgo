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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/dense_kernels.hpp"
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


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


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


#include "common/matrix/csr_kernels.hpp.inc"


namespace host_kernel {


template <int items_per_thread, typename ValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Dense<ValueType> *b,
                     matrix::Dense<ValueType> *c,
                     const matrix::Dense<ValueType> *alpha = nullptr,
                     const matrix::Dense<ValueType> *beta = nullptr)
{
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const dim3 grid(grid_num);
    const dim3 block(spmv_block_size);
    Array<IndexType> row_out(exec, grid_num);
    Array<ValueType> val_out(exec, grid_num);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        if (alpha == nullptr && beta == nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>
                <<<grid, block, 0, 0>>>(
                    static_cast<IndexType>(a->get_size()[0]),
                    as_cuda_type(a->get_const_values()),
                    a->get_const_col_idxs(),
                    as_cuda_type(a->get_const_row_ptrs()),
                    as_cuda_type(a->get_const_srow()), as_cuda_type(b_vals),
                    b->get_stride(), as_cuda_type(c_vals), c->get_stride(),
                    as_cuda_type(row_out.get_data()),
                    as_cuda_type(val_out.get_data()));
            kernel::abstract_reduce<<<1, spmv_block_size>>>(
                grid_num, as_cuda_type(val_out.get_data()),
                as_cuda_type(row_out.get_data()), as_cuda_type(c_vals),
                c->get_stride());

        } else if (alpha != nullptr && beta != nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>
                <<<grid, block, 0, 0>>>(
                    static_cast<IndexType>(a->get_size()[0]),
                    as_cuda_type(alpha->get_const_values()),
                    as_cuda_type(a->get_const_values()),
                    a->get_const_col_idxs(),
                    as_cuda_type(a->get_const_row_ptrs()),
                    as_cuda_type(a->get_const_srow()), as_cuda_type(b_vals),
                    b->get_stride(), as_cuda_type(beta->get_const_values()),
                    as_cuda_type(c_vals), c->get_stride(),
                    as_cuda_type(row_out.get_data()),
                    as_cuda_type(val_out.get_data()));
            kernel::abstract_reduce<<<1, spmv_block_size>>>(
                grid_num, as_cuda_type(val_out.get_data()),
                as_cuda_type(row_out.get_data()),
                as_cuda_type(alpha->get_const_values()), as_cuda_type(c_vals),
                c->get_stride());
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const CudaExecutor> exec)
{
    const int version =
        (exec->get_major_version() << 4) + exec->get_minor_version();
    // The num_item is decided to make the occupancy 100%
    // TODO: Extend this list when new GPU is released
    //       Tune this parameter
    // 128 threads/block the number of items per threads
    // 3.0 3.5: 6
    // 3.7: 14
    // 5.0, 5.3, 6.0, 6.2: 8
    // 5.2, 6.1, 7.0: 12
    int num_item = 6;
    switch (version) {
    case 0x50:
    case 0x53:
    case 0x60:
    case 0x62:
        num_item = 8;
        break;
    case 0x52:
    case 0x61:
    case 0x70:
        num_item = 12;
        break;
    case 0x37:
        num_item = 14;
    }
    // Ensure that the following is satisfied:
    // sizeof(IndexType) + sizeof(ValueType)
    // <= items_per_thread * sizeof(IndexType)
    constexpr int minimal_num =
        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
    int items_per_thread = num_item * 4 / sizeof(IndexType);
    return std::max(minimal_num, items_per_thread);
}


template <int subwarp_size, typename ValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c,
                    const matrix::Dense<ValueType> *alpha = nullptr,
                    const matrix::Dense<ValueType> *beta = nullptr)
{
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * classical_overweight;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());

    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        components::fill_array(exec, c->get_values(),
                               c->get_num_stored_elements(), zero<ValueType>());
        const IndexType nwarps = a->get_num_srow_elements();
        if (nwarps > 0) {
            const dim3 csr_block(config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv<<<csr_grid, csr_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(b->get_stride()), as_cuda_type(c->get_values()),
                as_cuda_type(c->get_stride()));
        } else {
            GKO_NOT_SUPPORTED(nwarps);
        }
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "classical") {
        IndexType max_length_per_row = 0;
        using Tcsr = matrix::Csr<ValueType, IndexType>;
        if (auto strategy =
                std::dynamic_pointer_cast<const typename Tcsr::classical>(
                    a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else if (auto strategy = std::dynamic_pointer_cast<
                       const typename Tcsr::automatical>(a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else {
            GKO_NOT_SUPPORTED(a->get_strategy());
        }
        host_kernel::select_classical_spmv(
            classical_kernels(),
            [&max_length_per_row](int compiled_info) {
                return max_length_per_row >= compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "sparselib" ||
               a->get_strategy()->get_name() == "cusparse") {
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            auto handle = exec->get_cusparse_handle();
            {
                cusparse::pointer_mode_guard pm_guard(handle);
                const auto alpha = one<ValueType>();
                const auto beta = zero<ValueType>();
                // TODO: add implementation for int64 and multiple RHS
                if (b->get_stride() != 1 || c->get_stride() != 1)
                    GKO_NOT_IMPLEMENTED;

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
                auto descr = cusparse::create_mat_descr();
                auto row_ptrs = a->get_const_row_ptrs();
                auto col_idxs = a->get_const_col_idxs();
                cusparse::spmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               a->get_size()[0], a->get_size()[1],
                               a->get_num_stored_elements(), &alpha, descr,
                               a->get_const_values(), row_ptrs, col_idxs,
                               b->get_const_values(), &beta, c->get_values());

                cusparse::destroy(descr);
#else  // CUDA_VERSION >= 11000
                cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
                cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG1;
                auto row_ptrs =
                    const_cast<IndexType *>(a->get_const_row_ptrs());
                auto col_idxs =
                    const_cast<IndexType *>(a->get_const_col_idxs());
                auto values = const_cast<ValueType *>(a->get_const_values());
                auto mat = cusparse::create_csr(
                    a->get_size()[0], a->get_size()[1],
                    a->get_num_stored_elements(), row_ptrs, col_idxs, values);
                auto b_val = const_cast<ValueType *>(b->get_const_values());
                auto c_val = c->get_values();
                auto vecb =
                    cusparse::create_dnvec(b->get_num_stored_elements(), b_val);
                auto vecc =
                    cusparse::create_dnvec(c->get_num_stored_elements(), c_val);
                size_type buffer_size = 0;
                cusparse::spmv_buffersize<ValueType>(handle, trans, &alpha, mat,
                                                     vecb, &beta, vecc, alg,
                                                     &buffer_size);

                gko::Array<char> buffer_array(exec, buffer_size);
                auto buffer = buffer_array.get_data();
                cusparse::spmv<ValueType>(handle, trans, &alpha, mat, vecb,
                                          &beta, vecc, alg, buffer);
                cusparse::destroy(vecb);
                cusparse::destroy(vecc);
                cusparse::destroy(mat);
#endif
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        dense::scale(exec, beta, c);

        const IndexType nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            const dim3 csr_block(config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv<<<csr_grid, csr_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(b->get_stride()), as_cuda_type(c->get_values()),
                as_cuda_type(c->get_stride()));
        } else {
            GKO_NOT_SUPPORTED(nwarps);
        }
    } else if (a->get_strategy()->get_name() == "sparselib" ||
               a->get_strategy()->get_name() == "cusparse") {
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            if (b->get_stride() != 1 || c->get_stride() != 1)
                GKO_NOT_IMPLEMENTED;

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
            auto descr = cusparse::create_mat_descr();
            auto row_ptrs = a->get_const_row_ptrs();
            auto col_idxs = a->get_const_col_idxs();
            cusparse::spmv(exec->get_cusparse_handle(),
                           CUSPARSE_OPERATION_NON_TRANSPOSE, a->get_size()[0],
                           a->get_size()[1], a->get_num_stored_elements(),
                           alpha->get_const_values(), descr,
                           a->get_const_values(), row_ptrs, col_idxs,
                           b->get_const_values(), beta->get_const_values(),
                           c->get_values());

            cusparse::destroy(descr);
#else  // CUDA_VERSION >= 11000
            cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG1;
            auto row_ptrs = const_cast<IndexType *>(a->get_const_row_ptrs());
            auto col_idxs = const_cast<IndexType *>(a->get_const_col_idxs());
            auto values = const_cast<ValueType *>(a->get_const_values());
            auto mat = cusparse::create_csr(a->get_size()[0], a->get_size()[1],
                                            a->get_num_stored_elements(),
                                            row_ptrs, col_idxs, values);
            auto b_val = const_cast<ValueType *>(b->get_const_values());
            auto c_val = c->get_values();
            auto vecb =
                cusparse::create_dnvec(b->get_num_stored_elements(), b_val);
            auto vecc =
                cusparse::create_dnvec(c->get_num_stored_elements(), c_val);
            size_type buffer_size = 0;
            cusparse::spmv_buffersize<ValueType>(
                exec->get_cusparse_handle(), trans, alpha->get_const_values(),
                mat, vecb, beta->get_const_values(), vecc, alg, &buffer_size);
            gko::Array<char> buffer_array(exec, buffer_size);
            auto buffer = buffer_array.get_data();
            cusparse::spmv<ValueType>(
                exec->get_cusparse_handle(), trans, alpha->get_const_values(),
                mat, vecb, beta->get_const_values(), vecc, alg, buffer);
            cusparse::destroy(vecb);
            cusparse::destroy(vecc);
            cusparse::destroy(mat);
#endif
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else if (a->get_strategy()->get_name() == "classical") {
        IndexType max_length_per_row = 0;
        using Tcsr = matrix::Csr<ValueType, IndexType>;
        if (auto strategy =
                std::dynamic_pointer_cast<const typename Tcsr::classical>(
                    a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else if (auto strategy = std::dynamic_pointer_cast<
                       const typename Tcsr::automatical>(a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else {
            GKO_NOT_SUPPORTED(a->get_strategy());
        }
        host_kernel::select_classical_spmv(
            classical_kernels(),
            [&max_length_per_row](int compiled_info) {
                return max_length_per_row >= compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto a_nnz = IndexType(a->get_num_stored_elements());
    auto a_vals = a->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto c_row_ptrs = c->get_row_ptrs();

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);

        auto alpha = one<ValueType>();
        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto null_value = static_cast<ValueType *>(nullptr);
        auto null_index = static_cast<IndexType *>(nullptr);
        auto zero_nnz = IndexType{};
        auto m = IndexType(a->get_size()[0]);
        auto n = IndexType(b->get_size()[1]);
        auto k = IndexType(a->get_size()[1]);
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto &c_col_idxs_array = c_builder.get_col_idx_array();
        auto &c_vals_array = c_builder.get_value_array();

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        auto a_descr = cusparse::create_mat_descr();
        auto b_descr = cusparse::create_mat_descr();
        auto c_descr = cusparse::create_mat_descr();
        auto d_descr = cusparse::create_mat_descr();
        auto info = cusparse::create_spgemm_info();
        // allocate buffer
        size_type buffer_size{};
        cusparse::spgemm_buffer_size(
            handle, m, n, k, &alpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            zero_nnz, null_index, null_index, info, buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        cusparse::spgemm_nnz(handle, m, n, k, a_descr, a_nnz, a_row_ptrs,
                             a_col_idxs, b_descr, b_nnz, b_row_ptrs, b_col_idxs,
                             d_descr, zero_nnz, null_index, null_index, c_descr,
                             c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        cusparse::spgemm(handle, m, n, k, &alpha, a_descr, a_nnz, a_vals,
                         a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                         b_row_ptrs, b_col_idxs, null_value, d_descr, zero_nnz,
                         null_value, null_index, null_index, c_descr, c_vals,
                         c_row_ptrs, c_col_idxs, info, buffer);

        cusparse::destroy(info);
        cusparse::destroy(d_descr);
        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);

#else   // CUDA_VERSION >= 11000
        const auto beta = zero<ValueType>();
        auto spgemm_descr = cusparse::create_spgemm_descr();
        auto a_descr = cusparse::create_csr(m, k, a_nnz,
                                            const_cast<IndexType *>(a_row_ptrs),
                                            const_cast<IndexType *>(a_col_idxs),
                                            const_cast<ValueType *>(a_vals));
        auto b_descr = cusparse::create_csr(k, n, b_nnz,
                                            const_cast<IndexType *>(b_row_ptrs),
                                            const_cast<IndexType *>(b_col_idxs),
                                            const_cast<ValueType *>(b_vals));
        auto c_descr = cusparse::create_csr(m, n, zero_nnz, null_index,
                                            null_index, null_value);

        // estimate work
        size_type buffer1_size{};
        cusparse::spgemm_work_estimation(handle, &alpha, a_descr, b_descr,
                                         &beta, c_descr, spgemm_descr,
                                         buffer1_size, nullptr);
        Array<char> buffer1{exec, buffer1_size};
        cusparse::spgemm_work_estimation(handle, &alpha, a_descr, b_descr,
                                         &beta, c_descr, spgemm_descr,
                                         buffer1_size, buffer1.get_data());

        // compute spgemm
        size_type buffer2_size{};
        cusparse::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, nullptr);
        Array<char> buffer2{exec, buffer2_size};
        cusparse::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, buffer2.get_data());

        // copy data to result
        auto c_nnz = cusparse::sparse_matrix_nnz(c_descr);
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        cusparse::csr_set_pointers(c_descr, c_row_ptrs,
                                   c_col_idxs_array.get_data(),
                                   c_vals_array.get_data());

        cusparse::spgemm_copy(handle, &alpha, a_descr, b_descr, &beta, c_descr,
                              spgemm_descr);

        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
        cusparse::destroy(spgemm_descr);
#endif  // CUDA_VERSION >= 11000
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void spgeam(syn::value_list<int, subwarp_size>,
            std::shared_ptr<const DefaultExecutor> exec, const ValueType *alpha,
            const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
            const ValueType *a_vals, const ValueType *beta,
            const IndexType *b_row_ptrs, const IndexType *b_col_idxs,
            const ValueType *b_vals, matrix::Csr<ValueType, IndexType> *c)
{
    auto m = static_cast<IndexType>(c->get_size()[0]);
    auto c_row_ptrs = c->get_row_ptrs();
    // count nnz for alpha * A + beta * B
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(m, subwarps_per_block);
    kernel::spgeam_nnz<subwarp_size><<<num_blocks, default_block_size>>>(
        a_row_ptrs, a_col_idxs, b_row_ptrs, b_col_idxs, m, c_row_ptrs);

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, m + 1);

    // accumulate non-zeros for alpha * A + beta * B
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto c_nnz = exec->copy_val_to_host(c_row_ptrs + m);
    c_builder.get_col_idx_array().resize_and_reset(c_nnz);
    c_builder.get_value_array().resize_and_reset(c_nnz);
    auto c_col_idxs = c->get_col_idxs();
    auto c_vals = c->get_values();
    kernel::spgeam<subwarp_size><<<num_blocks, default_block_size>>>(
        as_cuda_type(alpha), a_row_ptrs, a_col_idxs, as_cuda_type(a_vals),
        as_cuda_type(beta), b_row_ptrs, b_col_idxs, as_cuda_type(b_vals), m,
        c_row_ptrs, c_col_idxs, as_cuda_type(c_vals));
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_spgeam, spgeam);


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);

        auto valpha = exec->copy_val_to_host(alpha->get_const_values());
        auto a_nnz = IndexType(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = IndexType(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto vbeta = exec->copy_val_to_host(beta->get_const_values());
        auto d_nnz = IndexType(d->get_num_stored_elements());
        auto d_vals = d->get_const_values();
        auto d_row_ptrs = d->get_const_row_ptrs();
        auto d_col_idxs = d->get_const_col_idxs();
        auto m = IndexType(a->get_size()[0]);
        auto n = IndexType(b->get_size()[1]);
        auto k = IndexType(a->get_size()[1]);
        auto c_row_ptrs = c->get_row_ptrs();

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto &c_col_idxs_array = c_builder.get_col_idx_array();
        auto &c_vals_array = c_builder.get_value_array();
        auto a_descr = cusparse::create_mat_descr();
        auto b_descr = cusparse::create_mat_descr();
        auto c_descr = cusparse::create_mat_descr();
        auto d_descr = cusparse::create_mat_descr();
        auto info = cusparse::create_spgemm_info();
        // allocate buffer
        size_type buffer_size{};
        cusparse::spgemm_buffer_size(
            handle, m, n, k, &valpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, &vbeta, d_descr, d_nnz,
            d_row_ptrs, d_col_idxs, info, buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        cusparse::spgemm_nnz(handle, m, n, k, a_descr, a_nnz, a_row_ptrs,
                             a_col_idxs, b_descr, b_nnz, b_row_ptrs, b_col_idxs,
                             d_descr, d_nnz, d_row_ptrs, d_col_idxs, c_descr,
                             c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        cusparse::spgemm(handle, m, n, k, &valpha, a_descr, a_nnz, a_vals,
                         a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                         b_row_ptrs, b_col_idxs, &vbeta, d_descr, d_nnz, d_vals,
                         d_row_ptrs, d_col_idxs, c_descr, c_vals, c_row_ptrs,
                         c_col_idxs, info, buffer);

        cusparse::destroy(info);
        cusparse::destroy(d_descr);
        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
#else   // CUDA_VERSION >= 11000
        auto null_value = static_cast<ValueType *>(nullptr);
        auto null_index = static_cast<IndexType *>(nullptr);
        auto one_val = one<ValueType>();
        auto zero_val = zero<ValueType>();
        auto zero_nnz = IndexType{};
        auto spgemm_descr = cusparse::create_spgemm_descr();
        auto a_descr = cusparse::create_csr(m, k, a_nnz,
                                            const_cast<IndexType *>(a_row_ptrs),
                                            const_cast<IndexType *>(a_col_idxs),
                                            const_cast<ValueType *>(a_vals));
        auto b_descr = cusparse::create_csr(k, n, b_nnz,
                                            const_cast<IndexType *>(b_row_ptrs),
                                            const_cast<IndexType *>(b_col_idxs),
                                            const_cast<ValueType *>(b_vals));
        auto c_descr = cusparse::create_csr(m, n, zero_nnz, null_index,
                                            null_index, null_value);

        // estimate work
        size_type buffer1_size{};
        cusparse::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                         &zero_val, c_descr, spgemm_descr,
                                         buffer1_size, nullptr);
        Array<char> buffer1{exec, buffer1_size};
        cusparse::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                         &zero_val, c_descr, spgemm_descr,
                                         buffer1_size, buffer1.get_data());

        // compute spgemm
        size_type buffer2_size{};
        cusparse::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, nullptr);
        Array<char> buffer2{exec, buffer2_size};
        cusparse::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, buffer2.get_data());

        // write result to temporary storage
        auto c_tmp_nnz = cusparse::sparse_matrix_nnz(c_descr);
        Array<IndexType> c_tmp_row_ptrs_array(exec, m + 1);
        Array<IndexType> c_tmp_col_idxs_array(exec, c_tmp_nnz);
        Array<ValueType> c_tmp_vals_array(exec, c_tmp_nnz);
        cusparse::csr_set_pointers(c_descr, c_tmp_row_ptrs_array.get_data(),
                                   c_tmp_col_idxs_array.get_data(),
                                   c_tmp_vals_array.get_data());

        cusparse::spgemm_copy(handle, &one_val, a_descr, b_descr, &zero_val,
                              c_descr, spgemm_descr);

        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
        cusparse::destroy(spgemm_descr);

        auto spgeam_total_nnz = c_tmp_nnz + d->get_num_stored_elements();
        auto nnz_per_row = spgeam_total_nnz / m;
        select_spgeam(
            spgeam_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= nnz_per_row ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), exec,
            alpha->get_const_values(), c_tmp_row_ptrs_array.get_const_data(),
            c_tmp_col_idxs_array.get_const_data(),
            c_tmp_vals_array.get_const_data(), beta->get_const_values(),
            d_row_ptrs, d_col_idxs, d_vals, c);
#endif  // CUDA_VERSION >= 11000
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto total_nnz =
        a->get_num_stored_elements() + b->get_num_stored_elements();
    auto nnz_per_row = total_nnz / a->get_size()[0];
    select_spgeam(
        spgeam_kernels(),
        [&](int compiled_subwarp_size) {
            return compiled_subwarp_size >= nnz_per_row ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec,
        alpha->get_const_values(), a->get_const_row_ptrs(),
        a->get_const_col_idxs(), a->get_const_values(),
        beta->get_const_values(), b->get_const_row_ptrs(),
        b->get_const_col_idxs(), b->get_const_values(), c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    const auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::convert_row_ptrs_to_idxs<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(ptrs), as_cuda_type(idxs));
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(result->get_values()));

    auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(row_ptrs), as_cuda_type(col_idxs),
        as_cuda_type(vals), stride, as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();

    const auto slice_size = (result->get_slice_size() == 0)
                                ? matrix::default_slice_size
                                : result->get_slice_size();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? matrix::default_stride_factor
                                   : result->get_stride_factor();
    const int slice_num = ceildiv(num_rows, slice_size);

    const auto source_values = source->get_const_values();
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto grid_dim = ceildiv(num_rows, default_block_size);

    if (grid_dim > 0) {
        kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
            num_rows, as_cuda_type(source_row_ptrs),
            as_cuda_type(nnz_per_row.get_data()));
    }

    grid_dim = slice_num;

    if (grid_dim > 0) {
        kernel::calculate_slice_lengths<<<grid_dim, config::warp_size>>>(
            num_rows, slice_size, stride_factor,
            as_cuda_type(nnz_per_row.get_const_data()),
            as_cuda_type(slice_lengths), as_cuda_type(slice_sets));
    }

    components::prefix_sum(exec, slice_sets, slice_num + 1);

    grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        kernel::fill_in_sellp<<<grid_dim, default_block_size>>>(
            num_rows, slice_size, as_cuda_type(source_values),
            as_cuda_type(source_row_ptrs), as_cuda_type(source_col_idxs),
            as_cuda_type(slice_lengths), as_cuda_type(slice_sets),
            as_cuda_type(result_col_idxs), as_cuda_type(result_values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
{
    const auto source_values = source->get_const_values();
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    const auto stride = result->get_stride();
    const auto max_nnz_per_row = result->get_num_stored_elements_per_row();
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    const auto init_grid_dim =
        ceildiv(max_nnz_per_row * num_rows, default_block_size);

    kernel::initialize_zero_ell<<<init_grid_dim, default_block_size>>>(
        max_nnz_per_row, stride, as_cuda_type(result_values),
        as_cuda_type(result_col_idxs));

    const auto grid_dim =
        ceildiv(num_rows * config::warp_size, default_block_size);

    kernel::fill_in_ell<<<grid_dim, default_block_size>>>(
        num_rows, stride, as_cuda_type(source_values),
        as_cuda_type(source_row_ptrs), as_cuda_type(source_col_idxs),
        as_cuda_type(result_values), as_cuda_type(result_col_idxs));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    const auto num_rows = source->get_size()[0];

    if (num_rows == 0) {
        *result = 0;
        return;
    }

    const auto slice_num = ceildiv(num_rows, slice_size);
    const auto row_ptrs = source->get_const_row_ptrs();

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(row_ptrs), as_cuda_type(nnz_per_row.get_data()));

    grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);
    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);

    kernel::reduce_max_nnz_per_slice<<<grid_dim, default_block_size>>>(
        num_rows, slice_size, stride_factor,
        as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(max_nnz_per_slice.get_data()));

    grid_dim = ceildiv(slice_num, default_block_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_total_cols<<<grid_dim, default_block_size>>>(
        slice_num, as_cuda_type(max_nnz_per_slice.get_const_data()),
        as_cuda_type(block_results.get_data()));

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_total_cols<<<1, default_block_size>>>(
        grid_dim, as_cuda_type(block_results.get_const_data()),
        as_cuda_type(d_result.get_data()));

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
#else  // CUDA_VERSION >= 11000
        cudaDataType_t cu_value =
            gko::kernels::cuda::cuda_data_type<ValueType>();
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        size_type buffer_size = 0;
        cusparse::transpose_buffersize(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, &buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, buffer);
#endif
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(
            ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
#else  // CUDA_VERSION >= 11000
        cudaDataType_t cu_value =
            gko::kernels::cuda::cuda_data_type<ValueType>();
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        size_type buffer_size = 0;
        cusparse::transpose_buffersize(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, &buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, buffer);
#endif

        conjugate_kernel<<<grid_size, block_size, 0, 0>>>(
            trans->get_num_stored_elements(),
            as_cuda_type(trans->get_values()));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename IndexType>
void invert_permutation(std::shared_ptr<const DefaultExecutor> exec,
                        size_type size, const IndexType *permutation_indices,
                        IndexType *inv_permutation)
{
    auto num_blocks = ceildiv(size, default_block_size);
    inv_permutation_kernel<<<num_blocks, default_block_size>>>(
        size, permutation_indices, inv_permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const CudaExecutor> exec,
                 const IndexType *perm,
                 const matrix::Csr<ValueType, IndexType> *orig,
                 matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    row_ptr_permute_kernel<<<count_num_blocks, default_block_size>>>(
        num_rows, perm, orig->get_const_row_ptrs(),
        row_permuted->get_row_ptrs());
    components::prefix_sum(exec, row_permuted->get_row_ptrs(), num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    row_permute_kernel<config::warp_size>
        <<<copy_num_blocks, default_block_size>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), as_cuda_type(orig->get_const_values()),
            row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
            as_cuda_type(row_permuted->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const CudaExecutor> exec,
                         const IndexType *perm,
                         const matrix::Csr<ValueType, IndexType> *orig,
                         matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel<<<count_num_blocks, default_block_size>>>(
        num_rows, perm, orig->get_const_row_ptrs(),
        row_permuted->get_row_ptrs());
    components::prefix_sum(exec, row_permuted->get_row_ptrs(), num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_row_permute_kernel<config::warp_size>
        <<<copy_num_blocks, default_block_size>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), as_cuda_type(orig->get_const_values()),
            row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
            as_cuda_type(row_permuted->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const CudaExecutor> exec,
                            const IndexType *perm,
                            const matrix::Csr<ValueType, IndexType> *orig,
                            matrix::Csr<ValueType, IndexType> *column_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto nnz = orig->get_num_stored_elements();
    auto num_blocks = ceildiv(std::max(num_rows, nnz), default_block_size);
    col_permute_kernel<<<num_blocks, default_block_size>>>(
        num_rows, nnz, perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), as_cuda_type(orig->get_const_values()),
        column_permuted->get_row_ptrs(), column_permuted->get_col_idxs(),
        as_cuda_type(column_permuted->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const CudaExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto block_results = Array<size_type>(exec, default_block_size);
    auto d_result = Array<size_type>(exec, 1);

    const auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(source->get_const_row_ptrs()),
        as_cuda_type(nnz_per_row.get_data()));

    const auto n = ceildiv(num_rows, default_block_size);
    const auto reduce_dim = n <= default_block_size ? n : default_block_size;
    kernel::reduce_max_nnz<<<reduce_dim, default_block_size>>>(
        num_rows, as_cuda_type(nnz_per_row.get_const_data()),
        as_cuda_type(block_results.get_data()));

    kernel::reduce_max_nnz<<<1, default_block_size>>>(
        reduce_dim, as_cuda_type(block_results.get_const_data()),
        as_cuda_type(d_result.get_data()));

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const CudaExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
{
    auto ell_val = result->get_ell_values();
    auto ell_col = result->get_ell_col_idxs();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    const auto stride = result->get_ell_stride();
    const auto max_nnz_per_row = result->get_ell_num_stored_elements_per_row();
    const auto num_rows = result->get_size()[0];
    const auto coo_num_stored_elements = result->get_coo_num_stored_elements();
    auto grid_dim = ceildiv(max_nnz_per_row * num_rows, default_block_size);

    kernel::initialize_zero_ell<<<grid_dim, default_block_size>>>(
        max_nnz_per_row, stride, as_cuda_type(ell_val), as_cuda_type(ell_col));

    grid_dim = ceildiv(num_rows, default_block_size);
    auto coo_offset = Array<size_type>(exec, num_rows);
    kernel::calculate_hybrid_coo_row_nnz<<<grid_dim, default_block_size>>>(
        num_rows, max_nnz_per_row, as_cuda_type(source->get_const_row_ptrs()),
        as_cuda_type(coo_offset.get_data()));

    components::prefix_sum(exec, coo_offset.get_data(), num_rows);

    grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);
    kernel::fill_in_hybrid<<<grid_dim, default_block_size>>>(
        num_rows, stride, max_nnz_per_row,
        as_cuda_type(source->get_const_values()),
        as_cuda_type(source->get_const_row_ptrs()),
        as_cuda_type(source->get_const_col_idxs()),
        as_cuda_type(coo_offset.get_const_data()), as_cuda_type(ell_val),
        as_cuda_type(ell_col), as_cuda_type(coo_val), as_cuda_type(coo_col),
        as_cuda_type(coo_row));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto num_rows = source->get_size()[0];
    auto row_ptrs = source->get_const_row_ptrs();
    auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::calculate_nnz_per_row<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(row_ptrs), as_cuda_type(result->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const CudaExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        auto descr = cusparse::create_mat_descr();
        auto m = IndexType(to_sort->get_size()[0]);
        auto n = IndexType(to_sort->get_size()[1]);
        auto nnz = IndexType(to_sort->get_num_stored_elements());
        auto row_ptrs = to_sort->get_const_row_ptrs();
        auto col_idxs = to_sort->get_col_idxs();
        auto vals = to_sort->get_values();

        // copy values
        Array<ValueType> tmp_vals_array(exec, nnz);
        exec->copy(nnz, vals, tmp_vals_array.get_data());
        auto tmp_vals = tmp_vals_array.get_const_data();

        // init identity permutation
        Array<IndexType> permutation_array(exec, nnz);
        auto permutation = permutation_array.get_data();
        cusparse::create_identity_permutation(handle, nnz, permutation);

        // allocate buffer
        size_type buffer_size{};
        cusparse::csrsort_buffer_size(handle, m, n, nnz, row_ptrs, col_idxs,
                                      buffer_size);
        Array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();

        // sort column indices
        cusparse::csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
                          permutation, buffer);

        // sort values
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        cusparse::gather(handle, nnz, tmp_vals, vals, permutation);
#else  // CUDA_VERSION >= 11000
        auto val_vec = cusparse::create_spvec(nnz, nnz, permutation, vals);
        auto tmp_vec =
            cusparse::create_dnvec(nnz, const_cast<ValueType *>(tmp_vals));
        cusparse::gather(handle, tmp_vec, val_vec);
#endif

        cusparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    *is_sorted = true;
    auto cpu_array = Array<bool>::view(exec->get_master(), 1, is_sorted);
    auto gpu_array = Array<bool>{exec, cpu_array};
    auto block_size = default_block_size;
    auto num_rows = static_cast<IndexType>(to_check->get_size()[0]);
    auto num_blocks = ceildiv(num_rows, block_size);
    kernel::check_unsorted<<<num_blocks, block_size>>>(
        to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
        num_rows, gpu_array.get_data());
    cpu_array = gpu_array;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const auto nnz = orig->get_num_stored_elements();
    const auto diag_size = diag->get_size()[0];
    const auto num_blocks =
        ceildiv(config::warp_size * diag_size, default_block_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_row_ptrs = orig->get_const_row_ptrs();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    kernel::extract_diagonal<<<num_blocks, default_block_size>>>(
        diag_size, nnz, as_cuda_type(orig_values), as_cuda_type(orig_row_ptrs),
        as_cuda_type(orig_col_idxs), as_cuda_type(diag_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


}  // namespace csr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
