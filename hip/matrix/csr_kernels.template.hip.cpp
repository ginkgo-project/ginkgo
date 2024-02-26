// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_kernels.hpp"


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
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "accessor/hip_helper.hpp"
#include "core/base/array_access.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_accessor_helper.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/thrust.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
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
 * @ingroup csr
 */
namespace csr {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int classical_oversubscription = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/matrix/csr_common.hpp.inc"
#include "common/cuda_hip/matrix/csr_kernels.hpp.inc"


namespace host_kernel {
namespace {


template <int items_per_thread, typename MatrixValueType,
          typename InputValueType, typename OutputValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const HipExecutor> exec,
                     const matrix::Csr<MatrixValueType, IndexType>* a,
                     const matrix::Dense<InputValueType>* b,
                     matrix::Dense<OutputValueType>* c,
                     const matrix::Dense<MatrixValueType>* alpha = nullptr,
                     const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const auto grid = grid_num;
    const auto block = spmv_block_size;
    // TODO: workspace?
    array<IndexType> row_out(exec, grid_num);
    // TODO: should we store the value in arithmetic_type or output_type?
    array<arithmetic_type> val_out(exec, grid_num);

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        const auto column_span =
            acc::index_span(static_cast<acc::size_type>(column_id),
                            static_cast<acc::size_type>(column_id + 1));
        const auto b_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(b,
                                                                   column_span);
        auto c_vals =
            acc::helper::build_rrm_accessor<arithmetic_type>(c, column_span);
        if (alpha == nullptr && beta == nullptr) {
            if (grid_num > 0) {
                kernel::abstract_merge_path_spmv<items_per_thread>
                    <<<grid, block, 0, exec->get_stream()>>>(
                        static_cast<IndexType>(a->get_size()[0]),
                        acc::as_hip_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_hip_range(b_vals), acc::as_hip_range(c_vals),
                        as_device_type(row_out.get_data()),
                        as_device_type(val_out.get_data()));
            }
            kernel::
                abstract_reduce<<<1, spmv_block_size, 0, exec->get_stream()>>>(
                    grid_num, as_device_type(val_out.get_data()),
                    as_device_type(row_out.get_data()),
                    acc::as_hip_range(c_vals));

        } else if (alpha != nullptr && beta != nullptr) {
            if (grid_num > 0) {
                kernel::abstract_merge_path_spmv<items_per_thread>
                    <<<grid, block, 0, exec->get_stream()>>>(
                        static_cast<IndexType>(a->get_size()[0]),
                        as_device_type(alpha->get_const_values()),
                        acc::as_hip_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_hip_range(b_vals),
                        as_device_type(beta->get_const_values()),
                        acc::as_hip_range(c_vals),
                        as_device_type(row_out.get_data()),
                        as_device_type(val_out.get_data()));
            }
            kernel::
                abstract_reduce<<<1, spmv_block_size, 0, exec->get_stream()>>>(
                    grid_num, as_device_type(val_out.get_data()),
                    as_device_type(row_out.get_data()),
                    as_device_type(alpha->get_const_values()),
                    acc::as_hip_range(c_vals));
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const HipExecutor> exec)
{
#if GINKGO_HIP_PLATFORM_NVCC


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


#else


    // HIP uses the minimal num_item to make the code work correctly.
    // TODO: this parameter should be tuned.
    int num_item = 6;


#endif  // GINKGO_HIP_PLATFORM_NVCC


    // Ensure that the following is satisfied:
    // sizeof(IndexType) + sizeof(ValueType)
    // <= items_per_thread * sizeof(IndexType)
    constexpr int minimal_num =
        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
    int items_per_thread = num_item * 4 / sizeof(IndexType);
    return std::max(minimal_num, items_per_thread);
}


template <int subwarp_size, typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const HipExecutor> exec,
                    const matrix::Csr<MatrixValueType, IndexType>* a,
                    const matrix::Dense<InputValueType>* b,
                    matrix::Dense<OutputValueType>* c,
                    const matrix::Dense<MatrixValueType>* alpha = nullptr,
                    const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() *
                        classical_oversubscription;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const auto block = spmv_block_size;

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
    const auto b_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
    auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
    if (alpha == nullptr && beta == nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subwarp_size>
                <<<grid, block, 0, exec->get_stream()>>>(
                    a->get_size()[0], acc::as_hip_range(a_vals),
                    a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    acc::as_hip_range(b_vals), acc::as_hip_range(c_vals));
        }
    } else if (alpha != nullptr && beta != nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subwarp_size>
                <<<grid, block, 0, exec->get_stream()>>>(
                    a->get_size()[0], as_device_type(alpha->get_const_values()),
                    acc::as_hip_range(a_vals), a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    acc::as_hip_range(b_vals),
                    as_device_type(beta->get_const_values()),
                    acc::as_hip_range(c_vals));
        }
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void load_balance_spmv(std::shared_ptr<const HipExecutor> exec,
                       const matrix::Csr<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       matrix::Dense<OutputValueType>* c,
                       const matrix::Dense<MatrixValueType>* alpha = nullptr,
                       const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    if (beta) {
        dense::scale(exec, beta, c);
    } else {
        dense::fill(exec, c, zero<OutputValueType>());
    }
    const IndexType nwarps = a->get_num_srow_elements();
    if (nwarps > 0) {
        const dim3 csr_block(config::warp_size, warps_in_block, 1);
        const dim3 csr_grid(ceildiv(nwarps, warps_in_block), b->get_size()[1]);
        const auto a_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
        const auto b_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
        auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
        if (alpha) {
            if (csr_grid.x > 0 && csr_grid.y > 0) {
                kernel::abstract_spmv<<<csr_grid, csr_block, 0,
                                        exec->get_stream()>>>(
                    nwarps, static_cast<IndexType>(a->get_size()[0]),
                    as_device_type(alpha->get_const_values()),
                    acc::as_hip_range(a_vals), a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    as_device_type(a->get_const_srow()),
                    acc::as_hip_range(b_vals), acc::as_hip_range(c_vals));
            }
        } else {
            if (csr_grid.x > 0 && csr_grid.y > 0) {
                kernel::abstract_spmv<<<csr_grid, csr_block, 0,
                                        exec->get_stream()>>>(
                    nwarps, static_cast<IndexType>(a->get_size()[0]),
                    acc::as_hip_range(a_vals), a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    as_device_type(a->get_const_srow()),
                    acc::as_hip_range(b_vals), acc::as_hip_range(c_vals));
            }
        }
    }
}


template <typename ValueType, typename IndexType>
bool try_general_sparselib_spmv(std::shared_ptr<const HipExecutor> exec,
                                const ValueType* alpha,
                                const matrix::Csr<ValueType, IndexType>* a,
                                const matrix::Dense<ValueType>* b,
                                const ValueType* beta,
                                matrix::Dense<ValueType>* c)
{
    bool try_sparselib = hipsparse::is_supported<ValueType, IndexType>::value;
    try_sparselib =
        try_sparselib && b->get_stride() == 1 && c->get_stride() == 1;
    // rocSPARSE has issues with zero matrices
    try_sparselib = try_sparselib && a->get_num_stored_elements() > 0;
    if (try_sparselib) {
        auto descr = hipsparse::create_mat_descr();

        auto row_ptrs = a->get_const_row_ptrs();
        auto col_idxs = a->get_const_col_idxs();

        hipsparse::spmv(exec->get_hipsparse_handle(),
                        HIPSPARSE_OPERATION_NON_TRANSPOSE, a->get_size()[0],
                        a->get_size()[1], a->get_num_stored_elements(), alpha,
                        descr, a->get_const_values(), row_ptrs, col_idxs,
                        b->get_const_values(), beta, c->get_values());

        hipsparse::destroy(descr);
    }
    return try_sparselib;
}


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType,
          typename = std::enable_if_t<
              !std::is_same<MatrixValueType, InputValueType>::value ||
              !std::is_same<MatrixValueType, OutputValueType>::value>>
bool try_sparselib_spmv(std::shared_ptr<const HipExecutor> exec,
                        const matrix::Csr<MatrixValueType, IndexType>* a,
                        const matrix::Dense<InputValueType>* b,
                        matrix::Dense<OutputValueType>* c,
                        const matrix::Dense<MatrixValueType>* alpha = nullptr,
                        const matrix::Dense<OutputValueType>* beta = nullptr)
{
    // TODO: support sparselib mixed
    return false;
}

template <typename ValueType, typename IndexType>
bool try_sparselib_spmv(std::shared_ptr<const HipExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* a,
                        const matrix::Dense<ValueType>* b,
                        matrix::Dense<ValueType>* c,
                        const matrix::Dense<ValueType>* alpha = nullptr,
                        const matrix::Dense<ValueType>* beta = nullptr)
{
    if (alpha) {
        return try_general_sparselib_spmv(exec, alpha->get_const_values(), a, b,
                                          beta->get_const_values(), c);
    } else {
        auto handle = exec->get_hipsparse_handle();
        hipsparse::pointer_mode_guard pm_guard(handle);
        const auto valpha = one<ValueType>();
        const auto vbeta = zero<ValueType>();
        return try_general_sparselib_spmv(exec, &valpha, a, b, &vbeta, c);
    }
}


}  // anonymous namespace
}  // namespace host_kernel


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::Csr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
    } else if (a->get_strategy()->get_name() == "load_balance") {
        host_kernel::load_balance_spmv(exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "sparselib" ||
            a->get_strategy()->get_name() == "cusparse") {
            use_classical = !host_kernel::try_sparselib_spmv(exec, a, b, c);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
        }
    }
}


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Csr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
    } else if (a->get_strategy()->get_name() == "load_balance") {
        host_kernel::load_balance_spmv(exec, a, b, c, alpha, beta);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "sparselib" ||
            a->get_strategy()->get_name() == "cusparse") {
            use_classical =
                !host_kernel::try_sparselib_spmv(exec, a, b, c, alpha, beta);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c,
                alpha, beta);
        }
    }
}


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const HipExecutor> exec,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_hipsparse_handle();
        hipsparse::pointer_mode_guard pm_guard(handle);
        auto a_descr = hipsparse::create_mat_descr();
        auto b_descr = hipsparse::create_mat_descr();
        auto c_descr = hipsparse::create_mat_descr();
        auto d_descr = hipsparse::create_mat_descr();
        auto info = hipsparse::create_spgemm_info();

        auto alpha = one<ValueType>();
        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto null_value = static_cast<ValueType*>(nullptr);
        auto null_index = static_cast<IndexType*>(nullptr);
        auto zero_nnz = IndexType{};
        auto m = static_cast<IndexType>(a->get_size()[0]);
        auto n = static_cast<IndexType>(b->get_size()[1]);
        auto k = static_cast<IndexType>(a->get_size()[1]);
        auto c_row_ptrs = c->get_row_ptrs();
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto& c_col_idxs_array = c_builder.get_col_idx_array();
        auto& c_vals_array = c_builder.get_value_array();

        // allocate buffer
        size_type buffer_size{};
        hipsparse::spgemm_buffer_size(
            handle, m, n, k, &alpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            zero_nnz, null_index, null_index, info, buffer_size);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        hipsparse::spgemm_nnz(
            handle, m, n, k, a_descr, a_nnz, a_row_ptrs, a_col_idxs, b_descr,
            b_nnz, b_row_ptrs, b_col_idxs, d_descr, zero_nnz, null_index,
            null_index, c_descr, c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        hipsparse::spgemm(handle, m, n, k, &alpha, a_descr, a_nnz, a_vals,
                          a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                          b_row_ptrs, b_col_idxs, null_value, d_descr, zero_nnz,
                          null_value, null_index, null_index, c_descr, c_vals,
                          c_row_ptrs, c_col_idxs, info, buffer);

        hipsparse::destroy_spgemm_info(info);
        hipsparse::destroy(d_descr);
        hipsparse::destroy(c_descr);
        hipsparse::destroy(b_descr);
        hipsparse::destroy(a_descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const HipExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     const matrix::Dense<ValueType>* beta,
                     const matrix::Csr<ValueType, IndexType>* d,
                     matrix::Csr<ValueType, IndexType>* c)
{
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_hipsparse_handle();
        hipsparse::pointer_mode_guard pm_guard(handle);
        auto a_descr = hipsparse::create_mat_descr();
        auto b_descr = hipsparse::create_mat_descr();
        auto c_descr = hipsparse::create_mat_descr();
        auto d_descr = hipsparse::create_mat_descr();
        auto info = hipsparse::create_spgemm_info();

        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto d_vals = d->get_const_values();
        auto d_row_ptrs = d->get_const_row_ptrs();
        auto d_col_idxs = d->get_const_col_idxs();
        auto null_value = static_cast<ValueType*>(nullptr);
        auto null_index = static_cast<IndexType*>(nullptr);
        auto one_value = one<ValueType>();
        auto m = static_cast<IndexType>(a->get_size()[0]);
        auto n = static_cast<IndexType>(b->get_size()[1]);
        auto k = static_cast<IndexType>(a->get_size()[1]);

        // allocate buffer
        size_type buffer_size{};
        hipsparse::spgemm_buffer_size(
            handle, m, n, k, &one_value, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            IndexType{}, null_index, null_index, info, buffer_size);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        array<IndexType> c_tmp_row_ptrs_array(exec, m + 1);
        auto c_tmp_row_ptrs = c_tmp_row_ptrs_array.get_data();
        IndexType c_nnz{};
        hipsparse::spgemm_nnz(
            handle, m, n, k, a_descr, a_nnz, a_row_ptrs, a_col_idxs, b_descr,
            b_nnz, b_row_ptrs, b_col_idxs, d_descr, IndexType{}, null_index,
            null_index, c_descr, c_tmp_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros for A * B
        array<IndexType> c_tmp_col_idxs_array(exec, c_nnz);
        array<ValueType> c_tmp_vals_array(exec, c_nnz);
        auto c_tmp_col_idxs = c_tmp_col_idxs_array.get_data();
        auto c_tmp_vals = c_tmp_vals_array.get_data();
        hipsparse::spgemm(handle, m, n, k, &one_value, a_descr, a_nnz, a_vals,
                          a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                          b_row_ptrs, b_col_idxs, null_value, d_descr,
                          IndexType{}, null_value, null_index, null_index,
                          c_descr, c_tmp_vals, c_tmp_row_ptrs, c_tmp_col_idxs,
                          info, buffer);

        // destroy hipsparse context
        hipsparse::destroy_spgemm_info(info);
        hipsparse::destroy(d_descr);
        hipsparse::destroy(c_descr);
        hipsparse::destroy(b_descr);
        hipsparse::destroy(a_descr);

        auto total_nnz = c_nnz + d->get_num_stored_elements();
        auto nnz_per_row = total_nnz / m;
        select_spgeam(
            spgeam_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= nnz_per_row ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), exec,
            alpha->get_const_values(), c_tmp_row_ptrs, c_tmp_col_idxs,
            c_tmp_vals, beta->get_const_values(), d_row_ptrs, d_col_idxs,
            d_vals, c);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* orig,
               matrix::Csr<ValueType, IndexType>* trans)
{
    if (orig->get_size()[0] == 0) {
        return;
    }
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        hipsparseAction_t copyValues = HIPSPARSE_ACTION_NUMERIC;
        hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

        hipsparse::transpose(
            exec->get_hipsparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
    } else {
        fallback_transpose(exec, orig, trans);
    }
}


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* orig,
                    matrix::Csr<ValueType, IndexType>* trans)
{
    if (orig->get_size()[0] == 0) {
        return;
    }
    const auto block_size = default_block_size;
    const auto grid_size =
        ceildiv(trans->get_num_stored_elements(), block_size);
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        hipsparseAction_t copyValues = HIPSPARSE_ACTION_NUMERIC;
        hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

        hipsparse::transpose(
            exec->get_hipsparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
    } else {
        fallback_transpose(exec, orig, trans);
    }
    if (grid_size > 0 && is_complex<ValueType>()) {
        kernel::conjugate<<<grid_size, block_size, 0, exec->get_stream()>>>(
            trans->get_num_stored_elements(),
            as_device_type(trans->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const HipExecutor> exec,
                          matrix::Csr<ValueType, IndexType>* to_sort)
{
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_hipsparse_handle();
        auto descr = hipsparse::create_mat_descr();
        auto m = IndexType(to_sort->get_size()[0]);
        auto n = IndexType(to_sort->get_size()[1]);
        auto nnz = IndexType(to_sort->get_num_stored_elements());
        auto row_ptrs = to_sort->get_const_row_ptrs();
        auto col_idxs = to_sort->get_col_idxs();
        auto vals = to_sort->get_values();

        // copy values
        array<ValueType> tmp_vals_array(exec, nnz);
        exec->copy(nnz, vals, tmp_vals_array.get_data());
        auto tmp_vals = tmp_vals_array.get_const_data();

        // init identity permutation
        array<IndexType> permutation_array(exec, nnz);
        auto permutation = permutation_array.get_data();
        hipsparse::create_identity_permutation(handle, nnz, permutation);

        // allocate buffer
        size_type buffer_size{};
        hipsparse::csrsort_buffer_size(handle, m, n, nnz, row_ptrs, col_idxs,
                                       buffer_size);
        array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();

        // sort column indices
        hipsparse::csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
                           permutation, buffer);

        // sort values
        hipsparse::gather(handle, nnz, tmp_vals, vals, permutation);

        hipsparse::destroy(descr);
    } else {
        fallback_sort(exec, to_sort);
    }
}


}  // namespace csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
