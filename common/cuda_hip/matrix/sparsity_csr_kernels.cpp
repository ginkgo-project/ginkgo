// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sparsity_csr_kernels.hpp"

#include <thrust/sort.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "accessor/reduced_row_major.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup sparsity
 */
namespace sparsity_csr {


constexpr int classical_oversubscription = 32;
constexpr int default_block_size = 512;
#ifdef GKO_COMPILING_HIP
constexpr int spmv_block_size = 256;
#else
constexpr int spmv_block_size = 128;
#endif
constexpr int warps_in_block = 4;


using classical_kernels = syn::value_list<int, 2>;


#include "common/cuda_hip/matrix/csr_common.hpp.inc"

namespace kernel {


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename output_accessor, typename IndexType,
          typename Closure>
__device__ void device_classical_spmv(const size_type num_rows,
                                      const MatrixValueType* __restrict__ val,
                                      const IndexType* __restrict__ col_idxs,
                                      const IndexType* __restrict__ row_ptrs,
                                      acc::range<input_accessor> b,
                                      acc::range<output_accessor> c,
                                      Closure scale)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subrow = thread::get_subwarp_num_flat<subwarp_size>();
    const auto subid = subwarp_tile.thread_rank();
    const IndexType column_id = blockIdx.y;
    const auto value = static_cast<arithmetic_type>(val[0]);
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        auto temp_val = zero<arithmetic_type>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += value * b(col_idxs[ind], column_id);
        }
        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const arithmetic_type& a, const arithmetic_type& b) {
                       return a + b;
                   });
        if (subid == 0) {
            c(row, column_id) = scale(subwarp_result, c(row, column_id));
        }
    }
}


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const MatrixValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    acc::range<output_accessor> c)
{
    using type = typename output_accessor::arithmetic_type;
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, c,
        [](const type& x, const type& y) { return x; });
}


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const MatrixValueType* __restrict__ alpha,
    const MatrixValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    const typename output_accessor::storage_type* __restrict__ beta,
    acc::range<output_accessor> c)
{
    using type = typename output_accessor::arithmetic_type;
    const type alpha_val = alpha[0];
    const type beta_val = beta[0];
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, c,
        [&alpha_val, &beta_val](const type& x, const type& y) {
            return alpha_val * x + beta_val * y;
        });
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::SparsityCsr<ValueType, IndexType>* orig,
               matrix::SparsityCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void fallback_sort(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::SparsityCsr<ValueType, IndexType>* to_sort)
{
    const auto row_ptrs = to_sort->get_const_row_ptrs();
    const auto col_idxs = to_sort->get_col_idxs();
    const auto nnz = to_sort->get_num_nonzeros();
    const auto num_rows = to_sort->get_size()[0];
    array<IndexType> row_idx_array(exec, nnz);
    const auto row_idxs = row_idx_array.get_data();
    components::convert_ptrs_to_idxs(exec, row_ptrs, num_rows, row_idxs);
    // two sorts by integer keys hopefully enable Thrust to use cub's RadixSort
    thrust::sort_by_key(thrust_policy(exec), col_idxs, col_idxs + nnz,
                        row_idxs);
    thrust::stable_sort_by_key(thrust_policy(exec), row_idxs, row_idxs + nnz,
                               col_idxs);
}


namespace host_kernel {


template <int subwarp_size, typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::SparsityCsr<MatrixValueType, IndexType>* a,
                    const matrix::Dense<InputValueType>* b,
                    matrix::Dense<OutputValueType>* c,
                    const matrix::Dense<MatrixValueType>* alpha = nullptr,
                    const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using input_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, const InputValueType>;
    using output_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, OutputValueType>;

    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() *
                        classical_oversubscription;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const auto block = spmv_block_size;

    const auto b_vals = gko::acc::range<input_accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(b->get_size()[0]),
             static_cast<acc::size_type>(b->get_size()[1])}},
        b->get_const_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(b->get_stride())}});
    auto c_vals = gko::acc::range<output_accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(c->get_size()[0]),
             static_cast<acc::size_type>(c->get_size()[1])}},
        c->get_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(c->get_stride())}});
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
        return;
    }
    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                a->get_size()[0], as_device_type(a->get_const_value()),
                a->get_const_col_idxs(),
                as_device_type(a->get_const_row_ptrs()),
                acc::as_device_range(b_vals), acc::as_device_range(c_vals));
    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                a->get_size()[0], as_device_type(alpha->get_const_values()),
                as_device_type(a->get_const_value()), a->get_const_col_idxs(),
                as_device_type(a->get_const_row_ptrs()),
                acc::as_device_range(b_vals),
                as_device_type(beta->get_const_values()),
                acc::as_device_range(c_vals));
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel

template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::SparsityCsr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::SparsityCsr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::SparsityCsr<ValueType, IndexType>* to_sort)
{
    const auto nnz = static_cast<IndexType>(to_sort->get_num_nonzeros());
    const auto num_rows = static_cast<IndexType>(to_sort->get_size()[0]);
    const auto num_cols = static_cast<IndexType>(to_sort->get_size()[1]);
    const auto row_ptrs = to_sort->get_const_row_ptrs();
    const auto col_idxs = to_sort->get_col_idxs();
    if (sparselib::is_supported<ValueType, IndexType>::value) {
        const auto handle = exec->get_sparselib_handle();
        auto descr = sparselib::create_mat_descr();
        array<IndexType> permutation_array(exec, to_sort->get_num_nonzeros());
        auto permutation = permutation_array.get_data();
        components::fill_seq_array(exec, permutation,
                                   to_sort->get_num_nonzeros());
        size_type buffer_size{};
        sparselib::csrsort_buffer_size(handle, num_rows, num_cols, nnz,
                                       row_ptrs, col_idxs, buffer_size);
        array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();
        sparselib::csrsort(handle, num_rows, num_cols, nnz, descr, row_ptrs,
                           col_idxs, permutation, buffer);
        sparselib::destroy(descr);
    } else {
        fallback_sort(exec, to_sort);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* to_check, bool* is_sorted)
{
    *is_sorted = true;
    auto cpu_array = make_array_view(exec->get_master(), 1, is_sorted);
    auto gpu_array = array<bool>{exec, cpu_array};
    const auto num_rows = static_cast<IndexType>(to_check->get_size()[0]);
    auto num_blocks = ceildiv(num_rows, default_block_size);
    if (num_blocks > 0) {
        kernel::check_unsorted<<<num_blocks, default_block_size, 0,
                                 exec->get_stream()>>>(
            to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
            num_rows, gpu_array.get_data());
    }
    cpu_array = gpu_array;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity_csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
