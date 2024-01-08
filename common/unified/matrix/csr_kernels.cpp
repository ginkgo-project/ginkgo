// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Csr matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void inv_col_permute(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType* perm,
                     const matrix::Csr<ValueType, IndexType>* orig,
                     matrix::Csr<ValueType, IndexType>* col_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto nnz = orig->get_num_stored_elements();
    auto size = std::max(num_rows + 1, nnz);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tid, auto num_rows, auto num_nonzeros,
                      auto permutation, auto in_row_ptrs, auto in_col_idxs,
                      auto in_vals, auto out_row_ptrs, auto out_col_idxs,
                      auto out_vals) {
            if (tid < num_nonzeros) {
                out_col_idxs[tid] = permutation[in_col_idxs[tid]];
                out_vals[tid] = in_vals[tid];
            }
            if (tid <= num_rows) {
                out_row_ptrs[tid] = in_row_ptrs[tid];
            }
        },
        size, num_rows, nnz, perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), orig->get_const_values(),
        col_permuted->get_row_ptrs(), col_permuted->get_col_idxs(),
        col_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_COL_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_col_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           matrix::Csr<ValueType, IndexType>* col_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto nnz = orig->get_num_stored_elements();
    auto size = std::max(num_rows + 1, nnz);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tid, auto num_rows, auto num_nonzeros, auto scale,
                      auto permutation, auto in_row_ptrs, auto in_col_idxs,
                      auto in_vals, auto out_row_ptrs, auto out_col_idxs,
                      auto out_vals) {
            if (tid < num_nonzeros) {
                const auto out_col = permutation[in_col_idxs[tid]];
                out_col_idxs[tid] = out_col;
                out_vals[tid] = in_vals[tid] / scale[out_col];
            }
            if (tid <= num_rows) {
                out_row_ptrs[tid] = in_row_ptrs[tid];
            }
        },
        size, num_rows, nnz, scale, perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), orig->get_const_values(),
        col_permuted->get_row_ptrs(), col_permuted->get_col_idxs(),
        col_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_COL_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           matrix::Csr<ValueType, IndexType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto nnz, auto alpha, auto x) { x[nnz] *= alpha[0]; },
        x->get_num_stored_elements(), alpha->get_const_values(),
        x->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* alpha,
               matrix::Csr<ValueType, IndexType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto nnz, auto alpha, auto x) { x[nnz] /= alpha[0]; },
        x->get_num_stored_elements(), alpha->get_const_values(),
        x->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* matrix,
                      matrix::Sellp<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_values, auto row_ptrs,
                      auto slice_size, auto slice_sets, auto cols,
                      auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto out_idx = slice_begin * slice_size + local_row;
            for (auto i = row_begin; i < row_begin + slice_length; i++) {
                cols[out_idx] =
                    i < row_end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] =
                    i < row_end ? in_values[i] : zero(values[out_idx]);
                out_idx += slice_size;
            }
        },
        output->get_size()[0], matrix->get_const_col_idxs(),
        matrix->get_const_values(), matrix->get_const_row_ptrs(),
        output->get_slice_size(), output->get_const_slice_sets(),
        output->get_col_idxs(), output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    matrix::Ell<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_values, auto row_ptrs,
                      auto ell_length, auto ell_stride, auto cols,
                      auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            auto out_idx = row;
            for (auto i = row_begin; i < row_begin + ell_length; i++) {
                cols[out_idx] =
                    i < row_end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] =
                    i < row_end ? in_values[i] : zero(values[out_idx]);
                out_idx += ell_stride;
            }
        },
        output->get_size()[0], matrix->get_const_col_idxs(),
        matrix->get_const_values(), matrix->get_const_row_ptrs(),
        output->get_num_stored_elements_per_row(), output->get_stride(),
        output->get_col_idxs(), output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto cols, auto vals,
                      auto ell_stride, auto ell_max_nnz, auto ell_cols,
                      auto ell_vals, auto coo_row_ptrs, auto coo_row_idxs,
                      auto coo_col_idxs, auto coo_vals) {
            const auto row_begin = row_ptrs[row];
            const auto row_size = row_ptrs[row + 1] - row_begin;
            for (int64 i = 0; i < ell_max_nnz; i++) {
                const auto out_idx = row + ell_stride * i;
                const auto in_idx = i + row_begin;
                const bool use = i < row_size;
                ell_cols[out_idx] =
                    use ? cols[in_idx] : invalid_index<IndexType>();
                ell_vals[out_idx] = use ? vals[in_idx] : zero(vals[in_idx]);
            }
            const auto coo_begin = coo_row_ptrs[row];
            for (int64 i = ell_max_nnz; i < row_size; i++) {
                const auto in_idx = i + row_begin;
                const auto out_idx =
                    coo_begin + i - static_cast<int64>(ell_max_nnz);
                coo_row_idxs[out_idx] = row;
                coo_col_idxs[out_idx] = cols[in_idx];
                coo_vals[out_idx] = vals[in_idx];
            }
        },
        source->get_size()[0], source->get_const_row_ptrs(),
        source->get_const_col_idxs(), source->get_const_values(),
        result->get_ell_stride(), result->get_ell_num_stored_elements_per_row(),
        result->get_ell_col_idxs(), result->get_ell_values(), coo_row_ptrs,
        result->get_coo_row_idxs(), result->get_coo_col_idxs(),
        result->get_coo_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename IndexType>
void build_lookup_offsets(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* row_ptrs, const IndexType* col_idxs,
                          size_type num_rows,
                          matrix::csr::sparsity_type allowed,
                          IndexType* storage_offsets)
{
    using matrix::csr::sparsity_bitmap_block_size;
    using matrix::csr::sparsity_type;
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto num_rows,
                      auto allowed, auto storage_offsets) {
            const auto row_begin = row_ptrs[row];
            const auto row_len = row_ptrs[row + 1] - row_begin;
            const auto local_cols = col_idxs + row_begin;
            const auto min_col = row_len > 0 ? local_cols[0] : 0;
            const auto col_range =
                row_len > 0 ? local_cols[row_len - 1] - min_col + 1 : 0;
            if (csr_lookup_allowed(allowed, sparsity_type::full) &&
                row_len == col_range) {
                storage_offsets[row] = 0;
            } else {
                const auto hashmap_storage = row_len == 0 ? 1 : 2 * row_len;
                const auto bitmap_num_blocks = static_cast<int32>(
                    ceildiv(col_range, sparsity_bitmap_block_size));
                const auto bitmap_storage = 2 * bitmap_num_blocks;
                if (csr_lookup_allowed(allowed, sparsity_type::bitmap) &&
                    bitmap_storage <= hashmap_storage) {
                    storage_offsets[row] = bitmap_storage;
                } else {
                    storage_offsets[row] = hashmap_storage;
                }
            }
        },
        num_rows, row_ptrs, col_idxs, num_rows, allowed, storage_offsets);
    components::prefix_sum_nonnegative(exec, storage_offsets, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_CSR_BUILD_LOOKUP_OFFSETS_KERNEL);


template <typename IndexType>
void benchmark_lookup(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      size_type num_rows, const IndexType* storage_offsets,
                      const int64* row_desc, const int32* storage,
                      IndexType sample_size, IndexType* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto num_rows,
                      auto storage_offsets, auto storage, auto row_descs,
                      auto sample_size, auto result) {
            gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
                row_ptrs, col_idxs,  storage_offsets,
                storage,  row_descs, static_cast<size_type>(row)};
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto row_len = row_end - row_begin;
            for (IndexType sample = 0; sample < sample_size; sample++) {
                if (row_len > 0) {
                    const auto sample_idx = row_len * sample / sample_size;
                    const auto col = col_idxs[row_begin + sample_idx];
                    result[row * sample_size + sample] =
                        lookup.lookup_unsafe(col) + row_begin;
                } else {
                    result[row * sample_size + sample] =
                        invalid_index<IndexType>();
                }
            }
        },
        num_rows, row_ptrs, col_idxs, num_rows, storage_offsets, storage,
        row_desc, sample_size, result);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CSR_BENCHMARK_LOOKUP_KERNEL);


}  // namespace csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
