// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    const arithmetic_type value = val[0];
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        arithmetic_type temp_val = zero<arithmetic_type>();
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
