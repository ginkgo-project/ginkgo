/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::SparsityCsr<ValueType, IndexType>* input,
                   matrix::Dense<ValueType>* output) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_num_diagonal_elements(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* matrix,
    size_type* num_diagonal_elements) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs,
    matrix::SparsityCsr<ValueType, IndexType>* matrix) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::SparsityCsr<ValueType, IndexType>* orig,
               matrix::SparsityCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::SparsityCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);
