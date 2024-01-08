// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace {


/**
 * The device function of COO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_lines  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 *
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 * @tparam Closure  type of the function used to write the result
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmv_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType* __restrict__ val,
                            const IndexType* __restrict__ col,
                            const IndexType* __restrict__ row,
                            const ValueType* __restrict__ b,
                            const size_type b_stride, ValueType* __restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const auto column_id = blockIdx.y;
    size_type num = (nnz > start) * ceildiv(nnz - start, subwarp_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    for (; ind < ind_end; ind += subwarp_size) {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment = segment_scan<subwarp_size>(
                tile_block, curr_row, temp_val,
                [](ValueType a, ValueType b) { return a + b; });
            if (is_first_in_segment) {
                atomic_add(&(c[curr_row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_end;
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        // segmented scan
        bool is_first_in_segment = segment_scan<subwarp_size>(
            tile_block, curr_row, temp_val,
            [](ValueType a, ValueType b) { return a + b; });
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp_val));
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride)
{
    spmv_kernel(nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
                [](const ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType& x) { return scale_factor * x; });
}


/**
 * The device function of COO spmm
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_elems  the maximum number of nonzeros in each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param num_cols the number of columns of the matrix
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 *
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 * @tparam Closure  type of the function used to write the result
 */
template <typename ValueType, typename IndexType, typename Closure>
__device__ void spmm_kernel(const size_type nnz, const size_type num_elems,
                            const ValueType* __restrict__ val,
                            const IndexType* __restrict__ col,
                            const IndexType* __restrict__ row,
                            const size_type num_cols,
                            const ValueType* __restrict__ b,
                            const size_type b_stride, ValueType* __restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp = zero<ValueType>();
    const auto coo_idx =
        (static_cast<size_type>(blockDim.y) * blockIdx.x + threadIdx.y) *
        num_elems;
    const auto column_id = blockIdx.y * blockDim.x + threadIdx.x;
    const auto coo_end =
        (coo_idx + num_elems > nnz) ? nnz : coo_idx + num_elems;
    if (column_id < num_cols && coo_idx < nnz) {
        auto curr_row = row[coo_idx];
        auto idx = coo_idx;
        for (; idx < coo_end - 1; idx++) {
            temp += val[idx] * b[col[idx] * b_stride + column_id];
            const auto next_row = row[idx + 1];
            if (next_row != curr_row) {
                atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp));
                curr_row = next_row;
                temp = zero<ValueType>();
            }
        }
        temp += val[idx] * b[col[idx] * b_stride + column_id];
        atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp));
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_elems,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const size_type num_cols,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    spmm_kernel(nnz, num_elems, val, col, row, num_cols, b, b_stride, c,
                c_stride, [](const ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_elems,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ val,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const size_type num_cols, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmm_kernel(
        nnz, num_elems, val, col, row, num_cols, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType& x) { return scale_factor * x; });
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto b_ncols = b->get_size()[1];
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0 && b_ncols > 0) {
        // TODO: b_ncols needs to be tuned for ROCm.
        if (b_ncols < 4) {
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);

            abstract_spmv<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_lines, as_device_type(a->get_const_values()),
                a->get_const_col_idxs(),
                as_device_type(a->get_const_row_idxs()),
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));

            abstract_spmm<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_elems, as_device_type(a->get_const_values()),
                a->get_const_col_idxs(),
                as_device_type(a->get_const_row_idxs()), b_ncols,
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto b_ncols = b->get_size()[1];

    if (nwarps > 0 && b_ncols > 0) {
        // TODO: b_ncols needs to be tuned for ROCm.
        if (b_ncols < 4) {
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);

            abstract_spmv<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_lines, as_device_type(alpha->get_const_values()),
                as_device_type(a->get_const_values()), a->get_const_col_idxs(),
                as_device_type(a->get_const_row_idxs()),
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));

            abstract_spmm<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_elems, as_device_type(alpha->get_const_values()),
                as_device_type(a->get_const_values()), a->get_const_col_idxs(),
                as_device_type(a->get_const_row_idxs()), b_ncols,
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);
