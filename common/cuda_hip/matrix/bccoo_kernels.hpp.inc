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

/**
 * The device function of BCCOO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param chunk_data  the array where the data are
 * @param offsets_data  the array where the offset of each block is
 * @param types_data  the array where the type of each block is
 * @param cols_data  the array where the initial column of each block is
 * @param rows_data  the array where the initial row of each block is
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
__device__ void spmv_kernel(const size_type nnz, const size_type num_blks,
                            const size_type block_size,
                            const size_type num_lines,
                            const uint8* __restrict__ chunk_data,
                            const size_type* __restrict__ offsets_data,
                            const uint8* __restrict__ types_data,
                            const IndexType* __restrict__ cols_data,
                            const IndexType* __restrict__ rows_data,
                            const ValueType* __restrict__ b,
                            const size_type b_stride, ValueType* __restrict__ c,
                            const size_type c_stride, Closure scale)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    ValueType temp_val = zero<ValueType>();
    bool new_value = false;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        const auto tile_block =
            group::tiled_partition<subwarp_size>(group::this_thread_block());

        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        // compr_idxs idxs = {};
        // compr_idxs<IndexType> idxs;
        // idxs.blk = blk;
        // idxs.shf = offsets_data[blk];
        // idxs.row = rows_data[blk];
        compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
        //        compr_blk_idxs blk_idxs = {};
        //        init_block_indices(rows_data, cols_data, block_size_local,
        //        idxs,
        //                           types_data[blk], blk_idxs);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        if (blk_idxs.mul_row) {
            if (blk_idxs.row_16bits) {
                if (blk_idxs.col_8bits) {
                    loop_block_multi_row<uint16, uint8, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint16, uint8,
                        //                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                } else if (blk_idxs.col_16bits) {
                    loop_block_multi_row<uint16, uint16, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint16, uint16,
                        //                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                } else {
                    loop_block_multi_row<uint16, uint32, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint16, uint32,
                        //                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                }
            } else {
                if (blk_idxs.col_8bits) {
                    loop_block_multi_row<uint8, uint8, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint8, uint8, ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                } else if (blk_idxs.col_16bits) {
                    loop_block_multi_row<uint8, uint16, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint8, uint16,
                        //                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                } else {
                    loop_block_multi_row<uint8, uint32, IndexType>(
                        //                    loop_block_multi_row<subwarp_size,
                        //                    uint8, uint32,
                        //                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale);
                }
            }
        } else {
            if (blk_idxs.col_8bits) {
                //                loop_block_single_row<subwarp_size, uint8,
                //                ValueType>(
                loop_block_single_row<uint8>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale);
            } else if (blk_idxs.col_16bits) {
                loop_block_single_row<uint16>(
                    //                loop_block_single_row<subwarp_size,
                    //                uint16, ValueType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale);
            } else {
                loop_block_single_row<uint32>(
                    //                loop_block_single_row<subwarp_size,
                    //                uint32, ValueType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chk,
    const size_type* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, [](const ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const ValueType* __restrict__ alpha,
    const uint8* __restrict__ chk, const size_type* __restrict__ off,
    const uint8* __restrict__ typ, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, [&scale_factor](const ValueType& x) {
                    return scale_factor * x;
                });
}


/**
 * The device function of BCCOO extract_kernel
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param chunk_data  the array where the data are
 * @param offsets_data  the array where the offset of each block is
 * @param types_data  the array where the type of each block is
 * @param cols_data  the array where the initial column of each block is
 * @param rows_data  the array where the initial row of each block is
 * @param diag  the output dense vector
 *
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__device__ void extract_kernel(const size_type nnz, const size_type num_blks,
                               const size_type block_size,
                               const size_type num_lines,
                               const uint8* __restrict__ chunk_data,
                               const size_type* __restrict__ offsets_data,
                               const uint8* __restrict__ types_data,
                               const IndexType* __restrict__ cols_data,
                               const IndexType* __restrict__ rows_data,
                               ValueType* __restrict__ diag)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        // compr_idxs idxs = {};
        // compr_idxs<IndexType> idxs;
        // idxs.blk = blk;
        // idxs.shf = offsets_data[blk];
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        //        compr_blk_idxs blk_idxs = {};
        //        init_block_indices(rows_data, cols_data, block_size_local,
        //        idxs,
        //                           types_data[blk], blk_idxs);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                // get_block_position_value<IndexType,ValueType>(
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs, val);
                //                    pos, chunk_data, blk_idxs, idxs.row,
                //                    idxs.col, val);
                if (idxs.row == idxs.col) diag[idxs.col] = val;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_extract(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chk,
    const size_type* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    ValueType* __restrict__ diag)
{
    extract_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col,
                   row, diag);
}


/**
 * The device function of BCCOO compute_absolute_inplace
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param chunk_data  the array where the data are
 * @param offsets_data  the array where the offset of each block is
 * @param types_data  the array where the type of each block is
 * @param cols_data  the array where the initial column of each block is
 * @param rows_data  the array where the initial row of each block is
 *
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void absolute_inplace_kernel(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, uint8* __restrict__ chunk_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, Closure comp_abs)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        // compr_idxs idxs = {};
        // compr_idxs<IndexType> idxs;
        // idxs.blk = blk;
        // idxs.shf = offsets_data[blk];
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        //        compr_blk_idxs blk_idxs = {};
        //        init_block_indices(rows_data, cols_data, block_size_local,
        //        idxs,
        //                           types_data[blk], blk_idxs);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                // get_block_position_value_put<IndexType,ValueType, Closure>(
                get_block_position_value_put<IndexType, ValueType, Closure>(
                    pos, chunk_data, blk_idxs, idxs, val,
                    //                    pos, chunk_data, blk_idxs, idxs.row,
                    //                    idxs.col, val,
                    comp_abs);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_absolute_inplace(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, uint8* __restrict__ chk,
    const size_type* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row)
{
    absolute_inplace_kernel<config::warp_size, ValueType, IndexType>(
        nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
        [](ValueType x) { return abs(x); });
}


/**
 * The device function of BCCOO compute_absolute
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param chunk_data_src  the array where the data of source are
 * @param offsets_data_src  the array where the offset of each block of source
 * is
 * @param types_data_src  the array where the sorce type of each block of source
 * is
 * @param cols_data_src  the array where the initial column of each block of
 * source is
 * @param rows_data_src  the array where the initial row of each block of source
 * is
 * @param chunk_data_res  the array where the data of result are
 * @param offsets_data_res  the array where the offset of each block of result
 * is
 * @param types_data_res  the array where the sorce type of each block of result
 * is
 * @param cols_data_res  the array where the initial column of each block of
 * result is
 * @param rows_data_res  the array where the initial row of each block of result
 * is
 *
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void absolute_kernel(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chunk_data_src,
    const size_type* __restrict__ offsets_data_src,
    const uint8* __restrict__ types_data_src,
    const IndexType* __restrict__ cols_data_src,
    const IndexType* __restrict__ rows_data_src,
    uint8* __restrict__ chunk_data_res,
    size_type* __restrict__ offsets_data_res,
    uint8* __restrict__ types_data_res, IndexType* __restrict__ cols_data_res,
    IndexType* __restrict__ rows_data_res, Closure comp_abs)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    offsets_data_res[0] = 0;
    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);

        // compr_idxs idxs_src = {};
        // compr_idxs<IndexType> idxs_src;
        // idxs_src.blk = blk;
        // idxs_src.shf = offsets_data_src[blk];
        compr_idxs<IndexType> idxs_src(blk, offsets_data_src[blk]);
        //        compr_blk_idxs blk_idxs_src = {};
        //        init_block_indices(rows_data_src, cols_data_src,
        //        block_size_local,
        //                           idxs_src, types_data_src[blk],
        //                           blk_idxs_src);
        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local, idxs_src,
                                               types_data_src[blk]);

        rows_data_res[blk] = rows_data_src[blk];
        cols_data_res[blk] = cols_data_src[blk];
        types_data_res[blk] = types_data_src[blk];
        offsets_data_res[blk] =
            offsets_data_src[blk] -
            ((blk == 0)
                 ? 0
                 : (blk - 1) * block_size *
                       (sizeof(ValueType) - sizeof(remove_complex<ValueType>)));

        // compr_idxs idxs_res = {};
        // compr_idxs<IndexType> idxs_res;
        // idxs_res.blk = blk;
        // idxs_res.shf = offsets_data_res[blk];
        compr_idxs<IndexType> idxs_res(blk, offsets_data_res[blk]);
        //        compr_blk_idxs blk_idxs_res = {};
        //        init_block_indices(rows_data_res, cols_data_res,
        //        block_size_local,
        //                           idxs_res, types_data_res[blk],
        //                           blk_idxs_res);
        compr_blk_idxs<IndexType> blk_idxs_res(rows_data_res, cols_data_res,
                                               block_size_local, idxs_res,
                                               types_data_res[blk]);
        offsets_data_res[blk + 1] =
            blk_idxs_res.shf_val +
            block_size_local * sizeof(remove_complex<ValueType>);

        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value_put<
                    // IndexType, ValueType,
                    // remove_complex<ValueType>, Closure>(
                    IndexType, ValueType, remove_complex<ValueType>, Closure>(
                    pos, chunk_data_src, blk_idxs_src, chunk_data_res,
                    blk_idxs_res, idxs_src, val, comp_abs);
                //                    blk_idxs_res, idxs_src.row, idxs_src.col,
                //                    val, comp_abs);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_absolute(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chk_src,
    const size_type* __restrict__ off_src, const uint8* __restrict__ typ_src,
    const IndexType* __restrict__ col_src,
    const IndexType* __restrict__ row_src, uint8* __restrict__ chk_res,
    size_type* __restrict__ off_res, uint8* __restrict__ typ_res,
    IndexType* __restrict__ col_res, IndexType* __restrict__ row_res)
{
    absolute_kernel<config::warp_size, ValueType, IndexType>(
        nnz, num_blks, block_size, num_lines, chk_src, off_src, typ_src,
        col_src, row_src, chk_res, off_res, typ_res, col_res, row_res,
        [](ValueType x) { return abs(x); });
}


}  // namespace kernel


namespace kernel {


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_row_idxs_to_ptrs(
    const IndexType* __restrict__ idxs, size_type num_nonzeros,
    IndexType* __restrict__ ptrs, size_type length)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx == 0) {
        ptrs[0] = 0;
        ptrs[length - 1] = num_nonzeros;
    }

    if (0 < tidx && tidx < num_nonzeros) {
        if (idxs[tidx - 1] < idxs[tidx]) {
            for (auto i = idxs[tidx - 1] + 1; i <= idxs[tidx]; i++) {
                ptrs[i] = tidx;
            }
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(config::max_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType* __restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chunk_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, size_type stride,
    ValueType* __restrict__ result)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        // compr_idxs idxs = {};
        // compr_idxs<IndexType> idxs;
        // idxs.blk = blk;
        // idxs.shf = offsets_data[blk];
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        //        compr_blk_idxs blk_idxs = {};
        //        init_block_indices(rows_data, cols_data, block_size_local,
        //        idxs,
        //                           types_data[blk], blk_idxs);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                // get_block_position_value<IndexType,ValueType>(
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs, val);
                //                    pos, chunk_data, blk_idxs, idxs.row,
                //                    idxs.col, val);
                result[idxs.row * stride + idxs.col] = val;
            }
        }
    }
}


template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_coo(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chunk_data,
    //    const IndexType* __restrict__ offsets_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, IndexType* __restrict__ rows_idxs,
    IndexType* __restrict__ cols_idxs, ValueType* __restrict__ values)
{
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        // compr_idxs idxs = {};
        // compr_idxs<IndexType> idxs;
        // idxs.blk = blk;
        // idxs.shf = offsets_data[blk];
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        //        compr_blk_idxs blk_idxs = {};
        //        init_block_indices(rows_data, cols_data, block_size_local,
        //        idxs,
        //                           types_data[blk], blk_idxs);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                // get_block_position_value<IndexType,ValueType>(
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs, val);
                //                    pos, chunk_data, blk_idxs, idxs.row,
                //                    idxs.col, val);
                auto index = blk * block_size + pos;
                rows_idxs[index] = idxs.row;
                cols_idxs[index] = idxs.col;
                values[index] = val;
            }
        }
    }
}


}  // namespace kernel
