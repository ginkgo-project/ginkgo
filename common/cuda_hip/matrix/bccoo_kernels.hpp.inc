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
 * @param compressed_data  the array where the data are
 * @param block_offsets  the array where the offset of each block is
 * @param compression_types  the array where the type of each block is
 * @param start_cols  the array where the initial column of each block is
 * @param start_rows  the array where the initial row of each block is
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 * @tparam Closure  type of the function used to write the result
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmv_kernel(const IndexType nnz, const IndexType num_blks,
                            const IndexType block_size,
                            const IndexType num_lines,
                            const uint8* __restrict__ compressed_data,
                            const size_type* __restrict__ block_offsets,
                            const uint8* __restrict__ compression_types,
                            const IndexType* __restrict__ start_cols,
                            const IndexType* __restrict__ start_rows,
                            const ValueType* __restrict__ b,
                            const IndexType b_stride, ValueType* __restrict__ c,
                            const IndexType c_stride, Closure scale)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    ValueType temp_val = zero<ValueType>();

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        const auto tile_block =
            group::tiled_partition<subwarp_size>(group::this_thread_block());

        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, block_offsets[blk], start_rows[blk]);
        compr_blk_idxs<IndexType> blk_idxs(start_rows, start_cols,
                                           block_size_local, idxs,
                                           compression_types[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_spmv<uint16, uint8, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_spmv<uint16, uint16, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                } else {
                    loop_block_multi_row_spmv<uint16, uint32, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_spmv<uint8, uint8, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_spmv<uint8, uint16, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                } else {
                    loop_block_multi_row_spmv<uint8, uint32, IndexType>(
                        compressed_data, block_size_local, b, b_stride,
                        column_id, c, c_stride, idxs, blk_idxs, start_in_blk,
                        jump_in_blk, scale);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_spmv<uint8>(
                    compressed_data, block_size_local, b, b_stride, column_id,
                    c, c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                    scale);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_spmv<uint16>(
                    compressed_data, block_size_local, b, b_stride, column_id,
                    c, c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                    scale);
            } else {
                loop_block_single_row_spmv<uint32>(
                    compressed_data, block_size_local, b, b_stride, column_id,
                    c, c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                    scale);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const uint8* __restrict__ chk,
    const size_type* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const ValueType* __restrict__ b, const IndexType b_stride,
    ValueType* __restrict__ c, const IndexType c_stride)
{
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, [](const ValueType& x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const ValueType* __restrict__ alpha,
    const uint8* __restrict__ chk, const size_type* __restrict__ off,
    const uint8* __restrict__ typ, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const ValueType* __restrict__ b,
    const IndexType b_stride, ValueType* __restrict__ c,
    const IndexType c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, [&scale_factor](const ValueType& x) {
                    return scale_factor * x;
                });
}


/**
 * The global function of BCCOO abstract_extract
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param compressed_data  the array where the data are
 * @param block_offsets  the array where the offset of each block is
 * @param compression_types  the array where the type of each block is
 * @param start_cols  the array where the initial column of each block is
 * @param start_rows  the array where the initial row of each block is
 * @param diag  the output dense vector
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_extract(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const uint8* __restrict__ compressed_data,
    const size_type* __restrict__ block_offsets,
    const uint8* __restrict__ compression_types,
    const IndexType* __restrict__ start_cols,
    const IndexType* __restrict__ start_rows, ValueType* __restrict__ diag)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, block_offsets[blk]);
        compr_blk_idxs<IndexType> blk_idxs(start_rows, start_cols,
                                           block_size_local, idxs,
                                           compression_types[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_extract<uint16, uint8>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_extract<uint16, uint16>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else {
                    loop_block_multi_row_extract<uint16, uint32>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_extract<uint8, uint8>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_extract<uint8, uint16>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else {
                    loop_block_multi_row_extract<uint8, uint32>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_extract<uint8>(compressed_data, blk_idxs,
                                                     start_in_blk, jump_in_blk,
                                                     block_size_local, diag);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_extract<uint16>(compressed_data, blk_idxs,
                                                      start_in_blk, jump_in_blk,
                                                      block_size_local, diag);
            } else {
                loop_block_single_row_extract<uint32>(compressed_data, blk_idxs,
                                                      start_in_blk, jump_in_blk,
                                                      block_size_local, diag);
            }
        }
    }
}


/**
 * The global function of BCCOO compute_absolute_inplace
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param compressed_data  the array where the data are
 * @param block_offsets  the array where the offset of each block is
 * @param compression_types  the array where the type of each block is
 * @param start_cols  the array where the initial column of each block is
 * @param start_rows  the array where the initial row of each block is
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_absolute_inplace(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, uint8* __restrict__ compressed_data,
    const size_type* __restrict__ block_offsets,
    const uint8* __restrict__ compression_types,
    const IndexType* __restrict__ start_cols,
    const IndexType* __restrict__ start_rows)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, block_offsets[blk]);
        compr_blk_idxs<IndexType> blk_idxs(start_rows, start_cols,
                                           block_size_local, idxs,
                                           compression_types[blk]);
        loop_block_absolute<IndexType, ValueType>(
            compressed_data, blk_idxs, start_in_blk, jump_in_blk,
            block_size_local, [](ValueType x) { return abs(x); });
    }
}


/**
 * The global function of BCCOO compute_absolute
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param compressed_data_src  the array where the data of source are
 * @param block_offsets_src  the array where the offset of each block of
 * 																	source
 * is
 * @param compression_types_src  the array where the type of each block of
 * 																	source
 * is
 * @param start_cols_src  the array where the initial column of each block of
 * source is
 * @param start_rows_src  the array where the initial row of each block of
 * source is
 * @param compressed_data_res  the array where the data of result are
 * @param block_offsets_res  the array where the offset of each block of result
 * is
 * @param compression_types_res  the array where the sorce type of each block
 * 																	of
 * result is
 * @param start_cols_res  the array where the initial column of each block of
 * result is
 * @param start_rows_res  the array where the initial row of each block of
 * result is
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_absolute(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const uint8* __restrict__ compressed_data_src,
    const size_type* __restrict__ block_offsets_src,
    const uint8* __restrict__ compression_types_src,
    const IndexType* __restrict__ start_cols_src,
    const IndexType* __restrict__ start_rows_src,
    uint8* __restrict__ compressed_data_res,
    size_type* __restrict__ block_offsets_res,
    uint8* __restrict__ compression_types_res,
    IndexType* __restrict__ start_cols_res,
    IndexType* __restrict__ start_rows_res)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    auto comp_abs = [](ValueType x) { return abs(x); };

    if (start_blk == 0 && start_in_blk == 0) {
        block_offsets_res[0] = 0;
    }
    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);

        compr_idxs<IndexType> idxs_src(blk, block_offsets_src[blk]);
        compr_blk_idxs<IndexType> blk_idxs_src(start_rows_src, start_cols_src,
                                               block_size_local, idxs_src,
                                               compression_types_src[blk]);

        start_rows_res[blk] = start_rows_src[blk];
        start_cols_res[blk] = start_cols_src[blk];
        compression_types_res[blk] = compression_types_src[blk];
        size_type block_offsets_res_blk =
            block_offsets_src[blk] -
            ((blk == 0)
                 ? 0
                 : (blk - 1) * block_size *
                       (sizeof(ValueType) - sizeof(remove_complex<ValueType>)));

        compr_idxs<IndexType> idxs_res(blk, block_offsets_res_blk);
        compr_blk_idxs<IndexType> blk_idxs_res(start_rows_res, start_cols_res,
                                               block_size_local, idxs_res,
                                               compression_types_res[blk]);
        if (start_in_blk == 0) {
            block_offsets_res[blk + 1] =
                blk_idxs_res.shf_val +
                block_size_local * sizeof(remove_complex<ValueType>);
        }
        if (blk_idxs_src.is_multi_row()) {
            if (blk_idxs_src.is_row_16bits()) {
                if (blk_idxs_src.is_column_8bits()) {
                    loop_block_multi_row_absolute<uint16, uint8, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                } else if (blk_idxs_src.is_column_16bits()) {
                    loop_block_multi_row_absolute<uint16, uint16, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                } else {
                    loop_block_multi_row_absolute<uint16, uint32, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                }
            } else {
                if (blk_idxs_src.is_column_8bits()) {
                    loop_block_multi_row_absolute<uint8, uint8, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                } else if (blk_idxs_src.is_column_16bits()) {
                    loop_block_multi_row_absolute<uint8, uint16, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                } else {
                    loop_block_multi_row_absolute<uint8, uint32, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        compressed_data_src, blk_idxs_src, start_in_blk,
                        jump_in_blk, block_size_local, compressed_data_res,
                        blk_idxs_res, comp_abs);
                }
            }
        } else {
            if (blk_idxs_src.is_column_8bits()) {
                loop_block_single_row_absolute<uint8, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    compressed_data_src, blk_idxs_src, start_in_blk,
                    jump_in_blk, block_size_local, compressed_data_res,
                    blk_idxs_res, comp_abs);
            } else if (blk_idxs_src.is_column_16bits()) {
                loop_block_single_row_absolute<uint16, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    compressed_data_src, blk_idxs_src, start_in_blk,
                    jump_in_blk, block_size_local, compressed_data_res,
                    blk_idxs_res, comp_abs);
            } else {
                loop_block_single_row_absolute<uint32, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    compressed_data_src, blk_idxs_src, start_in_blk,
                    jump_in_blk, block_size_local, compressed_data_res,
                    blk_idxs_res, comp_abs);
            }
        }
    }
}


/**
 * The global function of BCCOO abstract_fill_in_coo
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param compressed_data  the array where the data are
 * @param block_offsets  the array where the offset of each block is
 * @param compression_types  the array where the type of each block is
 * @param start_cols  the array where the initial column of each block is
 * @param start_rows  the array where the initial row of each block is
 * @param diag  the output dense vector
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void abstract_fill_in_coo(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const uint8* __restrict__ compressed_data,
    const size_type* __restrict__ block_offsets,
    const uint8* __restrict__ compression_types,
    const IndexType* __restrict__ start_cols,
    const IndexType* __restrict__ start_rows, IndexType* __restrict__ rows_idxs,
    IndexType* __restrict__ cols_idxs, ValueType* __restrict__ values)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, block_offsets[blk]);
        compr_blk_idxs<IndexType> blk_idxs(start_rows, start_cols,
                                           block_size_local, idxs,
                                           compression_types[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_coo<uint16, uint8>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_coo<uint16, uint16>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                } else {
                    loop_block_multi_row_fill_in_coo<uint16, uint32>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_coo<uint8, uint8>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_coo<uint8, uint16>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                } else {
                    loop_block_multi_row_fill_in_coo<uint8, uint32>(
                        compressed_data, blk, blk_idxs, start_in_blk,
                        jump_in_blk, block_size, block_size_local, rows_idxs,
                        cols_idxs, values);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_fill_in_coo<uint8>(
                    compressed_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_fill_in_coo<uint16>(
                    compressed_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            } else {
                loop_block_single_row_fill_in_coo<uint32>(
                    compressed_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            }
        }
    }
}


/**
 * The global function of BCCOO abstract_fill_in_dense
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_blks  the number of blocks in the matrix
 * @param block_size  the number of nonzeros in each block
 * @param num_lines  the maximum round of each warp
 * @param compressed_data  the array where the data are
 * @param block_offsets  the array where the offset of each block is
 * @param compression_types  the array where the type of each block is
 * @param start_cols  the array where the initial column of each block is
 * @param start_rows  the array where the initial row of each block is
 * @param diag  the output dense vector
 *
 * @tparam subwarp_size   size of the subwarp
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subwarp_size = config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void abstract_fill_in_dense(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const uint8* __restrict__ compressed_data,
    const size_type* __restrict__ block_offsets,
    const uint8* __restrict__ compression_types,
    const IndexType* __restrict__ start_cols,
    const IndexType* __restrict__ start_rows, IndexType stride,
    ValueType* __restrict__ result)
{
    const IndexType column_id = blockIdx.y;
    const IndexType start_blk = blockIdx.x;
    const IndexType jump_blk = gridDim.x;

    const IndexType start_in_blk = threadIdx.y * subwarp_size + threadIdx.x;
    const IndexType jump_in_blk = blockDim.y * subwarp_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, block_offsets[blk]);
        compr_blk_idxs<IndexType> blk_idxs(start_rows, start_cols,
                                           block_size_local, idxs,
                                           compression_types[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_dense<uint16, uint8>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_dense<uint16, uint16>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else {
                    loop_block_multi_row_fill_in_dense<uint16, uint32>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_dense<uint8, uint8>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_dense<uint8, uint16>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else {
                    loop_block_multi_row_fill_in_dense<uint8, uint32>(
                        compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_fill_in_dense<uint8>(
                    compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_fill_in_dense<uint16>(
                    compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            } else {
                loop_block_single_row_fill_in_dense<uint32>(
                    compressed_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            }
        }
    }
}


}  // namespace kernel
