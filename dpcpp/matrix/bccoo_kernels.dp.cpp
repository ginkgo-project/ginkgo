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

#include "core/matrix/bccoo_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/dense_kernels.hpp"
//#include "dpcpp/components/format_conversion.dp.hpp"

#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/format_conversion.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/bccoo_helper.dp.hpp"

namespace gko {
namespace kernels {
/**
 * @brief DPCPP namespace.
 *
 * @ingroup dpcpp
 */
namespace dpcpp {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


constexpr int default_block_size = 256;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


namespace {


// #define OLD_BLOCK 1


/**
 * The device function of BCCOO spmv
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
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void spmv_kernel(const size_type nnz, const size_type num_blks,
                 const size_type block_size, const size_type num_lines,
                 const uint8* __restrict__ chunk_data,
                 //                 const IndexType* __restrict__ offsets_data,
                 const size_type* __restrict__ offsets_data,
                 const uint8* __restrict__ types_data,
                 const IndexType* __restrict__ cols_data,
                 const IndexType* __restrict__ rows_data,
                 const ValueType* __restrict__ b, const size_type b_stride,
                 ValueType* __restrict__ c, const size_type c_stride,
                 Closure scale, sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);       // blockIdx.y;
    const auto start_blk = item_ct1.get_group(2);       // blockIdx.x;
    const auto jump_blk = item_ct1.get_group_range(2);  // gridDim.x;

    //    const auto start_in_blk = threadIdx.y * subgroup_size + threadIdx.x;
    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    //    const auto jump_in_blk = blockDim.y * subgroup_size;
    const auto jump_in_blk = item_ct1.get_local_range(1) * subgroup_size;
    /*
                    if (item_ct1.get_global_linear_id() == 0) {
                            sycl::ext::oneapi::experimental::printf("kernel
       spmv_kernel(%d,%d)\n", subgroup_size,
       item_ct1.get_sub_group().get_local_range().get(0));
                            sycl::ext::oneapi::experimental::printf("%ld - %ld -
       %d %ld - %ld - %d\n", item_ct1.get_local_range(0),
                                    item_ct1.get_local_range(1),
                                    item_ct1.get_local_range(2),
                                    item_ct1.get_global_range(0),
                                    item_ct1.get_global_range(1),
                                    item_ct1.get_global_range(2));
                            sycl::ext::oneapi::experimental::printf("%ld  %ld -
       %d - %ld  %ld - %d - %d\n", column_id, start_blk, jump_blk, num_blks,
       start_in_blk, jump_in_blk, block_size);
                    }
    */
    ValueType temp_val = zero<ValueType>();
    bool new_value = false;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        const auto tile_block = group::tiled_partition<subgroup_size>(
            group::this_thread_block(item_ct1));
        /*
        if (item_ct1.get_global_linear_id() ==
           0) { sycl::ext::oneapi::experimental::printf("(X)(%d)\n", blk);
        }
        */
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        idxs.row = rows_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
#ifdef OLD_BLOCK
        size_type last_row =
            idxs.row +
            ((blk_idxs.mul_row)
                 ? ((blk_idxs.row_16bits)
                        ? get_value_chunk<uint16>(
                              chunk_data,
                              blk_idxs.shf_row +
                                  (block_size_local - 1) * sizeof(uint16))
                        : get_value_chunk<uint8>(
                              chunk_data,
                              blk_idxs.shf_row + block_size_local - 1))
                 : 0);
        /*
           if (item_ct1.get_global_linear_id() == 0) {
                sycl::ext::oneapi::experimental::printf("(Y)(%d)\n",
                     block_size_local);
           }
        */
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            /*
             if
               (item_ct1.get_global_linear_id() == 0) {
                 sycl::ext::oneapi::experimental::printf("(Z)(%d)\n", pos);
                                                            }
            */
            // if (item_ct1.get_global_id(2) < block_size_local)
            {
                idxs.row = blk_idxs.row_frs;
                new_value = (pos < block_size_local);
                if (new_value) {
                    ValueType val;
                    get_block_position_value<IndexType, ValueType>(
                        pos, chunk_data, blk_idxs, idxs.row, idxs.col, val);
                    temp_val += val * b[idxs.col * b_stride + column_id];
                } else {
                    temp_val = zero<ValueType>();
                }

                auto next_row =
                    (blk_idxs.mul_row)
                        ? ((pos + jump_in_blk < block_size_local)
                               ? blk_idxs.row_frs +
                                     ((blk_idxs.row_16bits)
                                          ? get_value_chunk<uint16>(
                                                chunk_data,
                                                blk_idxs.shf_row +
                                                    (pos + jump_in_blk) *
                                                        sizeof(uint16))
                                          : get_value_chunk<uint8>(
                                                chunk_data, blk_idxs.shf_row +
                                                                pos +
                                                                jump_in_blk))
                               : last_row)
                        : blk_idxs.row_frs;
                // segmented scan (Fail if some threads are not active in
                // workgroup?)
                if (tile_block.any(idxs.row != next_row)) {
                    bool is_first_in_segment = segment_scan<subgroup_size>(
                        tile_block, idxs.row, &temp_val);
                    //                    [](ValueType &a, ValueType &b) {
                    //                    return a + b; });
                    if (is_first_in_segment) {
                        atomic_add(&(c[idxs.row * c_stride + column_id]),
                                   scale(temp_val));
                    }
                    temp_val = zero<ValueType>();
                    new_value = false;
                }
            }
        }
        // segmented scan
        if (tile_block.any(new_value)) {
            bool is_first_in_segment =
                segment_scan<subgroup_size>(tile_block, idxs.row, &temp_val);
            //                [](ValueType a, ValueType b) { return a + b; });
            if (is_first_in_segment) {
                atomic_add(&(c[idxs.row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
#else
        if (blk_idxs.mul_row) {
            if (blk_idxs.row_16bits) {
                if (blk_idxs.col_8bits) {
                    loop_block_multi_row<subgroup_size, uint16, uint8,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else if (blk_idxs.col_16bits) {
                    loop_block_multi_row<subgroup_size, uint16, uint16,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else {
                    loop_block_multi_row<subgroup_size, uint16, uint32,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                }
            } else {
                if (blk_idxs.col_8bits) {
                    loop_block_multi_row<subgroup_size, uint8, uint8,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else if (blk_idxs.col_16bits) {
                    loop_block_multi_row<subgroup_size, uint8, uint16,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else {
                    loop_block_multi_row<subgroup_size, uint8, uint32,
                                         ValueType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                }
            }
        } else {
            if (blk_idxs.col_8bits) {
                loop_block_single_row<subgroup_size, uint8, ValueType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            } else if (blk_idxs.col_16bits) {
                loop_block_single_row<subgroup_size, uint16, ValueType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            } else {
                loop_block_single_row<subgroup_size, uint32, ValueType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            }
        }
#endif
    }
}


template <typename ValueType, typename IndexType>
void abstract_spmv(const size_type nnz, const size_type num_blks,
                   const size_type block_size, const size_type num_lines,
                   const uint8* __restrict__ chk,
                   //                   const IndexType* __restrict__ off,
                   const size_type* __restrict__ off,
                   const uint8* __restrict__ typ,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row,
                   const ValueType* __restrict__ b, const size_type b_stride,
                   ValueType* __restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    /*
                    if (item_ct1.get_global_linear_id() == 0) {
                            sycl::ext::oneapi::experimental::printf("NNZ(%d)\n",
       nnz);
                    }
    */
    spmv_kernel(
        nnz, num_blks, block_size, num_lines, chk, off, typ, col, row, b,
        b_stride, c, c_stride, [](const ValueType& x) { return x; }, item_ct1);
}


template <typename ValueType, typename IndexType>
void abstract_spmv(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const ValueType* __restrict__ alpha,
    //    const uint8* __restrict__ chk, const IndexType* __restrict__ off,
    const uint8* __restrict__ chk, const size_type* __restrict__ off,
    const uint8* __restrict__ typ, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride, sycl::nd_item<3> item_ct1)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_blks, block_size, num_lines, chk, off, typ, col, row, b,
        b_stride, c, c_stride,
        [scale_factor](const ValueType& x) { return scale_factor * x; },
        item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_spmv, abstract_spmv);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void fill_in_coo(const size_type nnz, const size_type num_blks,
                 const size_type block_size, const size_type num_lines,
                 const uint8* __restrict__ chunk_data,
                 //                 const IndexType* __restrict__ offsets_data,
                 const size_type* __restrict__ offsets_data,
                 const uint8* __restrict__ types_data,
                 const IndexType* __restrict__ cols_data,
                 const IndexType* __restrict__ rows_data,
                 IndexType* __restrict__ rows_idxs,
                 IndexType* __restrict__ cols_idxs,
                 ValueType* __restrict__ values, sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);
    const auto start_blk = item_ct1.get_group(2);
    const auto jump_blk = item_ct1.get_group_range(2);

    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const auto jump_in_blk = item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};

        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs.row, idxs.col, val);
                auto index = blk * block_size + pos;
                rows_idxs[index] = idxs.row;
                cols_idxs[index] = idxs.col;
                values[index] = val;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(fill_in_coo, fill_in_coo);

template <typename IndexType>
void convert_row_idxs_to_ptrs(const IndexType* __restrict__ idxs,
                              size_type num_nonzeros,
                              IndexType* __restrict__ ptrs, size_type length,
                              sycl::nd_item<3> item_ct1)
{
    const auto tidx = item_ct1.get_global_id(2);
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

GKO_ENABLE_DEFAULT_HOST(convert_row_idxs_to_ptrs, convert_row_idxs_to_ptrs);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void fill_in_dense(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chunk_data,
    //                   const IndexType* __restrict__ offsets_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, size_type stride,
    ValueType* __restrict__ result, sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);
    const auto start_blk = item_ct1.get_group(2);
    const auto jump_blk = item_ct1.get_group_range(2);

    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const auto jump_in_blk = item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};

        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs.row, idxs.col, val);
                result[idxs.row * stride + idxs.col] = val;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(fill_in_dense, fill_in_dense);


template <typename ValueType>
void initialize_zero_dense(size_type num_rows, size_type num_cols,
                           size_type stride, ValueType* __restrict__ result,
                           sycl::nd_item<3> item_ct1)
{
    const auto tidx_x = item_ct1.get_global_id(2);
    const auto tidx_y = item_ct1.get_global_id(1);
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}

GKO_ENABLE_DEFAULT_HOST(initialize_zero_dense, initialize_zero_dense);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void extract_kernel(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chunk_data,
    //                    const IndexType* __restrict__ offsets_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, ValueType* __restrict__ diag,
    sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);
    const auto start_blk = item_ct1.get_group(2);
    const auto jump_blk = item_ct1.get_group_range(2);

    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const auto jump_in_blk = item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};

        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value<IndexType, ValueType>(
                    pos, chunk_data, blk_idxs, idxs.row, idxs.col, val);
                if (idxs.row == idxs.col) diag[idxs.col] = val;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(extract_kernel, extract_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void absolute_inplace_kernel(
    const ValueType oldval, const size_type nnz, const size_type num_blks,
    const size_type block_size, const size_type num_lines,
    uint8* __restrict__ chunk_data,
    //                             const IndexType* __restrict__ offsets_data,
    const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, Closure comp_abs,
    sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);
    const auto start_blk = item_ct1.get_group(2);
    const auto jump_blk = item_ct1.get_group_range(2);

    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const auto jump_in_blk = item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};

        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value_put<IndexType, ValueType, Closure>(
                    pos, chunk_data, blk_idxs, idxs.row, idxs.col, val,
                    comp_abs);
            }
        }
    }
}

// GKO_ENABLE_DEFAULT_HOST(absolute_inplace_kernel, absolute_inplace_kernel);


template <typename ValueType, typename IndexType>
void abstract_absolute_inplace(
    const ValueType val, const size_type nnz, const size_type num_blks,
    const size_type block_size, const size_type num_lines,
    //    uint8* __restrict__ chk, const IndexType* __restrict__ off,
    uint8* __restrict__ chk, const size_type* __restrict__ off,
    const uint8* __restrict__ typ, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, sycl::nd_item<3> item_ct1)
{
    absolute_inplace_kernel(val, nnz, num_blks, block_size, num_lines, chk, off,
                            typ, col, row, ([](ValueType x) { return abs(x); }),
                            item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_absolute_inplace, abstract_absolute_inplace);

template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void absolute_kernel(
    ValueType val, const size_type nnz, const size_type num_blks,
    const size_type block_size, const size_type num_lines,
    const uint8* __restrict__ chunk_data_src,
    //                     const IndexType* __restrict__ offsets_data_src,
    const size_type* __restrict__ offsets_data_src,
    const uint8* __restrict__ types_data_src,
    const IndexType* __restrict__ cols_data_src,
    const IndexType* __restrict__ rows_data_src,
    uint8* __restrict__ chunk_data_res,
    //                     IndexType* __restrict__ offsets_data_res,
    size_type* __restrict__ offsets_data_res,
    uint8* __restrict__ types_data_res, IndexType* __restrict__ cols_data_res,
    IndexType* __restrict__ rows_data_res, Closure comp_abs,
    sycl::nd_item<3> item_ct1)
{
    const auto column_id = item_ct1.get_group(1);
    const auto start_blk = item_ct1.get_group(2);
    const auto jump_blk = item_ct1.get_group_range(2);

    const auto start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const auto jump_in_blk = item_ct1.get_local_range().get(1) * subgroup_size;
    /*
                    if (item_ct1.get_global_linear_id() == 0) {
                            sycl::ext::oneapi::experimental::printf("kernel
    absolute_kernel(%d,%d)\n", subgroup_size,
    item_ct1.get_sub_group().get_local_range().get(0));
                            sycl::ext::oneapi::experimental::printf("%ld - %ld -
    %d %ld - %ld - %d\n", item_ct1.get_local_range(0),
                                    item_ct1.get_local_range(1),
                                    item_ct1.get_local_range(2),
                                    item_ct1.get_global_range(0),
                                    item_ct1.get_global_range(1),
                                    item_ct1.get_global_range(2));
                            sycl::ext::oneapi::experimental::printf("%ld  %ld -
    %d - %ld  %ld - %d - %d\n", column_id, start_blk, jump_blk, num_blks,
    start_in_blk, jump_in_blk, block_size);
    //      sycl::ext::oneapi::experimental::printf("%f\n",
    scale(1.0));
                    }
    */
    offsets_data_res[0] = 0;
    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);

        compr_idxs idxs_src = {};
        compr_blk_idxs blk_idxs_src = {};
        idxs_src.blk = blk;
        idxs_src.shf = offsets_data_src[blk];
        init_block_indices(rows_data_src, cols_data_src, block_size_local,
                           idxs_src, types_data_src[blk], blk_idxs_src);

        rows_data_res[blk] = rows_data_src[blk];
        cols_data_res[blk] = cols_data_src[blk];
        types_data_res[blk] = types_data_src[blk];
        offsets_data_res[blk] =
            offsets_data_src[blk] -
            ((blk == 0)
                 ? 0
                 : (blk - 1) * block_size *
                       (sizeof(ValueType) - sizeof(remove_complex<ValueType>)));

        compr_idxs idxs_res = {};
        compr_blk_idxs blk_idxs_res = {};
        idxs_res.blk = blk;
        idxs_res.shf = offsets_data_res[blk];
        init_block_indices(rows_data_res, cols_data_res, block_size_local,
                           idxs_res, types_data_res[blk], blk_idxs_res);
        offsets_data_res[blk + 1] =
            blk_idxs_res.shf_val +
            block_size_local * sizeof(remove_complex<ValueType>);

        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
            if (pos < block_size_local) {
                ValueType val;
                get_block_position_value_put<
                    IndexType, ValueType, remove_complex<ValueType>, Closure>(
                    pos, chunk_data_src, blk_idxs_src, chunk_data_res,
                    blk_idxs_res, idxs_src.row, idxs_src.col, val, comp_abs);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void abstract_absolute(
    ValueType val, const size_type nnz, const size_type num_blks,
    const size_type block_size, const size_type num_lines,
    //    const uint8* __restrict__ chk_src, const IndexType* __restrict__
    //    off_src,
    const uint8* __restrict__ chk_src, const size_type* __restrict__ off_src,
    const uint8* __restrict__ typ_src, const IndexType* __restrict__ col_src,
    const IndexType* __restrict__ row_src, uint8* __restrict__ chk_res,
    //    IndexType* __restrict__ off_res, uint8* __restrict__ typ_res,
    size_type* __restrict__ off_res, uint8* __restrict__ typ_res,
    IndexType* __restrict__ col_res, IndexType* __restrict__ row_res,
    sycl::nd_item<3> item_ct1)
{
    absolute_kernel<config::warp_size, ValueType, IndexType>(
        val, nnz, num_blks, block_size, num_lines, chk_src, off_src, typ_src,
        col_src, row_src, chk_res, off_res, typ_res, col_res, row_res,
        [](ValueType x) { return abs(x); }, item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_absolute, abstract_absolute);

}  // namespace


void get_default_block_size(std::shared_ptr<const DpcppExecutor> exec,
                            size_type* block_size)
{
    *block_size = 32;
}


void get_default_compression(std::shared_ptr<const DpcppExecutor> exec,
                             matrix::bccoo::compression* compression)
{
    *compression = matrix::bccoo::compression::block;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto block_size = a->get_block_size();
    const auto num_blocks_matrix = a->get_num_blocks();
    const auto b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            abstract_spmv(bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                          num_blocks_matrix, block_size, num_lines,
                          (a->get_const_chunk()), (a->get_const_offsets()),
                          (a->get_const_types()), (a->get_const_cols()),
                          (a->get_const_rows()), (b->get_const_values()),
                          b->get_stride(), (c->get_values()), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto block_size = a->get_block_size();
    const auto num_blocks_matrix = a->get_num_blocks();
    const auto b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            abstract_spmv(bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                          num_blocks_matrix, block_size, num_lines,
                          (alpha->get_const_values()), (a->get_const_chunk()),
                          (a->get_const_offsets()), (a->get_const_types()),
                          (a->get_const_cols()), (a->get_const_rows()),
                          (b->get_const_values()), b->get_stride(),
                          (c->get_values()), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::bccoo::compression commpress_res,
                    const size_type block_size_res,
                    size_type* mem_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    const auto nnz = source->get_num_stored_elements();

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    const auto block_size = source->get_block_size();
    const auto num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            fill_in_coo(bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                        num_blocks_matrix, block_size, num_lines,
                        (source->get_const_chunk()),
                        (source->get_const_offsets()),
                        (source->get_const_types()), (source->get_const_cols()),
                        (source->get_const_rows()), (result->get_row_idxs()),
                        (result->get_col_idxs()), (result->get_values()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType* idxs, size_type num_nonzeros,
                              IndexType* ptrs, size_type length)
{
    const auto grid_dim = ceildiv(num_nonzeros, default_block_size);
    const dim3 bccoo_grid(grid_dim, 1);
    const dim3 bccoo_block(default_block_size, 1, 1);

    convert_row_idxs_to_ptrs(bccoo_grid, bccoo_block, 0, exec->get_queue(),
                             (idxs), num_nonzeros, (ptrs), length);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const auto nnz = source->get_num_stored_elements();
    const auto num_rows = source->get_size()[0];

    array<IndexType> row_idxs(exec, nnz);

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    const auto block_size = source->get_block_size();
    const auto num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            fill_in_coo(bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                        num_blocks_matrix, block_size, num_lines,
                        (source->get_const_chunk()),
                        (source->get_const_offsets()),
                        (source->get_const_types()), (source->get_const_cols()),
                        (source->get_const_rows()), (row_idxs.get_data()),
                        (result->get_col_idxs()), (result->get_values()));

            convert_row_idxs_to_ptrs(exec, row_idxs.get_data(), nnz, row_ptrs,
                                     num_rows + 1);
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();
    const auto block_size = source->get_block_size();
    const auto num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    const dim3 block_size_mat(config::warp_size,
                              config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size_mat.x),
                             ceildiv(num_rows, block_size_mat.y), 1);
    initialize_zero_dense(init_grid_dim, block_size_mat, 0, exec->get_queue(),
                          num_rows, num_cols, stride, (result->get_values()));

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            fill_in_dense(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                (source->get_const_chunk()), (source->get_const_offsets()),
                (source->get_const_types()), (source->get_const_cols()),
                (source->get_const_rows()), stride, (result->get_values()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto nnz = orig->get_num_stored_elements();
    const auto block_size = orig->get_block_size();
    const auto num_blocks_matrix = orig->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (orig->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            int num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            extract_kernel(bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                           num_blocks_matrix, block_size, num_lines,
                           (orig->get_const_chunk()),
                           (orig->get_const_offsets()),
                           (orig->get_const_types()), (orig->get_const_cols()),
                           (orig->get_const_rows()), (diag->get_values()));
        } else {
            GKO_NOT_SUPPORTED(orig);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const DpcppExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    const auto nnz = matrix->get_num_stored_elements();
    const auto block_size = matrix->get_block_size();
    const auto num_blocks_matrix = matrix->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (matrix->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            auto num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);
            ValueType val = {};  // Use to help compiler to interpret template

            abstract_absolute_inplace(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), val, nnz,
                num_blocks_matrix, block_size, num_lines, (matrix->get_chunk()),
                (matrix->get_const_offsets()), (matrix->get_const_types()),
                (matrix->get_const_cols()), (matrix->get_const_rows()));
        } else {
            GKO_NOT_SUPPORTED(matrix);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
{
    const auto nnz = source->get_num_stored_elements();
    const auto block_size = source->get_block_size();
    const auto num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            int num_blocks_grid = std::min(
                num_blocks_matrix, (size_type)ceildiv(nwarps, warps_in_block));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            auto num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);
            ValueType val = {};  // Use to help compiler to interpret template

            abstract_absolute(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), val, nnz,
                num_blocks_matrix, block_size, num_lines,
                (source->get_const_chunk()), (source->get_const_offsets()),
                (source->get_const_types()), (source->get_const_cols()),
                (source->get_const_rows()), (result->get_chunk()),
                (result->get_offsets()), (result->get_types()),
                (result->get_cols()), (result->get_rows()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
