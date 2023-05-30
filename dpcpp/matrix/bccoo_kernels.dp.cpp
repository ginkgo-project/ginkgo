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
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/dense_kernels.hpp"
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


using namespace matrix::bccoo;


constexpr int default_block_size = 256;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


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
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void spmv_kernel(const IndexType nnz, const IndexType num_blks,
                 const IndexType block_size, const IndexType num_lines,
                 const uint8* __restrict__ chunk_data,
                 const size_type* __restrict__ offsets_data,
                 const uint8* __restrict__ types_data,
                 const IndexType* __restrict__ cols_data,
                 const IndexType* __restrict__ rows_data,
                 const ValueType* __restrict__ b, const IndexType b_stride,
                 ValueType* __restrict__ c, const IndexType c_stride,
                 Closure scale, sycl::nd_item<3> item_ct1)
{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk = item_ct1.get_local_range(1) * subgroup_size;
    ValueType temp_val = zero<ValueType>();
    bool new_value = false;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        const auto tile_block = group::tiled_partition<subgroup_size>(
            group::this_thread_block(item_ct1));
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_spmv<uint16, uint8, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_spmv<uint16, uint16, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else {
                    loop_block_multi_row_spmv<uint16, uint32, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_spmv<uint8, uint8, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_spmv<uint8, uint16, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                } else {
                    loop_block_multi_row_spmv<uint8, uint32, IndexType>(
                        chunk_data, block_size_local, b, b_stride, column_id, c,
                        c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk,
                        scale, item_ct1);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_spmv<uint8, IndexType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_spmv<uint16, IndexType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            } else {
                loop_block_single_row_spmv<uint32, IndexType>(
                    chunk_data, block_size_local, b, b_stride, column_id, c,
                    c_stride, idxs, blk_idxs, start_in_blk, jump_in_blk, scale,
                    item_ct1);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void abstract_spmv(const IndexType nnz, const IndexType num_blks,
                   const IndexType block_size, const IndexType num_lines,
                   const uint8* __restrict__ chk,
                   const size_type* __restrict__ off,
                   const uint8* __restrict__ typ,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row,
                   const ValueType* __restrict__ b, const IndexType b_stride,
                   ValueType* __restrict__ c, const IndexType c_stride,
                   sycl::nd_item<3> item_ct1)
{
    spmv_kernel(
        nnz, num_blks, block_size, num_lines, chk, off, typ, col, row, b,
        b_stride, c, c_stride, [](const ValueType& x) { return x; }, item_ct1);
}


template <typename ValueType, typename IndexType>
void abstract_spmv(
    const IndexType nnz, const IndexType num_blks, const IndexType block_size,
    const IndexType num_lines, const ValueType* __restrict__ alpha,
    const uint8* __restrict__ chk, const size_type* __restrict__ off,
    const uint8* __restrict__ typ, const IndexType* __restrict__ col,
    const IndexType* __restrict__ row, const ValueType* __restrict__ b,
    const IndexType b_stride, ValueType* __restrict__ c,
    const IndexType c_stride, sycl::nd_item<3> item_ct1)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_blks, block_size, num_lines, chk, off, typ, col, row, b,
        b_stride, c, c_stride,
        [scale_factor](const ValueType& x) { return scale_factor * x; },
        item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_spmv, abstract_spmv);


/**
 * The global function of BCCOO abstract_extract
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
 * @tparam subgroup_size   size of the subgroup
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void abstract_extract(const IndexType nnz, const IndexType num_blks,
                      const IndexType block_size, const IndexType num_lines,
                      const uint8* __restrict__ chunk_data,
                      const size_type* __restrict__ offsets_data,
                      const uint8* __restrict__ types_data,
                      const IndexType* __restrict__ cols_data,
                      const IndexType* __restrict__ rows_data,
                      ValueType* __restrict__ diag, sycl::nd_item<3> item_ct1)
{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk =
        item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_extract<uint16, uint8>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_extract<uint16, uint16>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else {
                    loop_block_multi_row_extract<uint16, uint32>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_extract<uint8, uint8>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_extract<uint8, uint16>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                } else {
                    loop_block_multi_row_extract<uint8, uint32>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, diag);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_extract<uint8>(chunk_data, blk_idxs,
                                                     start_in_blk, jump_in_blk,
                                                     block_size_local, diag);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_extract<uint16>(chunk_data, blk_idxs,
                                                      start_in_blk, jump_in_blk,
                                                      block_size_local, diag);
            } else {
                loop_block_single_row_extract<uint32>(chunk_data, blk_idxs,
                                                      start_in_blk, jump_in_blk,
                                                      block_size_local, diag);
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(abstract_extract, abstract_extract);


/**
 * The global function of BCCOO abstract_absolute_inplace
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
 * @param comp_abs  the function to apply to obtain the solution
 *
 * @tparam subgroup_size   size of the subgroup
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 * @tparam Closure  type of the function used to write the result
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void abstract_absolute_inplace(
    const ValueType val, const IndexType nnz, const IndexType num_blks,
    const IndexType block_size, const IndexType num_lines,
    uint8* __restrict__ chunk_data, const size_type* __restrict__ offsets_data,
    const uint8* __restrict__ types_data,
    const IndexType* __restrict__ cols_data,
    const IndexType* __restrict__ rows_data, sycl::nd_item<3> item_ct1)

{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk =
        item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        loop_block_absolute<IndexType, ValueType>(
            chunk_data, blk_idxs, start_in_blk, jump_in_blk, block_size_local,
            [](ValueType x) { return abs(x); });
    }
}

GKO_ENABLE_DEFAULT_HOST(abstract_absolute_inplace, abstract_absolute_inplace);


/**
 * The global function of BCCOO abstract_absolute
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
 * @param comp_abs  the function to apply to obtain the solution
 *
 * @tparam subgroup_size   size of the subgroup
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 * @tparam Closure  type of the function used to write the result
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void abstract_absolute(const ValueType val, const IndexType nnz,
                       const IndexType num_blks, const IndexType block_size,
                       const IndexType num_lines,
                       const uint8* __restrict__ chunk_data_src,
                       const size_type* __restrict__ offsets_data_src,
                       const uint8* __restrict__ types_data_src,
                       const IndexType* __restrict__ cols_data_src,
                       const IndexType* __restrict__ rows_data_src,
                       uint8* __restrict__ chunk_data_res,
                       size_type* __restrict__ offsets_data_res,
                       uint8* __restrict__ types_data_res,
                       IndexType* __restrict__ cols_data_res,
                       IndexType* __restrict__ rows_data_res,
                       sycl::nd_item<3> item_ct1)
{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk =
        item_ct1.get_local_range().get(1) * subgroup_size;

    auto comp_abs = [](ValueType x) { return abs(x); };

    if (start_blk == 0 && start_in_blk == 0) {
        offsets_data_res[0] = 0;
    }
    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);

        compr_idxs<IndexType> idxs_src(blk, offsets_data_src[blk]);
        compr_blk_idxs<IndexType> blk_idxs_src(rows_data_src, cols_data_src,
                                               block_size_local, idxs_src,
                                               types_data_src[blk]);

        rows_data_res[blk] = rows_data_src[blk];
        cols_data_res[blk] = cols_data_src[blk];
        types_data_res[blk] = types_data_src[blk];
        size_type offsets_data_res_blk =
            offsets_data_src[blk] -
            ((blk == 0)
                 ? 0
                 : (blk - 1) * block_size *
                       (sizeof(ValueType) - sizeof(remove_complex<ValueType>)));

        compr_idxs<IndexType> idxs_res(blk, offsets_data_res_blk);
        compr_blk_idxs<IndexType> blk_idxs_res(rows_data_res, cols_data_res,
                                               block_size_local, idxs_res,
                                               types_data_res[blk]);
        if (start_in_blk == 0) {
            offsets_data_res[blk + 1] =
                blk_idxs_res.shf_val +
                block_size_local * sizeof(remove_complex<ValueType>);
        }
        if (blk_idxs_src.is_multi_row()) {
            if (blk_idxs_src.is_row_16bits()) {
                if (blk_idxs_src.is_column_8bits()) {
                    loop_block_multi_row_absolute<uint16, uint8, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                } else if (blk_idxs_src.is_column_16bits()) {
                    loop_block_multi_row_absolute<uint16, uint16, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                } else {
                    loop_block_multi_row_absolute<uint16, uint32, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                }
            } else {
                if (blk_idxs_src.is_column_8bits()) {
                    loop_block_multi_row_absolute<uint8, uint8, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                } else if (blk_idxs_src.is_column_16bits()) {
                    loop_block_multi_row_absolute<uint8, uint16, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                } else {
                    loop_block_multi_row_absolute<uint8, uint32, IndexType,
                                                  ValueType,
                                                  remove_complex<ValueType>>(
                        chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                        block_size_local, chunk_data_res, blk_idxs_res,
                        comp_abs);
                }
            }
        } else {
            if (blk_idxs_src.is_column_8bits()) {
                loop_block_single_row_absolute<uint8, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                    block_size_local, chunk_data_res, blk_idxs_res, comp_abs);
            } else if (blk_idxs_src.is_column_16bits()) {
                loop_block_single_row_absolute<uint16, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                    block_size_local, chunk_data_res, blk_idxs_res, comp_abs);
            } else {
                loop_block_single_row_absolute<uint32, IndexType, ValueType,
                                               remove_complex<ValueType>>(
                    chunk_data_src, blk_idxs_src, start_in_blk, jump_in_blk,
                    block_size_local, chunk_data_res, blk_idxs_res, comp_abs);
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(abstract_absolute, abstract_absolute);


/**
 * The global function of BCCOO abstract_fill_in_coo
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
 * @tparam subgroup_size   size of the subgroup
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void abstract_fill_in_coo(const IndexType nnz, const IndexType num_blks,
                          const IndexType block_size, const IndexType num_lines,
                          const uint8* __restrict__ chunk_data,
                          const size_type* __restrict__ offsets_data,
                          const uint8* __restrict__ types_data,
                          const IndexType* __restrict__ cols_data,
                          const IndexType* __restrict__ rows_data,
                          IndexType* __restrict__ rows_idxs,
                          IndexType* __restrict__ cols_idxs,
                          ValueType* __restrict__ values,
                          sycl::nd_item<3> item_ct1)
{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk =
        item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_coo<uint16, uint8>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_coo<uint16, uint16>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                } else {
                    loop_block_multi_row_fill_in_coo<uint16, uint32>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_coo<uint8, uint8>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_coo<uint8, uint16>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                } else {
                    loop_block_multi_row_fill_in_coo<uint8, uint32>(
                        chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                        block_size, block_size_local, rows_idxs, cols_idxs,
                        values);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_fill_in_coo<uint8>(
                    chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_fill_in_coo<uint16>(
                    chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            } else {
                loop_block_single_row_fill_in_coo<uint32>(
                    chunk_data, blk, blk_idxs, start_in_blk, jump_in_blk,
                    block_size, block_size_local, rows_idxs, cols_idxs, values);
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(abstract_fill_in_coo, abstract_fill_in_coo);


/**
 * The global function of BCCOO abstract_fill_in_dense
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
 * @tparam subgroup_size   size of the subgroup
 * @tparam ValueType  type of values stored in the matrix
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void abstract_fill_in_dense(const IndexType nnz, const IndexType num_blks,
                            const IndexType block_size,
                            const IndexType num_lines,
                            const uint8* __restrict__ chunk_data,
                            const size_type* __restrict__ offsets_data,
                            const uint8* __restrict__ types_data,
                            const IndexType* __restrict__ cols_data,
                            const IndexType* __restrict__ rows_data,
                            IndexType stride, ValueType* __restrict__ result,
                            sycl::nd_item<3> item_ct1)
{
    const IndexType column_id = item_ct1.get_group(1);
    const IndexType start_blk = item_ct1.get_group(2);
    const IndexType jump_blk = item_ct1.get_group_range(2);

    const IndexType start_in_blk =
        item_ct1.get_local_id(1) * subgroup_size + item_ct1.get_local_id(2);
    const IndexType jump_in_blk =
        item_ct1.get_local_range().get(1) * subgroup_size;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        IndexType block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
        compr_blk_idxs<IndexType> blk_idxs(
            rows_data, cols_data, block_size_local, idxs, types_data[blk]);
        if (blk_idxs.is_multi_row()) {
            if (blk_idxs.is_row_16bits()) {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_dense<uint16, uint8>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_dense<uint16, uint16>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else {
                    loop_block_multi_row_fill_in_dense<uint16, uint32>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                }
            } else {
                if (blk_idxs.is_column_8bits()) {
                    loop_block_multi_row_fill_in_dense<uint8, uint8>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else if (blk_idxs.is_column_16bits()) {
                    loop_block_multi_row_fill_in_dense<uint8, uint16>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                } else {
                    loop_block_multi_row_fill_in_dense<uint8, uint32>(
                        chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                        block_size_local, stride, result);
                }
            }
        } else {
            if (blk_idxs.is_column_8bits()) {
                loop_block_single_row_fill_in_dense<uint8>(
                    chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            } else if (blk_idxs.is_column_16bits()) {
                loop_block_single_row_fill_in_dense<uint16>(
                    chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            } else {
                loop_block_single_row_fill_in_dense<uint32>(
                    chunk_data, blk_idxs, start_in_blk, jump_in_blk,
                    block_size_local, stride, result);
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(abstract_fill_in_dense, abstract_fill_in_dense);


}  // namespace kernel


template <typename IndexType>
void get_default_block_size(std::shared_ptr<const DpcppExecutor> exec,
                            IndexType* block_size)
{
    *block_size = 32;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_DEFAULT_BLOCK_SIZE_KERNEL);


void get_default_compression(std::shared_ptr<const DpcppExecutor> exec,
                             compression* compression)
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
    const IndexType nnz = a->get_num_stored_elements();
    const IndexType block_size = a->get_block_size();
    const IndexType num_blocks_matrix = a->get_num_blocks();
    const IndexType b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_spmv(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines, a->get_const_chunk(),
                a->get_const_offsets(), a->get_const_types(),
                a->get_const_cols(), a->get_const_rows(), b->get_const_values(),
                static_cast<IndexType>(b->get_stride()), c->get_values(),
                static_cast<IndexType>(c->get_stride()));
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
    const IndexType nnz = a->get_num_stored_elements();
    const IndexType block_size = a->get_block_size();
    const IndexType num_blocks_matrix = a->get_num_blocks();
    const IndexType b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_spmv(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                alpha->get_const_values(), a->get_const_chunk(),
                a->get_const_offsets(), a->get_const_types(),
                a->get_const_cols(), a->get_const_rows(), b->get_const_values(),
                static_cast<IndexType>(b->get_stride()), c->get_values(),
                static_cast<IndexType>(c->get_stride()));
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
                    compression commpress_res, const IndexType block_size_res,
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
    const IndexType nnz = source->get_num_stored_elements();

    IndexType* row_idxs = result->get_row_idxs();
    IndexType* col_idxs = result->get_col_idxs();
    ValueType* values = result->get_values();

    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_coo(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                source->get_const_chunk(), source->get_const_offsets(),
                source->get_const_types(), source->get_const_cols(),
                source->get_const_rows(), result->get_row_idxs(),
                result->get_col_idxs(), result->get_values());
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const IndexType nnz = source->get_num_stored_elements();
    const IndexType num_rows = source->get_size()[0];

    array<IndexType> row_idxs(exec, nnz);

    IndexType* row_ptrs = result->get_row_ptrs();
    IndexType* col_idxs = result->get_col_idxs();
    ValueType* values = result->get_values();

    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_coo(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                source->get_const_chunk(), source->get_const_offsets(),
                source->get_const_types(), source->get_const_cols(),
                source->get_const_rows(), row_idxs.get_data(),
                result->get_col_idxs(), result->get_values());

            components::convert_idxs_to_ptrs(exec, row_idxs.get_data(), nnz,
                                             num_rows + 1, row_ptrs);
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
    const IndexType num_rows = result->get_size()[0];
    const IndexType num_cols = result->get_size()[1];
    const IndexType stride = result->get_stride();

    const IndexType nnz = source->get_num_stored_elements();
    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    const dim3 block_size_mat(config::warp_size,
                              config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size_mat.x),
                             ceildiv(num_rows, block_size_mat.y), 1);
    dense::fill(exec, result, zero<ValueType>());

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_dense(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                source->get_const_chunk(), source->get_const_offsets(),
                source->get_const_types(), source->get_const_cols(),
                source->get_const_rows(), stride, result->get_values());
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
    const IndexType nnz = orig->get_num_stored_elements();
    const IndexType block_size = orig->get_block_size();
    const IndexType num_blocks_matrix = orig->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (orig->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_extract(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), nnz,
                num_blocks_matrix, block_size, num_lines,
                orig->get_const_chunk(), orig->get_const_offsets(),
                orig->get_const_types(), orig->get_const_cols(),
                orig->get_const_rows(), diag->get_values());
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
    const IndexType nnz = matrix->get_num_stored_elements();
    const IndexType block_size = matrix->get_block_size();
    const IndexType num_blocks_matrix = matrix->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (matrix->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);
            // Use it to help compiler to interpret the template
            ValueType val = {};

            kernel::abstract_absolute_inplace(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), val, nnz,
                num_blocks_matrix, block_size, num_lines, matrix->get_chunk(),
                matrix->get_const_offsets(), matrix->get_const_types(),
                matrix->get_const_cols(), matrix->get_const_rows());
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
    const IndexType nnz = source->get_num_stored_elements();
    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_block_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);
            // Use it to help compiler to interpret the template
            ValueType val = {};

            kernel::abstract_absolute(
                bccoo_grid, bccoo_block, 0, exec->get_queue(), val, nnz,
                num_blocks_matrix, block_size, num_lines,
                source->get_const_chunk(), source->get_const_offsets(),
                source->get_const_types(), source->get_const_cols(),
                source->get_const_rows(), result->get_chunk(),
                result->get_offsets(), result->get_types(), result->get_cols(),
                result->get_rows());
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
