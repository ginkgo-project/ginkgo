/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

// #include <iostream>

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
/*
void spmv_kernel(const size_type nnz, const size_type num_lines,
                 const ValueType* __restrict__ val,
                 const IndexType* __restrict__ col,
                 const IndexType* __restrict__ row,
                 const ValueType* __restrict__ b, const size_type b_stride,
                 ValueType* __restrict__ c, const size_type c_stride,
                 Closure scale, sycl::nd_item<3> item_ct1)
*/
/*
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, 
                [&scale_factor](const ValueType& x) { return scale_factor * x; },
                item_ct1);
*/
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void spmv_kernel(const size_type nnz, const size_type num_blks,
                            const size_type block_size,
                            const size_type num_lines,
                            const uint8* __restrict__ chunk_data,
                            const IndexType* __restrict__ offsets_data,
                            const uint8* __restrict__ types_data,
                            const IndexType* __restrict__ cols_data,
                            const IndexType* __restrict__ rows_data,
                            const ValueType* __restrict__ b,
                            const size_type b_stride, ValueType* __restrict__ c,
                            const size_type c_stride, Closure scale,
                 						sycl::nd_item<3> item_ct1)
{
/*
    ValueType temp_val = zero<ValueType>();
    const auto start =
        static_cast<size_type>(item_ct1.get_local_range().get(2)) *
            item_ct1.get_group(2) * item_ct1.get_local_range().get(1) *
            num_lines +
        item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2) *
            num_lines;
    const auto column_id = item_ct1.get_group(1);
    size_type num = (nnz > start) * ceildiv(nnz - start, subgroup_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + item_ct1.get_local_id(2);
    const IndexType ind_end = ind_start + (num - 1) * subgroup_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    for (; ind < ind_end; ind += subgroup_size) {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        auto next_row = (ind + subgroup_size < nnz) ? row[ind + subgroup_size]
                                                    : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subgroup_size>(tile_block, curr_row, &temp_val);
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
        bool is_first_in_segment =
            segment_scan<subgroup_size>(tile_block, curr_row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp_val));
        }
    }
*/
/*
    const auto column_id = blockIdx.y;
    const auto start_blk = blockIdx.x;
    const auto jump_blk = gridDim.x;

    const auto start_in_blk = threadIdx.y * subgroup_size + threadIdx.x;
    const auto jump_in_blk = blockDim.y * subgroup_size;

    ValueType temp_val = zero<ValueType>();
    bool new_value = false;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
        const auto tile_block =
            group::tiled_partition<subgroup_size>(group::this_thread_block());

        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        idxs.row = rows_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        size_type last_row =
            idxs.row +
            ((blk_idxs.mul_row)
                 ? get_value_chunk<uint8>(
                       chunk_data, blk_idxs.shf_row + block_size_local - 1)
                 : 0);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
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
                                 get_value_chunk<uint8>(
                                     chunk_data,
                                     blk_idxs.shf_row + pos + jump_in_blk)
                           : last_row)
                    : blk_idxs.row_frs;
            // segmented scan
            if (tile_block.any(idxs.row != next_row)) {
                bool is_first_in_segment = segment_scan<subgroup_size>(
                    tile_block, idxs.row, temp_val,
                    [](ValueType a, ValueType b) { return a + b; });
                if (is_first_in_segment) {
                    atomic_add(&(c[idxs.row * c_stride + column_id]),
                               scale(temp_val));
                }
                temp_val = zero<ValueType>();
                new_value = false;
            }
        }
        // segmented scan
        if (tile_block.any(new_value)) {
            bool is_first_in_segment = segment_scan<subgroup_size>(
                tile_block, idxs.row, temp_val,
                [](ValueType a, ValueType b) { return a + b; });
            if (is_first_in_segment) {
                atomic_add(&(c[idxs.row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
    }
*/
//		if (item_ct1.get_global_id(0) == 0)
/**/
 		if (item_ct1.get_global_linear_id() == 0) {
			sycl::ext::oneapi::experimental::printf("kernel spmv_kernel(%d,%d)\n", subgroup_size, item_ct1.get_sub_group().get_local_range().get(0));
			sycl::ext::oneapi::experimental::printf("%ld - %ld - %d %ld - %ld - %d\n",
				item_ct1.get_local_range(0),
				item_ct1.get_local_range(1),
				item_ct1.get_local_range(2),
				item_ct1.get_global_range(0),
				item_ct1.get_global_range(1),
				item_ct1.get_global_range(2));
//			sycl::ext::oneapi::experimental::printf("%f\n", scale(1.0));
		}
/**/
    const auto column_id = item_ct1.get_group(1); // blockIdx.y;
    const auto start_blk = item_ct1.get_group(2); // blockIdx.x;
//    const auto jump_blk = item_ct1.get_num_group(0); // gridDim.x;
//    const auto jump_blk = item_ct1.get_global_range(2); // gridDim.x;
    const auto jump_blk = item_ct1.get_group_range(2); // gridDim.x;

//    const auto start_in_blk = threadIdx.y * subgroup_size + threadIdx.x;
    const auto start_in_blk = item_ct1.get_local_id(1) * subgroup_size + 
															item_ct1.get_local_id(2);
//    const auto jump_in_blk = blockDim.y * subgroup_size;
    const auto jump_in_blk = item_ct1.get_local_range(1) * subgroup_size;

    ValueType temp_val = zero<ValueType>();
    bool new_value = false;

    for (IndexType blk = start_blk; blk < num_blks; blk += jump_blk) {
    		const auto tile_block = group::tiled_partition<subgroup_size>(
        		group::this_thread_block(item_ct1));

        size_type block_size_local =
            std::min(block_size, nnz - block_size * blk);
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        idxs.blk = blk;
        idxs.shf = offsets_data[blk];
        idxs.row = rows_data[blk];
        init_block_indices(rows_data, cols_data, block_size_local, idxs,
                           types_data[blk], blk_idxs);
        size_type last_row =
            idxs.row +
            ((blk_idxs.mul_row)
                 ? get_value_chunk<uint8>(
                       chunk_data, blk_idxs.shf_row + block_size_local - 1)
                 : 0);
        for (size_type pos = start_in_blk; pos < block_size_local;
             pos += jump_in_blk) {
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
                                 get_value_chunk<uint8>(
                                     chunk_data,
                                     blk_idxs.shf_row + pos + jump_in_blk)
                           : last_row)
                    : blk_idxs.row_frs;
            // segmented scan
            if (tile_block.any(idxs.row != next_row)) {
                bool is_first_in_segment = segment_scan<subgroup_size>(
                    tile_block, idxs.row, &temp_val);
//                    [](ValueType &a, ValueType &b) { return a + b; });
                if (is_first_in_segment) {
/*
										ValueType aux = scale(temp_val);
										sycl::ext::oneapi::experimental::printf("AT1 = (%f,%f)\n", temp_val, aux);
                    atomic_add(&(c[idxs.row * c_stride + column_id]), aux);
										sycl::ext::oneapi::experimental::printf("AT1 = (%f)\n", c[idxs.row * c_stride + column_id]);
*/
                    atomic_add(&(c[idxs.row * c_stride + column_id]), scale(temp_val));
/* */
                }
                temp_val = zero<ValueType>();
                new_value = false;
            }
	
        }
        // segmented scan
        if (tile_block.any(new_value)) {
            bool is_first_in_segment = segment_scan<subgroup_size>(
                tile_block, idxs.row, &temp_val);
//                [](ValueType a, ValueType b) { return a + b; });
            if (is_first_in_segment) {
/* 
								ValueType aux = scale(temp_val);
								sycl::ext::oneapi::experimental::printf("AT2 = (%f,%f)\n", temp_val, aux);
                atomic_add(&(c[idxs.row * c_stride + column_id]), aux);   
								sycl::ext::oneapi::experimental::printf("AT2 = (%f)\n", c[idxs.row * c_stride + column_id]);
*/
                atomic_add(&(c[idxs.row * c_stride + column_id]), scale(temp_val));
/**/
            }
            temp_val = zero<ValueType>();
        }
    }
}


template <typename ValueType, typename IndexType>
/*
void abstract_spmv(const size_type nnz, const size_type num_lines,
                   const ValueType* __restrict__ val,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row,
                   const ValueType* __restrict__ b, const size_type b_stride,
                   ValueType* __restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
        [](const ValueType& x) { return x; }, item_ct1);
}
*/
void abstract_spmv(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const uint8* __restrict__ chk,
    const IndexType* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride,
    sycl::nd_item<3> item_ct1)
{
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, [](const ValueType& x) { return x; },
								item_ct1);
}

/*
            abstract_spmv(bccoo_grid, bccoo_block, 0, exec->get_queue(),
                nnz, num_blocks_matrix, block_size, num_lines,
                (alpha->get_const_values()),
                (a->get_const_chunk()),
                (a->get_const_offsets()),
                (a->get_const_types()),
                (a->get_const_cols()),
                (a->get_const_rows()),
                (b->get_const_values()), b->get_stride(),
                (c->get_values()), c->get_stride());

template <typename ValueType, typename IndexType>
void abstract_spmv(const size_type nnz, const size_type num_lines,
                   const ValueType* __restrict__ alpha,
                   const ValueType* __restrict__ val,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row,
                   const ValueType* __restrict__ b, const size_type b_stride,
                   ValueType* __restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType& x) { return scale_factor * x; },
        item_ct1);
}
*/
template <typename ValueType, typename IndexType>
void abstract_spmv(
    const size_type nnz, const size_type num_blks, const size_type block_size,
    const size_type num_lines, const ValueType* __restrict__ alpha,
		const uint8* __restrict__ chk,
    const IndexType* __restrict__ off, const uint8* __restrict__ typ,
    const IndexType* __restrict__ col, const IndexType* __restrict__ row,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride,
    sycl::nd_item<3> item_ct1)
{
		ValueType scale_factor = alpha[0];
//		if (item_ct1.get_global_linear_id() == 0) {
//			sycl::ext::oneapi::experimental::printf("alpha = %f\n", scale_factor);
//		}
    spmv_kernel(nnz, num_blks, block_size, num_lines, chk, off, typ, col, row,
                b, b_stride, c, c_stride, 
//                [](const ValueType& x) { return x; },
                [](const ValueType& x) { return x + x; },
//								[&scale_factor](const ValueType& x) { return scale_factor * x; },
								item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_spmv, abstract_spmv);


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
void spmm_kernel(const size_type nnz, const size_type num_elems,
                 const ValueType* __restrict__ val,
                 const IndexType* __restrict__ col,
                 const IndexType* __restrict__ row, const size_type num_cols,
                 const ValueType* __restrict__ b, const size_type b_stride,
                 ValueType* __restrict__ c, const size_type c_stride,
                 Closure scale, sycl::nd_item<3> item_ct1)
{
/*
    ValueType temp = zero<ValueType>();
    const auto coo_idx =
        (static_cast<size_type>(item_ct1.get_local_range().get(1)) *
             item_ct1.get_group(2) +
         item_ct1.get_local_id(1)) *
        num_elems;
    const auto column_id =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
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
*/
}


template <typename ValueType, typename IndexType>
void abstract_spmm(const size_type nnz, const size_type num_elems,
                   const ValueType* __restrict__ val,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row, const size_type num_cols,
                   const ValueType* __restrict__ b, const size_type b_stride,
                   ValueType* __restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    spmm_kernel(
        nnz, num_elems, val, col, row, num_cols, b, b_stride, c, c_stride,
        [](const ValueType& x) { return x; }, item_ct1);
}

template <typename ValueType, typename IndexType>
void abstract_spmm(const size_type nnz, const size_type num_elems,
                   const ValueType* __restrict__ alpha,
                   const ValueType* __restrict__ val,
                   const IndexType* __restrict__ col,
                   const IndexType* __restrict__ row, const size_type num_cols,
                   const ValueType* __restrict__ b, const size_type b_stride,
                   ValueType* __restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    ValueType scale_factor = alpha[0];
    spmm_kernel(
        nnz, num_elems, val, col, row, num_cols, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType& x) { return scale_factor * x; },
        item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_spmm, abstract_spmm);


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
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) 
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
           const matrix::Dense<ValueType>* b,
           matrix::Dense<ValueType>* c)
{
/*
    const auto nnz = a->get_num_stored_elements();
    const auto b_ncols = b->get_size()[1];
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        if (b_ncols < 4) {
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            abstract_spmv(coo_grid, coo_block, 0, exec->get_queue(), nnz,
                          num_lines, a->get_const_values(),
                          a->get_const_col_idxs(), a->get_const_row_idxs(),
                          b->get_const_values(), b->get_stride(),
                          c->get_values(), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            abstract_spmm(coo_grid, coo_block, 0, exec->get_queue(), nnz,
                          num_elems, a->get_const_values(),
                          a->get_const_col_idxs(), a->get_const_row_idxs(),
                          b_ncols, b->get_const_values(), b->get_stride(),
                          c->get_values(), c->get_stride());
        }
    }
*/
/* */
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

            abstract_spmv(bccoo_grid, bccoo_block, 0, exec->get_queue(),
                nnz, num_blocks_matrix, block_size, num_lines,
                (a->get_const_chunk()),
                (a->get_const_offsets()),
                (a->get_const_types()),
                (a->get_const_cols()),
                (a->get_const_rows()),
                (b->get_const_values()), b->get_stride(),
                (c->get_values()), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
/* */
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
/*
    const auto nnz = a->get_num_stored_elements();
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto b_ncols = b->get_size()[1];

    if (nwarps > 0) {
        if (b_ncols < 4) {
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            abstract_spmv(coo_grid, coo_block, 0, exec->get_queue(), nnz,
                          num_lines, alpha->get_const_values(),
                          a->get_const_values(), a->get_const_col_idxs(),
                          a->get_const_row_idxs(), b->get_const_values(),
                          b->get_stride(), c->get_values(), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            abstract_spmm(coo_grid, coo_block, 0, exec->get_queue(), nnz,
                          num_elems, alpha->get_const_values(),
                          a->get_const_values(), a->get_const_col_idxs(),
                          a->get_const_row_idxs(), b_ncols,
                          b->get_const_values(), b->get_stride(),
                          c->get_values(), c->get_stride());
        }
    }
*/
/* */
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

            abstract_spmv(bccoo_grid, bccoo_block, 0, exec->get_queue(),
                nnz, num_blocks_matrix, block_size, num_lines,
                (alpha->get_const_values()),
                (a->get_const_chunk()),
                (a->get_const_offsets()),
                (a->get_const_types()),
                (a->get_const_cols()),
                (a->get_const_rows()),
                (b->get_const_values()), b->get_stride(),
                (c->get_values()), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
/* */
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
/*
template <typename ValueType, typename IndexType>
void convert_to_compression(std::shared_ptr<const DpcppExecutor> exec,
                            const matrix::Bccoo<ValueType, IndexType>* source,
                            matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COMPRESSION_KERNEL);
*/
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
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType* idxs, size_type num_nonzeros,
                              IndexType* ptrs,
                              size_type length) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);

template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const DpcppExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      remove_complex<matrix::Bccoo<ValueType, IndexType>>*
                          result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
