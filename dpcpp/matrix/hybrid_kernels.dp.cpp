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

#include "core/matrix/hybrid_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/format_conversion.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Hybrid matrix format namespace.
 *
 * @ingroup hybrid
 */
namespace hybrid {


constexpr int default_block_size = 256;
constexpr int warps_in_block = 4;


namespace kernel {


/**
 * The global function for counting the number of nonzeros per row of COO.
 * It is almost like COO spmv routine.
 * It performs is_nonzeros(Coo) times the vector whose values are one
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_line  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param row  the row index array of the matrix
 * @param nnz_per_row  the output nonzeros per row
 */
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void count_coo_row_nnz(const size_type nnz, const size_type num_lines,
                       const ValueType* __restrict__ val,
                       const IndexType* __restrict__ row,
                       IndexType* __restrict__ nnz_per_row,
                       sycl::nd_item<3> item_ct1)
{
    IndexType temp_val = 0;
    const auto start =
        static_cast<size_type>(item_ct1.get_local_range().get(2)) *
            item_ct1.get_group(2) * item_ct1.get_local_range().get(1) *
            num_lines +
        item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2) *
            num_lines;
    size_type num = (nnz > start) * ceildiv(nnz - start, subgroup_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + item_ct1.get_local_id(2);
    const IndexType ind_end = ind_start + (num - 1) * subgroup_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    for (; ind < ind_end; ind += subgroup_size) {
        temp_val += ind < nnz && val[ind] != zero<ValueType>();
        auto next_row = (ind + subgroup_size < nnz) ? row[ind + subgroup_size]
                                                    : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subgroup_size>(tile_block, curr_row, &temp_val);
            if (is_first_in_segment) {
                atomic_add(&(nnz_per_row[curr_row]), temp_val);
            }
            temp_val = 0;
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_end;
        temp_val += ind < nnz && val[ind] != zero<ValueType>();
        // segmented scan

        bool is_first_in_segment =
            segment_scan<subgroup_size>(tile_block, curr_row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(nnz_per_row[curr_row]), temp_val);
        }
    }
}

template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void count_coo_row_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, const size_type nnz,
                       const size_type num_lines, const ValueType* val,
                       const IndexType* row, IndexType* nnz_per_row)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                count_coo_row_nnz<subgroup_size>(nnz, num_lines, val, row,
                                                 nnz_per_row, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_csr(size_type num_rows, size_type max_nnz_per_row,
                 size_type stride, const ValueType* __restrict__ ell_val,
                 const IndexType* __restrict__ ell_col,
                 const ValueType* __restrict__ coo_val,
                 const IndexType* __restrict__ coo_col,
                 const IndexType* __restrict__ coo_offset,
                 IndexType* __restrict__ result_row_ptrs,
                 IndexType* __restrict__ result_col_idxs,
                 ValueType* __restrict__ result_values,
                 sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_rows) {
        auto write_to = result_row_ptrs[tidx];
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            const auto source_idx = tidx + stride * i;
            if (ell_val[source_idx] != zero<ValueType>()) {
                result_values[write_to] = ell_val[source_idx];
                result_col_idxs[write_to] = ell_col[source_idx];
                write_to++;
            }
        }
        for (auto i = coo_offset[tidx]; i < coo_offset[tidx + 1]; i++) {
            if (coo_val[i] != zero<ValueType>()) {
                result_values[write_to] = coo_val[i];
                result_col_idxs[write_to] = coo_col[i];
                write_to++;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(fill_in_csr, fill_in_csr);


template <typename ValueType1, typename ValueType2>
void add(size_type num, ValueType1* __restrict__ val1,
         const ValueType2* __restrict__ val2, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num) {
        val1[tidx] += val2[tidx];
    }
}

GKO_ENABLE_DEFAULT_HOST(add, add);


}  // namespace kernel


template <typename ValueType, typename IndexType>
void split_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& data,
    const int64* row_ptrs, size_type ell_limit, size_type num_rows,
    Array<matrix_data_entry<ValueType, IndexType>>& ell_data,
    Array<matrix_data_entry<ValueType, IndexType>>& coo_data)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_SPLIT_MATRIX_DATA_KERNEL);


}  // namespace hybrid
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
