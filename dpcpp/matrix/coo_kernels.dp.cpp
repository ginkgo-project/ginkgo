// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/coo_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/format_conversion.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The DPCPP namespace.
 *
 * @ingroup dpcpp
 */
namespace dpcpp {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


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
template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
void spmv_kernel(const size_type nnz, const size_type num_lines,
                 const ValueType* __restrict__ val,
                 const IndexType* __restrict__ col,
                 const IndexType* __restrict__ row,
                 const ValueType* __restrict__ b, const size_type b_stride,
                 ValueType* __restrict__ c, const size_type c_stride,
                 Closure scale, sycl::nd_item<3> item_ct1)
{
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
}


template <typename ValueType, typename IndexType>
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


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
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
void spmv2(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
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
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
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
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


}  // namespace coo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
