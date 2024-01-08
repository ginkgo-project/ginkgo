// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cb_gmres_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/cb_gmres_accessor.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The CB_GMRES solver namespace.
 *
 * @ingroup cb_gmres
 */
namespace cb_gmres {


constexpr int default_block_size = 256;
constexpr int default_dot_dim = 16;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


#include "dpcpp/solver/common_gmres_kernels.dp.inc"


template <typename ValueType>
void zero_matrix_kernel(size_type m, size_type n, size_type stride,
                        ValueType* __restrict__ array,
                        sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < n) {
        auto pos = tidx;
        for (size_type k = 0; k < m; ++k) {
            array[pos] = zero<ValueType>();
            pos += stride;
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(zero_matrix_kernel, zero_matrix_kernel);


// Must be called with at least `num_rows * stride_krylov` threads in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
void restart_1_kernel(size_type num_rows, size_type num_rhs,
                      size_type krylov_dim, Accessor3d krylov_bases,
                      ValueType* __restrict__ residual_norm_collection,
                      size_type stride_residual_nc, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    // krylov indices
    const auto krylov_idx = global_id / krylov_stride[0];
    const auto reminder = global_id % krylov_stride[0];
    const auto krylov_row = reminder / krylov_stride[1];
    const auto rhs = reminder % krylov_stride[1];

    // residual_norm indices (separated for better coalesced access)
    const auto residual_row = global_id / stride_residual_nc;
    const auto residual_col = global_id % stride_residual_nc;

    if (krylov_idx < krylov_dim + 1 && krylov_row < num_rows && rhs < num_rhs) {
        krylov_bases(krylov_idx, krylov_row, rhs) = zero<ValueType>();
    }

    if (residual_row < krylov_dim + 1 && residual_col < num_rhs) {
        residual_norm_collection[residual_row * stride_residual_nc +
                                 residual_col] = zero<ValueType>();
    }
}

template <size_type block_size, typename ValueType, typename Accessor3d>
void restart_1_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, size_type num_rows, size_type num_rhs,
                      size_type krylov_dim, Accessor3d krylov_bases,
                      ValueType* residual_norm_collection,
                      size_type stride_residual_nc)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                restart_1_kernel<block_size>(
                    num_rows, num_rhs, krylov_dim, krylov_bases,
                    residual_norm_collection, stride_residual_nc, item_ct1);
            });
    });
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
void restart_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType* __restrict__ residual, size_type stride_residual,
    const remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ residual_norm_collection, Accessor3d krylov_bases,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    size_type* __restrict__ final_iter_nums, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const auto row_idx = global_id / krylov_stride[1];
    const auto col_idx = global_id % krylov_stride[1];

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases(0, row_idx, col_idx) = value;
        next_krylov_basis[row_idx * stride_next_krylov + col_idx] = value;
    }
}

template <size_type block_size, typename ValueType, typename Accessor3d>
void restart_2_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, size_type num_rows, size_type num_rhs,
                      const ValueType* residual, size_type stride_residual,
                      const remove_complex<ValueType>* residual_norm,
                      ValueType* residual_norm_collection,
                      Accessor3d krylov_bases, ValueType* next_krylov_basis,
                      size_type stride_next_krylov, size_type* final_iter_nums)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                restart_2_kernel<block_size>(
                    num_rows, num_rhs, residual, stride_residual, residual_norm,
                    residual_norm_collection, krylov_bases, next_krylov_basis,
                    stride_next_krylov, final_iter_nums, item_ct1);
            });
    });
}


void increase_final_iteration_numbers_kernel(
    size_type* __restrict__ final_iter_nums,
    const stopping_status* __restrict__ stop_status, size_type total_number,
    sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    if (global_id < total_number) {
        final_iter_nums[global_id] += !stop_status[global_id].has_stopped();
    }
}

GKO_ENABLE_DEFAULT_HOST(increase_final_iteration_numbers_kernel,
                        increase_final_iteration_numbers_kernel);


template <typename ValueType>
void multinorm2_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* __restrict__ norms,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1,
    uninitialized_array<remove_complex<ValueType>,
                        default_dot_dim*(default_dot_dim + 1)>*
        reduction_helper_array)
{
    using rc_vtype = remove_complex<ValueType>;
    const auto tidx = item_ct1.get_local_id(2);
    const auto tidy = item_ct1.get_local_id(1);
    const auto col_idx = item_ct1.get_group(2) * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, item_ct1.get_group_range(1));
    const auto start_row = item_ct1.get_group(1) * num;
    const auto end_row = ((item_ct1.get_group(1) + 1) * num > num_rows)
                             ? num_rows
                             : (item_ct1.get_group(1) + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    rc_vtype* __restrict__ reduction_helper = (*reduction_helper_array);
    rc_vtype local_res = zero<rc_vtype>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_res += squared_norm(next_krylov_basis[next_krylov_idx]);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    group::this_thread_block(item_ct1).sync();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block = group::tiled_partition<default_dot_dim>(
        group::this_thread_block(item_ct1));
    const auto sum = ::gko::kernels::dpcpp::reduce(
        tile_block, local_res,
        [](const rc_vtype& a, const rc_vtype& b) { return a + b; });
    const auto new_col_idx = item_ct1.get_group(2) * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_add(norms + norms_idx, sum);
    }
}

template <typename ValueType>
void multinorm2_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, size_type num_rows,
                       size_type num_cols, const ValueType* next_krylov_basis,
                       size_type stride_next_krylov,
                       remove_complex<ValueType>* norms,
                       const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<
            uninitialized_array<remove_complex<ValueType>,
                                default_dot_dim*(default_dot_dim + 1)>,
            0, sycl::access_mode::read_write, sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(default_dot_dim)]] {
                    multinorm2_kernel(
                        num_rows, num_cols, next_krylov_basis,
                        stride_next_krylov, norms, stop_status, item_ct1,
                        reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


template <typename ValueType>
void multinorminf_without_stop_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* __restrict__ norms,
    size_type stride_norms, sycl::nd_item<3> item_ct1,
    uninitialized_array<remove_complex<ValueType>,
                        default_dot_dim*(default_dot_dim + 1)>*
        reduction_helper_array)
{
    using rc_vtype = remove_complex<ValueType>;
    const auto tidx = item_ct1.get_local_id(2);
    const auto tidy = item_ct1.get_local_id(1);
    const auto col_idx = item_ct1.get_group(2) * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, item_ct1.get_group_range(1));
    const auto start_row = item_ct1.get_group(1) * num;
    const auto end_row = ((item_ct1.get_group(1) + 1) * num > num_rows)
                             ? num_rows
                             : (item_ct1.get_group(1) + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    rc_vtype* __restrict__ reduction_helper = (*reduction_helper_array);
    rc_vtype local_max = zero<rc_vtype>();
    if (col_idx < num_cols) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            local_max =
                (local_max >= std::abs(next_krylov_basis[next_krylov_idx]))
                    ? local_max
                    : std::abs(next_krylov_basis[next_krylov_idx]);
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_max;
    group::this_thread_block(item_ct1).sync();
    local_max = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block = group::tiled_partition<default_dot_dim>(
        group::this_thread_block(item_ct1));
    const auto value = ::gko::kernels::dpcpp::reduce(
        tile_block, local_max, [](const rc_vtype& a, const rc_vtype& b) {
            return ((a >= b) ? a : b);
        });
    const auto new_col_idx = item_ct1.get_group(2) * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols) {
        const auto norms_idx = new_col_idx;
        atomic_max(norms + norms_idx, value);
    }
}

template <typename ValueType>
void multinorminf_without_stop_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_rows, size_type num_cols, const ValueType* next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* norms,
    size_type stride_norms)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<
            uninitialized_array<remove_complex<ValueType>,
                                default_dot_dim*(default_dot_dim + 1)>,
            0, sycl::access_mode::read_write, sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(default_dot_dim)]] {
                    multinorminf_without_stop_kernel(
                        num_rows, num_cols, next_krylov_basis,
                        stride_next_krylov, norms, stride_norms, item_ct1,
                        reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


// ONLY computes the inf-norm (into norms2) when compute_inf is true
template <bool compute_inf, typename ValueType>
void multinorm2_inf_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov,
    remove_complex<ValueType>* __restrict__ norms1,
    remove_complex<ValueType>* __restrict__ norms2,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1,
    uninitialized_array<remove_complex<ValueType>,
                        (1 + compute_inf) *
                            default_dot_dim*(default_dot_dim + 1)>*
        reduction_helper_array)
{
    using rc_vtype = remove_complex<ValueType>;
    const auto tidx = item_ct1.get_local_id(2);
    const auto tidy = item_ct1.get_local_id(1);
    const auto col_idx = item_ct1.get_group(2) * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, item_ct1.get_group_range(1));
    const auto start_row = item_ct1.get_group(1) * num;
    const auto end_row = ((item_ct1.get_group(1) + 1) * num > num_rows)
                             ? num_rows
                             : (item_ct1.get_group(1) + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    rc_vtype* __restrict__ reduction_helper_add = (*reduction_helper_array);
    rc_vtype* __restrict__ reduction_helper_max =
        static_cast<rc_vtype*>((*reduction_helper_array)) +
        default_dot_dim * (default_dot_dim + 1);
    rc_vtype local_res = zero<rc_vtype>();
    rc_vtype local_max = zero<rc_vtype>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto num = next_krylov_basis[next_krylov_idx];
            local_res += squared_norm(num);
            if (compute_inf) {
                local_max =
                    ((local_max >= std::abs(num)) ? local_max : std::abs(num));
            }
        }
    }
    // Add reduction
    reduction_helper_add[tidx * (default_dot_dim + 1) + tidy] = local_res;
    if (compute_inf) {
        reduction_helper_max[tidx * (default_dot_dim + 1) + tidy] = local_max;
    }
    group::this_thread_block(item_ct1).sync();
    local_res = reduction_helper_add[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block = group::tiled_partition<default_dot_dim>(
        group::this_thread_block(item_ct1));
    const auto sum = ::gko::kernels::dpcpp::reduce(
        tile_block, local_res,
        [](const rc_vtype& a, const rc_vtype& b) { return a + b; });
    rc_vtype reduced_max{};
    if (compute_inf) {
        local_max = reduction_helper_max[tidy * (default_dot_dim + 1) + tidx];
        reduced_max = ::gko::kernels::dpcpp::reduce(
            tile_block, local_max, [](const rc_vtype& a, const rc_vtype& b) {
                return ((a >= b) ? a : b);
            });
    }
    const auto new_col_idx = item_ct1.get_group(2) * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto norms_idx = new_col_idx;
        atomic_add(norms1 + norms_idx, sum);
        if (compute_inf) {
            atomic_max(norms2 + norms_idx, reduced_max);
        }
    }
}

template <bool compute_inf, typename ValueType>
void multinorm2_inf_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_rows, size_type num_cols, const ValueType* next_krylov_basis,
    size_type stride_next_krylov, remove_complex<ValueType>* norms1,
    remove_complex<ValueType>* norms2, const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<
            uninitialized_array<remove_complex<ValueType>,
                                (1 + compute_inf) *
                                    default_dot_dim*(default_dot_dim + 1)>,
            0, sycl::access_mode::read_write, sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(default_dot_dim)]] {
                    multinorm2_inf_kernel<compute_inf>(
                        num_rows, num_cols, next_krylov_basis,
                        stride_next_krylov, norms1, norms2, stop_status,
                        item_ct1, reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


template <int dot_dim, typename ValueType, typename Accessor3d>
void multidot_kernel(
    size_type num_rows, size_type num_cols,
    const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, const Accessor3d krylov_bases,
    ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, dot_dim * dot_dim>& reduction_helper_array)
{
    /*
     * In general in this kernel:
     * grid_dim
     *   x: for col_idx (^= which right hand side)
     *   y: for row_idx
     *   z: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for col_idx (must be < dot_dim)
     *   y: for row_idx (must be < dot_dim)
     *   (z not used, must be set to 1 in dim)
     */
    const size_type tidx = item_ct1.get_local_id(2);
    const size_type tidy = item_ct1.get_local_id(1);
    const size_type col_idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    const size_type num_rows_per_thread =
        ceildiv(num_rows, item_ct1.get_group_range(1));
    const size_type start_row =
        item_ct1.get_group(1) * num_rows_per_thread + item_ct1.get_local_id(1);
    const auto end_row =
        min((item_ct1.get_group(1) + 1) * num_rows_per_thread, num_rows);
    const size_type k = item_ct1.get_group(0);
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row;
             i += item_ct1.get_local_range().get(1)) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            ValueType other_basis = krylov_bases(k, i, col_idx);
            local_res += next_krylov_basis[next_krylov_idx] * conj(other_basis);
        }
    }
    // Transpose local_res, so each warp contains a local_res from the same
    // right hand side
    reduction_helper[tidx * dot_dim + tidy] = local_res;
    auto thread_block = group::this_thread_block(item_ct1);
    thread_block.sync();
    local_res = reduction_helper[tidy * dot_dim + tidx];
    const auto new_col_idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + tidy;
    const auto tile_block = group::tiled_partition<dot_dim>(thread_block);
    const auto sum = ::gko::kernels::dpcpp::reduce(
        tile_block, local_res,
        [](const ValueType& a, const ValueType& b) { return a + b; });
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}

template <int dot_dim, typename ValueType, typename Accessor3d>
void multidot_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, size_type num_rows, size_type num_cols,
                     const ValueType* next_krylov_basis,
                     size_type stride_next_krylov,
                     const Accessor3d krylov_bases, ValueType* hessenberg_iter,
                     size_type stride_hessenberg,
                     const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, dot_dim * dot_dim>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(dot_dim)]] {
                    multidot_kernel<dot_dim>(
                        num_rows, num_cols, next_krylov_basis,
                        stride_next_krylov, krylov_bases, hessenberg_iter,
                        stride_hessenberg, stop_status, item_ct1,
                        *reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


template <int block_size, typename ValueType, typename Accessor3d>
void singledot_kernel(
    size_type num_rows, const ValueType* __restrict__ next_krylov_basis,
    size_type stride_next_krylov, const Accessor3d krylov_bases,
    ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, block_size>& reduction_helper_array)
{
    /*
     * In general in this kernel:
     * grid_dim
     *   x: for row_idx
     *   y: for num_iters (number of krylov vectors)
     * block_dim
     *   x: for row_idx (must be block_size)
     *   (y and z not used, must be set to 1 in dim)
     */
    const size_type tidx = item_ct1.get_local_id(2);
    constexpr size_type col_idx{0};
    const size_type k = item_ct1.get_group(1);
    const size_type num_rows_per_thread =
        ceildiv(num_rows, item_ct1.get_group_range(2));
    const size_type start_row =
        item_ct1.get_group(2) * num_rows_per_thread + item_ct1.get_local_id(2);
    const auto end_row =
        min((item_ct1.get_group(2) + 1) * num_rows_per_thread, num_rows);
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
    if (!stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row; i < end_row; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            ValueType other_basis = krylov_bases(k, i, col_idx);
            local_res += next_krylov_basis[next_krylov_idx] * conj(other_basis);
        }
    }
    // Transpose local_res, so each warp contains a local_res from the same
    // right hand side
    reduction_helper[tidx] = local_res;
    auto thread_block = group::this_thread_block(item_ct1);
    thread_block.sync();
    ::gko::kernels::dpcpp::reduce(
        thread_block, reduction_helper,
        [](const ValueType& a, const ValueType& b) { return a + b; });
    if (tidx == 0 && !stop_status[col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, reduction_helper[0]);
    }
}

template <int block_size, typename ValueType, typename Accessor3d>
void singledot_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue* queue, size_type num_rows,
                      const ValueType* next_krylov_basis,
                      size_type stride_next_krylov,
                      const Accessor3d krylov_bases, ValueType* hessenberg_iter,
                      size_type stride_hessenberg,
                      const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    singledot_kernel<block_size>(
                        num_rows, next_krylov_basis, stride_next_krylov,
                        krylov_bases, hessenberg_iter, stride_hessenberg,
                        stop_status, item_ct1,
                        *reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
void update_next_krylov_kernel(
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases,
    const ValueType* __restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        auto local_res = next_krylov_basis[next_krylov_idx];
        for (size_type k = 0; k < num_iters; ++k) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;

            local_res -= hessenberg_iter[hessenberg_idx] *
                         krylov_bases(k, row_idx, col_idx);
        }
        next_krylov_basis[next_krylov_idx] = local_res;
    }
}

template <int block_size, typename ValueType, typename Accessor3d>
void update_next_krylov_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases, const ValueType* hessenberg_iter,
    size_type stride_hessenberg, const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                update_next_krylov_kernel<block_size>(
                    num_iters, num_rows, num_cols, next_krylov_basis,
                    stride_next_krylov, krylov_bases, hessenberg_iter,
                    stride_hessenberg, stop_status, item_ct1);
            });
    });
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
void update_next_krylov_and_add_kernel(
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases, ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg, const ValueType* __restrict__ buffer_iter,
    size_type stride_buffer, const stopping_status* __restrict__ stop_status,
    const stopping_status* __restrict__ reorth_status,
    sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped() &&
        !reorth_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;
        auto local_res = next_krylov_basis[next_krylov_idx];
        for (size_type k = 0; k < num_iters; ++k) {
            const auto hessenberg_idx = k * stride_hessenberg + col_idx;
            const auto buffer_idx = k * stride_buffer + col_idx;
            local_res -=
                buffer_iter[buffer_idx] * krylov_bases(k, row_idx, col_idx);
            if ((row_idx == 0) && !reorth_status[col_idx].has_stopped()) {
                hessenberg_iter[hessenberg_idx] += buffer_iter[buffer_idx];
            }
        }
        next_krylov_basis[next_krylov_idx] = local_res;
    }
}

template <int block_size, typename ValueType, typename Accessor3d>
void update_next_krylov_and_add_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_iters, size_type num_rows, size_type num_cols,
    ValueType* next_krylov_basis, size_type stride_next_krylov,
    const Accessor3d krylov_bases, ValueType* hessenberg_iter,
    size_type stride_hessenberg, const ValueType* buffer_iter,
    size_type stride_buffer, const stopping_status* stop_status,
    const stopping_status* reorth_status)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                update_next_krylov_and_add_kernel<block_size>(
                    num_iters, num_rows, num_cols, next_krylov_basis,
                    stride_next_krylov, krylov_bases, hessenberg_iter,
                    stride_hessenberg, buffer_iter, stride_buffer, stop_status,
                    reorth_status, item_ct1);
            });
    });
}


// Must be called with at least `num_rhs` threads
template <int block_size, typename ValueType, typename Accessor3d>
void check_arnoldi_norms(
    size_type num_rhs, remove_complex<ValueType>* __restrict__ arnoldi_norm,
    size_type stride_norm, ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg, size_type iter, Accessor3d krylov_bases,
    const stopping_status* __restrict__ stop_status,
    stopping_status* __restrict__ reorth_status,
    size_type* __restrict__ num_reorth, sycl::nd_item<3> item_ct1)
{
    const remove_complex<ValueType> eta_squared = 1.0 / 2.0;
    const auto col_idx = thread::get_thread_id_flat(item_ct1);
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;

    if (col_idx < num_rhs && !stop_status[col_idx].has_stopped()) {
        const auto num0 = (std::sqrt(eta_squared * arnoldi_norm[col_idx]));
        const auto num11 = std::sqrt(arnoldi_norm[col_idx + stride_norm]);
        const auto num2 = has_scalar ? (arnoldi_norm[col_idx + 2 * stride_norm])
                                     : remove_complex<ValueType>{};
        if (num11 < num0) {
            reorth_status[col_idx].reset();
            atomic_add(num_reorth, one<size_type>());
        } else {
            reorth_status[col_idx].stop(1);
        }
        arnoldi_norm[col_idx] = num0;
        arnoldi_norm[col_idx + stride_norm] = num11;
        hessenberg_iter[iter * stride_hessenberg + col_idx] = num11;
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, iter, col_idx, num2 / num11);
    }
}

template <int block_size, typename ValueType, typename Accessor3d>
void check_arnoldi_norms(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                         sycl::queue* queue, size_type num_rhs,
                         remove_complex<ValueType>* arnoldi_norm,
                         size_type stride_norm, ValueType* hessenberg_iter,
                         size_type stride_hessenberg, size_type iter,
                         Accessor3d krylov_bases,
                         const stopping_status* stop_status,
                         stopping_status* reorth_status, size_type* num_reorth)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                check_arnoldi_norms<block_size>(
                    num_rhs, arnoldi_norm, stride_norm, hessenberg_iter,
                    stride_hessenberg, iter, krylov_bases, stop_status,
                    reorth_status, num_reorth, item_ct1);
            });
    });
}


template <int block_size, typename RealValueType, typename Accessor3d>
void set_scalar_kernel(size_type num_rhs, size_type num_blocks,
                       const RealValueType* __restrict__ residual_norm,
                       size_type stride_residual,
                       const RealValueType* __restrict__ arnoldi_inf,
                       size_type stride_inf, Accessor3d krylov_bases,
                       sycl::nd_item<3> item_ct1)
{
    static_assert(!is_complex_s<RealValueType>::value,
                  "ValueType must not be complex!");
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const auto blk_idx = global_id / krylov_stride[1];
    const auto col_idx = global_id % krylov_stride[1];

    if (blk_idx < num_blocks && col_idx < num_rhs) {
        if (blk_idx == 0) {
            const auto num1 = residual_norm[col_idx];
            const auto num2 = arnoldi_inf[col_idx];
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, {0}, col_idx, num2 / num1);
        } else {
            const auto num = one<RealValueType>();
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, blk_idx, col_idx, num);
        }
    }
}

template <int block_size, typename RealValueType, typename Accessor3d>
void set_scalar_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, size_type num_rhs,
                       size_type num_blocks, const RealValueType* residual_norm,
                       size_type stride_residual,
                       const RealValueType* arnoldi_inf, size_type stride_inf,
                       Accessor3d krylov_bases)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                set_scalar_kernel<block_size>(
                    num_rhs, num_blocks, residual_norm, stride_residual,
                    arnoldi_inf, stride_inf, krylov_bases, item_ct1);
            });
    });
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
void update_krylov_next_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType* __restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3d krylov_bases, const ValueType* __restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;

        const auto next_krylov_value =
            next_krylov_basis[next_krylov_idx] / hessenberg;

        next_krylov_basis[next_krylov_idx] = next_krylov_value;
        krylov_bases(iter + 1, row_idx, col_idx) = next_krylov_value;
    }
}

template <int block_size, typename ValueType, typename Accessor3d>
void update_krylov_next_krylov_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType* next_krylov_basis, size_type stride_next_krylov,
    Accessor3d krylov_bases, const ValueType* hessenberg_iter,
    size_type stride_hessenberg, const stopping_status* stop_status)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                update_krylov_next_krylov_kernel<block_size>(
                    iter, num_rows, num_cols, next_krylov_basis,
                    stride_next_krylov, krylov_bases, hessenberg_iter,
                    stride_hessenberg, stop_status, item_ct1);
            });
    });
}


// Must be called with at least `stride_preconditioner * num_rows` threads
// in total.
template <size_type block_size, typename ValueType, typename Accessor3d>
void calculate_Qy_kernel(size_type num_rows, size_type num_cols,
                         const Accessor3d krylov_bases,
                         const ValueType* __restrict__ y, size_type stride_y,
                         ValueType* __restrict__ before_preconditioner,
                         size_type stride_preconditioner,
                         const size_type* __restrict__ final_iter_nums,
                         sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        ValueType temp = zero<ValueType>();
        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            temp += krylov_bases(j, row_id, col_id) * y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}

template <size_type block_size, typename ValueType, typename Accessor3d>
void calculate_Qy_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                         sycl::queue* queue, size_type num_rows,
                         size_type num_cols, const Accessor3d krylov_bases,
                         const ValueType* y, size_type stride_y,
                         ValueType* before_preconditioner,
                         size_type stride_preconditioner,
                         const size_type* final_iter_nums)
{
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             calculate_Qy_kernel<block_size>(
                                 num_rows, num_cols, krylov_bases, y, stride_y,
                                 before_preconditioner, stride_preconditioner,
                                 final_iter_nums, item_ct1);
                         });
    });
}


template <typename ValueType>
void zero_matrix(std::shared_ptr<const DpcppExecutor> exec, size_type m,
                 size_type n, size_type stride, ValueType* array)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    zero_matrix_kernel(grid_size, block_size, 0, exec->get_queue(), m, n,
                       stride, array);
}


template <typename ValueType>
void initialize(std::shared_ptr<const DpcppExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                array<stopping_status>* stop_status, size_type krylov_dim)
{
    const auto num_threads = std::max(b->get_size()[0] * b->get_stride(),
                                      krylov_dim * b->get_size()[1]);
    const dim3 grid_dim(ceildiv(num_threads, default_block_size), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), b->get_size()[0],
        b->get_size()[1], krylov_dim, b->get_const_values(), b->get_stride(),
        residual->get_values(), residual->get_stride(),
        givens_sin->get_values(), givens_sin->get_stride(),
        givens_cos->get_values(), givens_cos->get_stride(),
        stop_status->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL);


template <typename ValueType, typename Accessor3d>
void restart(std::shared_ptr<const DpcppExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             Accessor3d krylov_bases,
             matrix::Dense<ValueType>* next_krylov_basis,
             array<size_type>* final_iter_nums, array<char>& reduction_tmp,
             size_type krylov_dim)
{
    constexpr bool use_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const dim3 grid_dim_1(
        ceildiv((krylov_dim + 1) * krylov_stride[0], default_block_size), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;
    const auto stride_arnoldi = arnoldi_norm->get_stride();

    restart_1_kernel<block_size>(
        grid_dim_1, block_dim, 0, exec->get_queue(), residual->get_size()[0],
        residual->get_size()[1], krylov_dim, krylov_bases,
        residual_norm_collection->get_values(),
        residual_norm_collection->get_stride());
    kernels::dpcpp::dense::compute_norm2_dispatch(exec, residual, residual_norm,
                                                  reduction_tmp);

    if (use_scalar) {
        components::fill_array(exec,
                               arnoldi_norm->get_values() + 2 * stride_arnoldi,
                               num_rhs, zero<remove_complex<ValueType>>());
        const dim3 grid_size_nrm(ceildiv(num_rhs, default_dot_dim),
                                 exec->get_num_computing_units() * 2);
        const dim3 block_size_nrm(default_dot_dim, default_dot_dim);
        multinorminf_without_stop_kernel(
            grid_size_nrm, block_size_nrm, 0, exec->get_queue(), num_rows,
            num_rhs, residual->get_const_values(), residual->get_stride(),
            arnoldi_norm->get_values() + 2 * stride_arnoldi, 0);
    }

    if (gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value) {
        set_scalar_kernel<default_block_size>(
            ceildiv(num_rhs * (krylov_dim + 1), default_block_size),
            default_block_size, 0, exec->get_queue(), num_rhs, krylov_dim + 1,
            residual_norm->get_const_values(), residual_norm->get_stride(),
            arnoldi_norm->get_const_values() + 2 * stride_arnoldi,
            stride_arnoldi, krylov_bases);
    }

    const dim3 grid_dim_2(
        ceildiv(std::max<size_type>(num_rows, 1) * krylov_stride[1],
                default_block_size),
        1, 1);
    restart_2_kernel<block_size>(
        grid_dim_2, block_dim, 0, exec->get_queue(), residual->get_size()[0],
        residual->get_size()[1], residual->get_const_values(),
        residual->get_stride(), residual_norm->get_const_values(),
        residual_norm_collection->get_values(), krylov_bases,
        next_krylov_basis->get_values(), next_krylov_basis->get_stride(),
        final_iter_nums->get_data());
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_RESTART_KERNEL);


template <typename ValueType, typename Accessor3dim>
void finish_arnoldi_CGS(std::shared_ptr<const DpcppExecutor> exec,
                        matrix::Dense<ValueType>* next_krylov_basis,
                        Accessor3dim krylov_bases,
                        matrix::Dense<ValueType>* hessenberg_iter,
                        matrix::Dense<ValueType>* buffer_iter,
                        matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
                        size_type iter, const stopping_status* stop_status,
                        stopping_status* reorth_status,
                        array<size_type>* num_reorth)
{
    const auto dim_size = next_krylov_basis->get_size();
    if (dim_size[1] == 0) {
        return;
    }
    using non_complex = remove_complex<ValueType>;
    // optimization parameter
    constexpr int singledot_block_size = default_dot_dim;
    constexpr bool use_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3dim>::value;
    const auto stride_next_krylov = next_krylov_basis->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    const auto stride_buffer = buffer_iter->get_stride();
    const auto stride_arnoldi = arnoldi_norm->get_stride();
    const dim3 grid_size(ceildiv(dim_size[1], default_dot_dim),
                         exec->get_num_computing_units() * 2);
    const dim3 grid_size_num_iters(ceildiv(dim_size[1], default_dot_dim),
                                   exec->get_num_computing_units() * 2,
                                   iter + 1);
    const dim3 block_size(default_dot_dim, default_dot_dim);
    // Note: having iter first (instead of row_idx information) is likely
    //       beneficial for avoiding atomic_add conflicts, but that needs
    //       further investigation.
    const dim3 grid_size_iters_single(exec->get_num_computing_units() * 2,
                                      iter + 1);
    const dim3 block_size_iters_single(singledot_block_size);
    size_type num_reorth_host;

    components::fill_array(exec, arnoldi_norm->get_values(), dim_size[1],
                           zero<non_complex>());
    multinorm2_kernel(grid_size, block_size, 0, exec->get_queue(), dim_size[0],
                      dim_size[1], next_krylov_basis->get_const_values(),
                      stride_next_krylov, arnoldi_norm->get_values(),
                      stop_status);
    zero_matrix(exec, iter + 1, dim_size[1], stride_hessenberg,
                hessenberg_iter->get_values());
    if (dim_size[1] > 1) {
        multidot_kernel<default_dot_dim>(
            grid_size_num_iters, block_size, 0, exec->get_queue(), dim_size[0],
            dim_size[1], next_krylov_basis->get_const_values(),
            stride_next_krylov, krylov_bases, hessenberg_iter->get_values(),
            stride_hessenberg, stop_status);
    } else {
        singledot_kernel<singledot_block_size>(
            grid_size_iters_single, block_size_iters_single, 0,
            exec->get_queue(), dim_size[0],
            next_krylov_basis->get_const_values(), stride_next_krylov,
            krylov_bases, hessenberg_iter->get_values(), stride_hessenberg,
            stop_status);
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    // end
    update_next_krylov_kernel<default_block_size>(
        ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
        default_block_size, 0, exec->get_queue(), iter + 1, dim_size[0],
        dim_size[1], next_krylov_basis->get_values(), stride_next_krylov,
        krylov_bases, hessenberg_iter->get_const_values(), stride_hessenberg,
        stop_status);

    // for i in 1:iter
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end
    components::fill_array(exec, arnoldi_norm->get_values() + stride_arnoldi,
                           dim_size[1], zero<non_complex>());
    if (use_scalar) {
        components::fill_array(exec,
                               arnoldi_norm->get_values() + 2 * stride_arnoldi,
                               dim_size[1], zero<non_complex>());
    }
    multinorm2_inf_kernel<use_scalar>(
        grid_size, block_size, 0, exec->get_queue(), dim_size[0], dim_size[1],
        next_krylov_basis->get_const_values(), stride_next_krylov,
        arnoldi_norm->get_values() + stride_arnoldi,
        arnoldi_norm->get_values() + 2 * stride_arnoldi, stop_status);
    // nrmN = norm(next_krylov_basis)
    components::fill_array(exec, num_reorth->get_data(), 1, zero<size_type>());
    check_arnoldi_norms<default_block_size>(
        ceildiv(dim_size[1], default_block_size), default_block_size, 0,
        exec->get_queue(), dim_size[1], arnoldi_norm->get_values(),
        stride_arnoldi, hessenberg_iter->get_values(), stride_hessenberg,
        iter + 1, krylov_bases, stop_status, reorth_status,
        num_reorth->get_data());
    num_reorth_host = get_element(*num_reorth, 0);
    // num_reorth_host := number of next_krylov vector to be reorthogonalization
    for (size_type l = 1; (num_reorth_host > 0) && (l < 3); l++) {
        zero_matrix(exec, iter + 1, dim_size[1], stride_buffer,
                    buffer_iter->get_values());
        if (dim_size[1] > 1) {
            multidot_kernel<default_dot_dim>(
                grid_size_num_iters, block_size, 0, exec->get_queue(),
                dim_size[0], dim_size[1], next_krylov_basis->get_const_values(),
                stride_next_krylov, krylov_bases, buffer_iter->get_values(),
                stride_buffer, stop_status);
        } else {
            singledot_kernel<singledot_block_size>(
                grid_size_iters_single, block_size_iters_single, 0,
                exec->get_queue(), dim_size[0],
                next_krylov_basis->get_const_values(), stride_next_krylov,
                krylov_bases, buffer_iter->get_values(), stride_buffer,
                stop_status);
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        update_next_krylov_and_add_kernel<default_block_size>(
            ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
            default_block_size, 0, exec->get_queue(), iter + 1, dim_size[0],
            dim_size[1], next_krylov_basis->get_values(), stride_next_krylov,
            krylov_bases, hessenberg_iter->get_values(), stride_hessenberg,
            buffer_iter->get_const_values(), stride_buffer, stop_status,
            reorth_status);
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        components::fill_array(exec,
                               arnoldi_norm->get_values() + stride_arnoldi,
                               dim_size[1], zero<non_complex>());
        if (use_scalar) {
            components::fill_array(
                exec, arnoldi_norm->get_values() + 2 * stride_arnoldi,
                dim_size[1], zero<non_complex>());
        }
        multinorm2_inf_kernel<use_scalar>(
            grid_size, block_size, 0, exec->get_queue(), dim_size[0],
            dim_size[1], next_krylov_basis->get_const_values(),
            stride_next_krylov, arnoldi_norm->get_values() + stride_arnoldi,
            arnoldi_norm->get_values() + 2 * stride_arnoldi, stop_status);
        // nrmN = norm(next_krylov_basis)
        components::fill_array(exec, num_reorth->get_data(), 1,
                               zero<size_type>());
        check_arnoldi_norms<default_block_size>(
            ceildiv(dim_size[1], default_block_size), default_block_size, 0,
            exec->get_queue(), dim_size[1], arnoldi_norm->get_values(),
            stride_arnoldi, hessenberg_iter->get_values(), stride_hessenberg,
            iter + 1, krylov_bases, stop_status, reorth_status,
            num_reorth->get_data());
        num_reorth_host = get_element(*num_reorth, 0);
    }

    update_krylov_next_krylov_kernel<default_block_size>(
        ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
        default_block_size, 0, exec->get_queue(), iter, dim_size[0],
        dim_size[1], next_krylov_basis->get_values(), stride_next_krylov,
        krylov_bases, hessenberg_iter->get_const_values(), stride_hessenberg,
        stop_status);
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // krylov_bases(:, iter + 1) = next_krylov_basis
    // End of arnoldi
}

template <typename ValueType>
void givens_rotation(std::shared_ptr<const DpcppExecutor> exec,
                     matrix::Dense<ValueType>* givens_sin,
                     matrix::Dense<ValueType>* givens_cos,
                     matrix::Dense<ValueType>* hessenberg_iter,
                     matrix::Dense<remove_complex<ValueType>>* residual_norm,
                     matrix::Dense<ValueType>* residual_norm_collection,
                     size_type iter, const array<stopping_status>* stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<unsigned int>(ceildiv(num_cols, block_size)), 1, 1};

    givens_rotation_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(),
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1], iter,
        hessenberg_iter->get_values(), hessenberg_iter->get_stride(),
        givens_sin->get_values(), givens_sin->get_stride(),
        givens_cos->get_values(), givens_cos->get_stride(),
        residual_norm->get_values(), residual_norm_collection->get_values(),
        residual_norm_collection->get_stride(), stop_status->get_const_data());
}


template <typename ValueType, typename Accessor3d>
void arnoldi(std::shared_ptr<const DpcppExecutor> exec,
             matrix::Dense<ValueType>* next_krylov_basis,
             matrix::Dense<ValueType>* givens_sin,
             matrix::Dense<ValueType>* givens_cos,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             Accessor3d krylov_bases, matrix::Dense<ValueType>* hessenberg_iter,
             matrix::Dense<ValueType>* buffer_iter,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             size_type iter, array<size_type>* final_iter_nums,
             const array<stopping_status>* stop_status,
             array<stopping_status>* reorth_status,
             array<size_type>* num_reorth)
{
    increase_final_iteration_numbers_kernel(
        static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_size(), default_block_size)),
        default_block_size, 0, exec->get_queue(), final_iter_nums->get_data(),
        stop_status->get_const_data(), final_iter_nums->get_size());
    finish_arnoldi_CGS(exec, next_krylov_basis, krylov_bases, hessenberg_iter,
                       buffer_iter, arnoldi_norm, iter,
                       stop_status->get_const_data(), reorth_status->get_data(),
                       num_reorth);
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL);


template <typename ValueType>
void solve_upper_triangular(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Dense<ValueType>* residual_norm_collection,
    const matrix::Dense<ValueType>* hessenberg, matrix::Dense<ValueType>* y,
    const array<size_type>* final_iter_nums)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{static_cast<unsigned int>(ceildiv(num_rhs, block_size)),
                        1, 1};

    solve_upper_triangular_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), hessenberg->get_size()[1],
        num_rhs, residual_norm_collection->get_const_values(),
        residual_norm_collection->get_stride(), hessenberg->get_const_values(),
        hessenberg->get_stride(), y->get_values(), y->get_stride(),
        final_iter_nums->get_const_data());
}


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(std::shared_ptr<const DpcppExecutor> exec,
                  ConstAccessor3d krylov_bases, size_type num_krylov_bases,
                  const matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    const auto num_rows = before_preconditioner->get_size()[0];
    const auto num_cols = before_preconditioner->get_size()[1];
    const auto stride_before_preconditioner =
        before_preconditioner->get_stride();

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim{
        static_cast<unsigned int>(
            ceildiv(num_rows * stride_before_preconditioner, block_size)),
        1, 1};
    const dim3 block_dim{block_size, 1, 1};


    calculate_Qy_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), num_rows, num_cols,
        krylov_bases, y->get_const_values(), y->get_stride(),
        before_preconditioner->get_values(), stride_before_preconditioner,
        final_iter_nums->get_const_data());
    // Calculate qy
    // before_preconditioner = krylov_bases * y
}


template <typename ValueType, typename ConstAccessor3d>
void solve_krylov(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  ConstAccessor3d krylov_bases,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    if (before_preconditioner->get_size()[1] == 0) {
        return;
    }
    // since hessenberg has dims:  iters x iters * num_rhs
    // krylov_bases has dims:  (iters + 1) x sysmtx[0] x num_rhs
    const auto iters =
        hessenberg->get_size()[1] / before_preconditioner->get_size()[1];
    const auto num_krylov_bases = iters + 1;
    solve_upper_triangular(exec, residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    calculate_qy(exec, krylov_bases, num_krylov_bases, y, before_preconditioner,
                 final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace cb_gmres
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
