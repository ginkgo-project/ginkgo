// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_DP_HPP_
#define GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_DP_HPP_


#include <memory>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "core/base/array_access.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/memory.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {


constexpr int fallback_block_size = 32;


/** Returns an unsigned type matching the size of the given float type. */
template <typename T>
struct float_to_unsigned_impl {};

template <>
struct float_to_unsigned_impl<double> {
    using type = uint64;
};

template <>
struct float_to_unsigned_impl<float> {
    using type = uint32;
};

template <>
struct float_to_unsigned_impl<__half> {
    using type = uint16;
};


/**
 * Checks if a floating point number representation matches the representation
 * of the quiet NaN with value gko::nan() exactly.
 */
template <typename T>
__dpct_inline__ std::enable_if_t<!is_complex_s<T>::value, bool> is_nan_exact(
    const T& value)
{
    using type = typename float_to_unsigned_impl<T>::type;
    type value_bytes{};
    type nan_bytes{};
    auto nan_value = nan<T>();
    using std::memcpy;
    memcpy(&value_bytes, &value, sizeof(value));
    memcpy(&nan_bytes, &nan_value, sizeof(value));
    return value_bytes == nan_bytes;
}


/**
 * Checks if any component of the complex value matches the quiet NaN with
 * value gko::nan() exactly.
 */
template <typename T>
__dpct_inline__ std::enable_if_t<is_complex_s<T>::value, bool> is_nan_exact(
    const T& value)
{
    return is_nan_exact(value.real()) || is_nan_exact(value.imag());
}


template <bool is_upper, typename ValueType, typename IndexType>
void sptrsv_naive_legacy_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter, sycl::nd_item<3> item_ct1,
    IndexType& shared_block_base_idx)
{
    if (item_ct1.get_local_id(2) == 0) {
        shared_block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * fallback_block_size;
    }
    item_ct1.barrier();

    const auto full_gid = static_cast<IndexType>(item_ct1.get_local_id(2)) +
                          shared_block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward
    // upper tri matrix: start at last entry (row_end - 1), run backward
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    ValueType sum = zero<ValueType>();
    auto j = row_begin;
    auto col = colidxs[j];
    while (j != row_end) {
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        auto x_val = load_relaxed(x + col * x_stride + rhs);
        while (!is_nan_exact(x_val)) {
            sum += vals[j] * x_val;
            j += row_step;
            col = colidxs[j];
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            x_val = load_relaxed(x + col * x_stride + rhs);
        }
        // to avoid the kernel hanging on matrices without diagonal,
        // we bail out if we are past the triangle, even if it's not
        // the diagonal entry. This may lead to incorrect results,
        // but prevents an infinite loop.
        if (is_upper ? row >= col : row <= col) {
            // assert(row == col);
            auto diag = unit_diag ? one<ValueType>() : vals[j];
            const auto r = (b[row * b_stride + rhs] - sum) / diag;
            store_relaxed(x + row * x_stride + rhs, r);
            // after we encountered the diagonal, we are done
            // this also skips entries outside the triangle
            j = row_end;
            if (is_nan_exact(r)) {
                store_relaxed(x + row * x_stride + rhs, zero<ValueType>());
                *nan_produced = true;
            }
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
        }
    }
}

template <bool is_upper, typename ValueType, typename IndexType>
void sptrsv_naive_legacy_kernel(
    dim3 grid, dim3 block, gko::size_type, sycl::queue* queue,
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<IndexType, 0> shared_block_base_idx_acc_ct1(cgh);
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    sptrsv_naive_legacy_kernel<is_upper>(
                        rowptrs, colidxs, vals, b, b_stride, x, x_stride, n,
                        nrhs, unit_diag, nan_produced, atomic_counter, item_ct1,
                        *shared_block_base_idx_acc_ct1.get_pointer());
                });
    });
}


template <bool is_upper, typename ValueType, typename IndexType>
void sptrsv_naive_caching(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          bool unit_diag, const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x)
{
    const auto n = matrix->get_size()[0];
    const auto nrhs = b->get_size()[1];

    // Initialize x to all NaNs.
    dense::fill(exec, x, nan<ValueType>());

    array<bool> nan_produced(exec, 1);
    array<IndexType> atomic_counter(exec, 1);
    // TODO: one submit
    nan_produced.fill(false);
    atomic_counter.fill(0);

    const dim3 block_size(fallback_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);
    exec->synchronize();
    std::cout << "start trs " << block_size.x << std::endl;
    sptrsv_naive_legacy_kernel<is_upper>(
        grid_size, block_size, 0, exec->get_queue(),
        matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
        as_device_type(matrix->get_const_values()),
        as_device_type(b->get_const_values()), b->get_stride(),
        as_device_type(x->get_values()), x->get_stride(), n, nrhs, unit_diag,
        nan_produced.get_data(), atomic_counter.get_data());
    exec->synchronize();
    std::cout << "finish trs" << std::endl;
#if GKO_VERBOSE_LEVEL >= 1
    if (get_element(nan_produced, 0)) {
        std::cerr
            << "Error: triangular solve produced NaN, either not all diagonal "
               "elements are nonzero, or the system is very ill-conditioned. "
               "The NaN will be replaced with a zero.\n";
    }
#endif  // GKO_VERBOSE_LEVEL >= 1
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_DP_HPP_
