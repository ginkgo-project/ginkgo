// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/diagonal_block_manipulation.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/components/warp_blas.dp.hpp"
#include "dpcpp/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


namespace kernel {


template <int max_block_size, typename ReducedType, typename Group,
          typename ValueType, typename IndexType>
__dpct_inline__ bool validate_precision_reduction_feasibility(
    Group& __restrict__ group, IndexType block_size,
    ValueType* __restrict__ row, ValueType* __restrict__ work, size_type stride)
{
    using gko::detail::float_traits;
    // save original data and reduce precision
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (int i = 0; i < max_block_size; ++i) {
            if (i < block_size) {
                work[i * stride + group.thread_rank()] = row[i];
                row[i] =
                    static_cast<ValueType>(static_cast<ReducedType>(row[i]));
            }
        }
    }

    // compute the condition number
    uint32 perm = group.thread_rank();
    uint32 trans_perm = perm;
    auto block_cond = compute_infinity_norm<max_block_size>(group, block_size,
                                                            block_size, row);
    auto succeeded = invert_block<max_block_size>(
        group, static_cast<uint32>(block_size), row, perm, trans_perm);
    block_cond *= compute_infinity_norm<max_block_size>(group, block_size,
                                                        block_size, row);

    // restore original data
    if (group.thread_rank() < block_size) {
#pragma unroll
        for (int i = 0; i < max_block_size; ++i) {
            if (i < block_size) {
                row[i] = work[i * stride + group.thread_rank()];
            }
        }
    }

    return succeeded && block_cond >= 1.0 &&
           block_cond * static_cast<remove_complex<ValueType>>(
                            float_traits<remove_complex<ValueType>>::eps) <
               remove_complex<ValueType>{1e-3};
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void generate(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, ValueType* __restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, max_block_size * warps_per_block>* workspace)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto block = group::this_thread_block(item_ct1);
    ValueType row[max_block_size];

    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, config::warp_size / subwarp_size, row_ptrs, col_idxs, values,
        block_ptrs, num_blocks, row, 1,
        *workspace + item_ct1.get_local_id(0) * max_block_size, item_ct1);
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    if (block_id < num_blocks) {
        const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
        uint32 perm = subwarp.thread_rank();
        uint32 trans_perm = subwarp.thread_rank();
        invert_block<max_block_size>(subwarp, static_cast<uint32>(block_size),
                                     row, perm, trans_perm);
        copy_matrix<max_block_size, and_transpose>(
            subwarp, block_size, row, 1, perm, trans_perm,
            block_data + storage_scheme.get_global_block_offset(block_id),
            storage_scheme.get_stride());
    }
}

template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void generate(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* values, ValueType* block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* block_ptrs, size_type num_blocks)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<
            uninitialized_array<ValueType, max_block_size * warps_per_block>, 0,
            sycl::access_mode::read_write, sycl::access::target::local>
            workspace_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(subwarp_size)]] {
                    generate<max_block_size, subwarp_size, warps_per_block>(
                        num_rows, row_ptrs, col_idxs, values, block_data,
                        storage_scheme, block_ptrs, num_blocks, item_ct1,
                        workspace_acc_ct1.get_pointer().get());
                });
    });
}

namespace detail {


/**
 * TODO: Less threads involve the verificator1 when the function try calling the
 * verificator1 twice on some CPU. Need to investigate furthermore and get the
 * reproducer. The current workaround we pass the same lambda function as
 * verificator1 to verificator3. Others is copied from
 * core/preconditioner/jacobi_utils.hpp.
 */
template <typename ValueType, typename AccuracyType, typename CondType,
          typename Predicate1, typename Predicate2, typename Predicate3>
GKO_ATTRIBUTES GKO_INLINE uint32 get_supported_storage_reductions_dpcpp(
    AccuracyType accuracy, CondType cond, Predicate1 verificator1,
    Predicate2 verificator2, Predicate3 verificator3)
{
    using gko::detail::float_traits;
    using type = remove_complex<ValueType>;
    using prd = preconditioner::detail::precision_reduction_descriptor;
    auto accurate = [&cond, &accuracy](type eps) {
        return cond * eps < accuracy;
    };
    uint8 is_verified1 = 2;
    auto supported = static_cast<uint32>(prd::p0n0);
    // the following code uses short-circuiting to avoid calling possibly
    // expensive verificatiors multiple times
    if (accurate(float_traits<truncate_type<truncate_type<type>>>::eps)) {
        supported |= prd::p2n0;
    }
    if (accurate(float_traits<truncate_type<reduce_precision<type>>>::eps) &&
        (is_verified1 = verificator1())) {
        supported |= prd::p1n1;
    }
    if (accurate(float_traits<reduce_precision<reduce_precision<type>>>::eps) &&
        is_verified1 != 0 && verificator2()) {
        supported |= prd::p0n2;
    }
    if (accurate(float_traits<truncate_type<type>>::eps)) {
        supported |= prd::p1n0;
    }
    if (accurate(float_traits<reduce_precision<type>>::eps) &&
        (is_verified1 == 1 ||
         (is_verified1 == 2 && (is_verified1 = verificator3())))) {
        supported |= prd::p0n1;
    }
    return supported;
}


}  // namespace detail


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void adaptive_generate(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, remove_complex<ValueType> accuracy,
    ValueType* __restrict__ block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    remove_complex<ValueType>* __restrict__ conditioning,
    precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, max_block_size * warps_per_block>* workspace)
{
    // extract blocks
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto block = group::this_thread_block(item_ct1);
    ValueType row[max_block_size];

    csr::extract_transposed_diag_blocks<max_block_size, warps_per_block>(
        block, config::warp_size / subwarp_size, row_ptrs, col_idxs, values,
        block_ptrs, num_blocks, row, 1,
        *workspace + item_ct1.get_local_id(0) * max_block_size, item_ct1);

    // compute inverse and figure out the correct precision
    const auto subwarp = group::tiled_partition<subwarp_size>(block);
    const uint32 block_size =
        block_id < num_blocks ? block_ptrs[block_id + 1] - block_ptrs[block_id]
                              : 0;
    uint32 perm = subwarp.thread_rank();
    uint32 trans_perm = subwarp.thread_rank();
    auto prec_descriptor = ~uint32{};
    if (block_id < num_blocks) {
        auto block_cond = compute_infinity_norm<max_block_size>(
            subwarp, block_size, block_size, row);
        invert_block<max_block_size>(subwarp, block_size, row, perm,
                                     trans_perm);
        block_cond *= compute_infinity_norm<max_block_size>(subwarp, block_size,
                                                            block_size, row);
        conditioning[block_id] = block_cond;
        const auto prec = block_precisions[block_id];
        prec_descriptor =
            preconditioner::detail::precision_reduction_descriptor::singleton(
                prec);
        if (prec == precision_reduction::autodetect()) {
            using detail::get_supported_storage_reductions_dpcpp;
            prec_descriptor = get_supported_storage_reductions_dpcpp<ValueType>(
                accuracy, block_cond,
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target = reduce_precision<ValueType>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                },
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target =
                        reduce_precision<reduce_precision<ValueType>>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                },
                [&subwarp, &block_size, &row, &block_data, &storage_scheme,
                 &block_id] {
                    using target = reduce_precision<ValueType>;
                    return validate_precision_reduction_feasibility<
                        max_block_size, target>(
                        subwarp, block_size, row,
                        block_data +
                            storage_scheme.get_global_block_offset(block_id),
                        storage_scheme.get_stride());
                });
        }
    }

    // make sure all blocks in the group have the same precision
    const auto warp = group::tiled_partition<config::warp_size>(block);
    const auto prec = preconditioner::detail::get_optimal_storage_reduction(
        ::gko::kernels::dpcpp::reduce(
            warp, prec_descriptor, [](uint32 x, uint32 y) { return x & y; }));

    // store the block back into memory
    if (block_id < num_blocks) {
        block_precisions[block_id] = prec;
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, prec,
            copy_matrix<max_block_size, and_transpose>(
                subwarp, block_size, row, 1, perm, trans_perm,
                reinterpret_cast<resolved_precision*>(
                    block_data + storage_scheme.get_group_offset(block_id)) +
                    storage_scheme.get_block_offset(block_id),
                storage_scheme.get_stride()));
    }
}

template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void adaptive_generate(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* values, remove_complex<ValueType> accuracy,
    ValueType* block_data,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    remove_complex<ValueType>* conditioning,
    precision_reduction* block_precisions, const IndexType* block_ptrs,
    size_type num_blocks)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<
            uninitialized_array<ValueType, max_block_size * warps_per_block>, 0,
            sycl::access_mode::read_write, sycl::access::target::local>
            workspace_acc_ct1(cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(subwarp_size)]] {
                                 adaptive_generate<max_block_size, subwarp_size,
                                                   warps_per_block>(
                                     num_rows, row_ptrs, col_idxs, values,
                                     accuracy, block_data, storage_scheme,
                                     conditioning, block_precisions, block_ptrs,
                                     num_blocks, item_ct1,
                                     workspace_acc_ct1.get_pointer().get());
                             });
    });
}


}  // namespace kernel


// clang-format off
#cmakedefine GKO_JACOBI_BLOCK_SIZE @GKO_JACOBI_BLOCK_SIZE@
// clang-format on
// make things easier for IDEs
#ifndef GKO_JACOBI_BLOCK_SIZE
#define GKO_JACOBI_BLOCK_SIZE 1
#endif


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::value_list<int, max_block_size>,
              std::shared_ptr<const DefaultExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* mtx,
              remove_complex<ValueType> accuracy, ValueType* block_data,
              const preconditioner::block_interleaved_storage_scheme<IndexType>&
                  storage_scheme,
              remove_complex<ValueType>* conditioning,
              precision_reduction* block_precisions,
              const IndexType* block_ptrs, size_type num_blocks)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_generate<max_block_size, subwarp_size,
                                  warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), mtx->get_size()[0],
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            mtx->get_const_values(), accuracy, block_data, storage_scheme,
            conditioning, block_precisions, block_ptrs, num_blocks);
    } else {
        kernel::generate<max_block_size, subwarp_size, warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), mtx->get_size()[0],
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            mtx->get_const_values(), block_data, storage_scheme, block_ptrs,
            num_blocks);
    }
}


#define DECLARE_JACOBI_GENERATE_INSTANTIATION(ValueType, IndexType)          \
    void generate<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE,        \
                  ValueType, IndexType>(                                     \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>,                         \
        std::shared_ptr<const DefaultExecutor>,                              \
        const matrix::Csr<ValueType, IndexType>*, remove_complex<ValueType>, \
        ValueType*,                                                          \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&,  \
        remove_complex<ValueType>*, precision_reduction*, const IndexType*,  \
        size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_GENERATE_INSTANTIATION);


}  // namespace jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
