// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/unified/matrix/amp_algorithms.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The AMP matrix format namespace.
 *
 * @ingroup amp
 */
namespace amp {


constexpr int default_block_size = 512;
constexpr int num_thread_blocks_per_cu = 4;
namespace gkerd = gko::kernels::GKO_DEVICE_NAMESPACE;


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::AMP<MatrixValueType, IndexType>* const a,
          const matrix::Dense<InputValueType>* const b,
          matrix::Dense<OutputValueType>* const c)
{
    using highest_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    constexpr bool atomic = 1;
// not support 16 bit atomic
#if !defined(CUDA_VERSION)
    // We do atomic on shared memory
    // If atomic is also true, we also do atomic on out_vector.
    constexpr bool shared_half =
        sizeof(remove_complex<highest_type>) == sizeof(int16);
    constexpr bool atomic_half_out =
        atomic && sizeof(remove_complex<OutputValueType>) == sizeof(int16);
    if constexpr (shared_half || atomic_half_out) {
        GKO_KERNEL_NOT_FOUND;
    }
#else
    constexpr bool shared_half =
        sizeof(remove_complex<highest_type>) == sizeof(half);
    constexpr bool atomic_half_out =
        atomic && sizeof(remove_complex<OutputValueType>) == sizeof(half);
    constexpr bool shared_bfloat16 =
        sizeof(remove_complex<highest_type>) == sizeof(bfloat16);
    constexpr bool atomic_bfloat16_out =
        atomic && sizeof(remove_complex<OutputValueType>) == sizeof(bfloat16);
    const auto compute_capability =
        as<CudaExecutor>(exec)->get_compute_capability();
    if ((shared_half || atomic_half_out) && compute_capability < 70) {
        GKO_KERNEL_NOT_FOUND;
    } else if ((shared_bfloat16 || atomic_bfloat16_out) &&
               compute_capability < 80) {
        GKO_KERNEL_NOT_FOUND;
    }
#endif
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::AMP<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_KERNEL);


template <int q, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_max_nnzs(
    const float tolerance, const size_type nrows, const size_type ostride,
    const size_type omax_nnz, const ValueType* const __restrict__ ovals,
    const IndexType* const __restrict__ ocolids,
    remove_complex<ValueType>* const __restrict__ rownorms,
    int* const __restrict__ max_bin_nnzs_blocks)
{
    using real_type = remove_complex<ValueType>;
    // Compute minimum representable values for each bin
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    // thread-level reduction
    std::array<int, q> max_nnz_thread = {};

    const int start_row = thread::get_thread_id_flat();
    for (int irow = start_row; irow < nrows; irow += gridDim.x) {
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (int j = 0; j < omax_nnz; j++) {
            if (ocolids[j * ostride + irow] == invalid_index<IndexType>()) {
                break;
            } else {
                rnorm += abs(ovals[j * ostride + irow]);
            }
        }
        rownorms[irow] = rnorm;

        // Compute lower limits of each precision bin
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);

        // Get max nnz per row for each precision bin matrix
        std::array<int, q> row_nnz = {};
        for (int j = 0; j < omax_nnz; j++) {
            const int ibin = get_adjusted_bin<real_type>(
                min_bin, min_repr, abs(ovals[j * ostride + irow]));
            if (ibin >= 0) {
                row_nnz[ibin]++;
            }
        }
#pragma unroll
        for (int k = 0; k < q; k++) {
            max_nnz_thread[k] = std::max(max_nnz_thread[k], row_nnz[k]);
        }
    }

    auto warp_tile =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
#pragma unroll
    for (int k = 0; k < q; k++) {
        // warp-reduce
        max_nnz_thread[k] = reduce(warp_tile, max_nnz_thread[k],
                                   [](int a, int b) { return a < b ? b : a; });
    }

    // copy warp sums into shared memory
    const auto warp_id = threadIdx.x / config::warp_size;
    constexpr auto num_warps = default_block_size / config::warp_size;
    __shared__ int warp_max[num_warps * q];
    __syncthreads();
    if (threadIdx.x % config::warp_size == 0) {
#pragma unroll
        for (int k = 0; k < q; k++) {
            warp_max[warp_id + k * num_warps] = max_nnz_thread[k];
        }
    }

    // block reduction: one warp handles the reduction for one precision bucket
    for (int k = warp_id; k < q; k += num_warps) {
        int local = warp_max[warp_tile.thread_rank() + k * num_warps];
        local = reduce(warp_tile, local,
                       [](int a, int b) { return a < b ? b : a; });
        if (warp_tile.thread_rank() == 0) {
            warp_max[k * num_warps] = local;
        }
    }
    __syncthreads();
    if (threadIdx.x < q) {
        max_bin_nnzs_blocks[blockIdx.x + threadIdx.x * gridDim.x] =
            warp_max[threadIdx.x * num_warps];
    }
}

template <int q>
__global__ __launch_bounds__(default_block_size) void finish_reduce(
    int* const __restrict__ max_bins_nnz_blocks, const int stride)
{
    const auto group = group::this_thread_block();
    multireduce(group, max_bins_nnz_blocks, stride, q,
                [](int a, int b) { return a < b ? b : a; });
}

template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<int, ValueType>& max_nnz_per_row,
    array<remove_complex<ValueType>>& rownorms)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = narrow_types<ValueType>::num_types;

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const auto ovals = as_device_type(a->get_const_values());
    const IndexType* const ocolids = a->get_const_col_idxs();

    const auto num_cus = exec->get_num_multiprocessor();
    const auto grid_size = num_cus * num_thread_blocks_per_cu;
    // const auto grid_size = ceildiv(nrows, block_size);
    const auto block_size = default_block_size;
    gko::array<int> max_nnz_arr(exec, q * grid_size);
    max_nnz_arr.fill(0);
    const auto max_nnz_ptr = max_nnz_arr.get_data();
    const auto rownorms_ptr = rownorms.get_data();

    compute_max_nnzs<q><<<grid_size, block_size, 0, exec->get_stream()>>>(
        tolerance, nrows, ostride, omax_nnz, as_device_type(ovals), ocolids,
        as_device_type(rownorms_ptr), max_nnz_ptr);
    finish_reduce<q>
        <<<1, block_size, 0, exec->get_stream()>>>(max_nnz_ptr, grid_size);
    exec->synchronize();

    std::vector<int> max_nnz_host = max_nnz_arr.copy_to_host();
    for (int k = 0; k < q; k++) {
        max_nnz_per_row[k] = max_nnz_host[k * grid_size];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


}  // namespace amp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
