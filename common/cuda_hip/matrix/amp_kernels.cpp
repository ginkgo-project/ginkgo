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


constexpr uint32 default_block_size = 512;
constexpr uint32 num_thread_blocks_per_cu = 4;
namespace gkerd = gko::kernels::GKO_DEVICE_NAMESPACE;

// Tuple of device const pointers to relevant scalar types
template <typename highest_type>
using ScalarDCPtrTuple =
    gko::instantiation_tuple_t<gko::generator<gko::ptr_to_const_type>,
                               typename narrow_types<highest_type>::type>;

// spmv kernel: 1 thread block-column per RHS
// TODO: Optimize cache usage using intrinsics
template <typename IValueType, typename MValueType, typename OValueType,
          typename IndexType, typename InitialCombiner, typename Combiner>
__device__ void ell_amp_spmv_impl(
    const size_type nrows, const uint32 nrhs,
    precision_array<size_type, MValueType> bin_strides,
    precision_array<size_type, MValueType> bin_max_nnz_row,
    precision_array<const IndexType*, MValueType> bin_col_idxs,
    ScalarDCPtrTuple<MValueType> bin_values, const uint32 x_stride,
    const IValueType* const __restrict__ x, const uint32 y_stride,
    OValueType* const __restrict__ y, InitialCombiner initial_op, Combiner op)
{
    constexpr int q = narrow_types<MValueType>::num_types;
    const auto irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow >= nrows) {
        return;
    }
    const auto irhs = blockIdx.y;
    if (irhs >= nrhs) {
        return;
    }
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename narrow_types<MValueType>::type>::type;
        // We need mult type because complex numbers of different precisions
        // don't get automatically promoted.
        using mult_type = gko::highest_precision<value_type, IValueType>;
        using highest_type = gko::highest_precision<mult_type, OValueType>;
        const auto stride = bin_strides[k];
        auto avals = std::get<k>(bin_values);
        auto acols = bin_col_idxs[k];
        const auto max_nnz = bin_max_nnz_row[k];
        if (max_nnz > 0) {
            highest_type sum = 0;
            for (int j = 0; j < max_nnz; j++) {
                if (acols[irow + j * stride] >= 0) {
                    sum += static_cast<highest_type>(
                        static_cast<mult_type>(avals[irow + j * stride]) *
                        static_cast<mult_type>(
                            x[acols[irow + j * stride] * x_stride + irhs]));
                }
            }
            if constexpr (k == 0) {
                y[irow * y_stride + irhs] =
                    initial_op(sum, y[irow * y_stride + irhs]);
            } else {
                y[irow * y_stride + irhs] += op(sum);
            }
        }
    });
}

template <typename IValueType, typename MValueType, typename OValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void ell_amp_basic_spmv(
    const size_type nrows, const uint32 nrhs,
    precision_array<size_type, MValueType> bin_strides,
    precision_array<size_type, MValueType> bin_max_nnz_row,
    precision_array<const IndexType*, MValueType> bin_col_idxs,
    ScalarDCPtrTuple<MValueType> bin_values, const uint32 x_stride,
    const IValueType* const __restrict__ x, const uint32 y_stride,
    OValueType* const __restrict__ y)
{
    ell_amp_spmv_impl<IValueType, MValueType, OValueType, IndexType>(
        nrows, nrhs, bin_strides, bin_max_nnz_row, bin_col_idxs, bin_values,
        x_stride, x, y_stride, y, [](auto sum, auto& x) { return sum; },
        [](auto x) { return x; });
}

template <typename IValueType, typename MValueType, typename OValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void ell_amp_adv_spmv(
    const size_type nrows, const uint32 nrhs,
    const MValueType* const __restrict__ alpha,
    precision_array<size_type, MValueType> bin_strides,
    precision_array<size_type, MValueType> bin_max_nnz_row,
    precision_array<const IndexType*, MValueType> bin_col_idxs,
    ScalarDCPtrTuple<MValueType> bin_values, const uint32 x_stride,
    const IValueType* const __restrict__ x,
    const OValueType* const __restrict__ beta, const uint32 y_stride,
    OValueType* const __restrict__ y)
{
    using highest_type =
        gko::highest_precision<IValueType, MValueType, OValueType>;
    const auto alval = static_cast<highest_type>(alpha[0]);
    const auto beval = beta[0];
    ell_amp_spmv_impl<IValueType, MValueType, OValueType, IndexType>(
        nrows, nrhs, bin_strides, bin_max_nnz_row, bin_col_idxs, bin_values,
        x_stride, x, y_stride, y,
        [alval, beval](auto sum, OValueType& x) {
            return static_cast<OValueType>(
                beval * x + alval * static_cast<highest_type>(sum));
        },
        [alval](auto sum) {
            return static_cast<OValueType>(alval *
                                           static_cast<highest_type>(sum));
        });
}

template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::AMP<MatrixValueType, IndexType>* const a,
          const matrix::Dense<InputValueType>* const b,
          matrix::Dense<OutputValueType>* const c)
{
    using DMValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<MatrixValueType>;
    using DIValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<InputValueType>;
    using DOValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<OutputValueType>;

    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto c_ptr = as_device_type(c->get_values());
    auto b_ptr = as_device_type(b->get_const_values());
    const auto nrows = a->get_size()[0];
    const auto nrhs = static_cast<uint32>(c->get_size()[1]);

    // Get precision buckets' arrays
    ScalarDCPtrTuple<DMValueType> xvalues;
    precision_array<const IndexType*, DMValueType> xcol_idxs;
    precision_array<size_type, DMValueType> bin_strides;
    precision_array<size_type, DMValueType> max_nnzs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using EllType = matrix::Ell<value_type, IndexType>;
        auto ematk = dynamic_cast<const EllType*>(a->get_bin_matrix(k));
        if (!ematk) {
            GKO_NOT_SUPPORTED(ematk);
        }
        xcol_idxs[k] = ematk->get_const_col_idxs();
        bin_strides[k] = ematk->get_stride();
        max_nnzs[k] = ematk->get_num_stored_elements_per_row();
        std::get<k>(xvalues) = as_device_type(ematk->get_const_values());
    });

    constexpr auto block_size = default_block_size;
    const auto num_blocks = static_cast<uint32>(ceildiv(nrows, block_size));
    const dim3 grid{num_blocks, nrhs, 1};
    ell_amp_basic_spmv<DIValueType, DMValueType, DOValueType, IndexType>
        <<<grid, block_size, 0, exec->get_stream()>>>(
            nrows, nrhs, bin_strides, max_nnzs, xcol_idxs, xvalues,
            static_cast<uint32>(b->get_stride()), b_ptr,
            static_cast<uint32>(c->get_stride()), c_ptr);
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
    using DMValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<MatrixValueType>;
    using DIValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<InputValueType>;
    using DOValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<OutputValueType>;

    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto c_ptr = as_device_type(c->get_values());
    auto b_ptr = as_device_type(b->get_const_values());
    auto alpha_ptr = as_device_type(alpha->get_const_values());
    auto beta_ptr = as_device_type(beta->get_const_values());
    const auto nrows = a->get_size()[0];
    const auto nrhs = static_cast<uint32>(c->get_size()[1]);

    // Get precision buckets' arrays
    ScalarDCPtrTuple<DMValueType> xvalues;
    precision_array<const IndexType*, DMValueType> xcol_idxs;
    precision_array<size_type, DMValueType> bin_strides;
    precision_array<size_type, DMValueType> max_nnzs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using EllType = matrix::Ell<value_type, IndexType>;
        auto ematk = dynamic_cast<const EllType*>(a->get_bin_matrix(k));
        if (!ematk) {
            GKO_NOT_SUPPORTED(ematk);
        }
        xcol_idxs[k] = ematk->get_const_col_idxs();
        bin_strides[k] = ematk->get_stride();
        max_nnzs[k] = ematk->get_num_stored_elements_per_row();
        std::get<k>(xvalues) = as_device_type(ematk->get_const_values());
    });

    constexpr auto block_size = default_block_size;
    const auto num_blocks = static_cast<uint32>(ceildiv(nrows, block_size));
    const dim3 grid{num_blocks, nrhs, 1};
    ell_amp_adv_spmv<DIValueType, DMValueType, DOValueType, IndexType>
        <<<grid, block_size, 0, exec->get_stream()>>>(
            nrows, nrhs, alpha_ptr, bin_strides, max_nnzs, xcol_idxs, xvalues,
            static_cast<uint32>(b->get_stride()), b_ptr, beta_ptr,
            static_cast<uint32>(c->get_stride()), c_ptr);
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
    int max_nnz_thread[q];
#pragma unroll
    for (int i = 0; i < q; i++) {
        max_nnz_thread[i] = 0;
    }

    const int start_row = thread::get_thread_id_flat();
    for (int irow = start_row; irow < nrows; irow += gridDim.x * blockDim.x) {
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
        int row_nnz[q];
#pragma unroll
        for (int i = 0; i < q; i++) {
            row_nnz[i] = 0;
        }
        for (int j = 0; j < omax_nnz; j++) {
            const int ibin = get_adjusted_bin<real_type>(
                min_bin, min_repr, abs(ovals[j * ostride + irow]));
            if (ibin >= 0) {
                row_nnz[ibin]++;
            }
        }
#pragma unroll
        for (int k = 0; k < q; k++) {
            max_nnz_thread[k] = max(max_nnz_thread[k], row_nnz[k]);
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
    __syncthreads();

    // block reduction: one warp handles the reduction for one precision bucket
    for (int k = warp_id; k < q; k += num_warps) {
        int local = warp_tile.thread_rank() < num_warps
                        ? warp_max[warp_tile.thread_rank() + k * num_warps]
                        : 0;
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
    int* const __restrict__ data, const int len, const int stride)
{
    const auto group = group::this_thread_block();
    // multireduce(group, max_bins_nnz_blocks, stride, q,
    //             [](int a, int b) { return a < b ? b : a; });
    const auto local_id = group.thread_rank();

    for (int k = group.size() / 2; k >= config::warp_size; k /= 2) {
        group.sync();
        if (local_id < k && local_id < len) {
            for (int j = 0; j < q; j++) {
                const int a = data[j * stride + local_id];
                const int b =
                    (local_id + k < len) ? data[j * stride + local_id + k] : 0;
                const int ans = max(a, b);
                data[j * stride + local_id] = ans;
            }
        }
    }

    group.sync();

    const auto warp = group::tiled_partition<config::warp_size>(group);
    const auto warp_id = group.thread_rank() / warp.size();
    if (warp_id > 0) {
        return;
    }
    for (int j = 0; j < q; j++) {
        auto val = warp.thread_rank() < len
                       ? data[j * stride + warp.thread_rank()]
                       : 0;
        auto result = reduce(warp, val, [](int a, int b) { return max(a, b); });
        if (warp.thread_rank() == 0) {
            data[j * stride] = result;
        }
    }
}

template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<int, ValueType>& max_nnz_per_row,
    array<gko::remove_complex<ValueType>>& rownorms)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = narrow_types<ValueType>::num_types;

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const auto ovals = as_device_type(a->get_const_values());
    const IndexType* const ocolids = a->get_const_col_idxs();

    const auto num_cus = exec->get_num_multiprocessor();
    const auto num_blocks = num_cus * num_thread_blocks_per_cu;
    // const auto grid_size = ceildiv(nrows, block_size);
    const auto block_size = default_block_size;
    gko::array<int> max_nnz_arr(exec, q * num_blocks);
    max_nnz_arr.fill(0);
    const auto max_nnz_ptr = max_nnz_arr.get_data();
    const auto rownorms_ptr = rownorms.get_data();

    compute_max_nnzs<q><<<num_blocks, block_size, 0, exec->get_stream()>>>(
        tolerance, nrows, ostride, omax_nnz, as_device_type(ovals), ocolids,
        as_device_type(rownorms_ptr), max_nnz_ptr);
    finish_reduce<q><<<1, block_size, 0, exec->get_stream()>>>(
        max_nnz_ptr, num_blocks, num_blocks);
    exec->synchronize();

    std::vector<int> max_nnz_host = max_nnz_arr.copy_to_host();
    for (int k = 0; k < q; k++) {
        max_nnz_per_row[k] = max_nnz_host[k * num_blocks];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


}  // namespace amp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
