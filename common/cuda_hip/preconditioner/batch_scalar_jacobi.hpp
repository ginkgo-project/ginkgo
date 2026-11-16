// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_
#define GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_preconditioner {


/**
 * (Scalar) Jacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class ScalarJacobi final {
public:
    using value_type = ValueType;
    using index_type = int;
    /**
     * The type of the work vector and of the vectors it is applied to, which is
     * the type the solvers compute in.
     */
    using arithmetic_type = device_type<device_arithmetic_type<value_type>>;

    /**
     * The size of the work vector (in bytes) required in case of dynamic
     * allocation.
     */
    __host__ __device__ static constexpr int dynamic_work_size(
        const int num_rows, int)
    {
        return num_rows * sizeof(arithmetic_type);
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by dynamic_work_size.
     */
    template <typename MatValues>
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::ell::batch_item<
            const value_type, const index_type, MatValues>& mat,
        arithmetic_type* const __restrict__ work)
    {
        const auto col = mat.col_idxs;
        const auto val = mat.values;
        work_ = work;
        for (int tidx = threadIdx.x; tidx < mat.num_rows; tidx += blockDim.x) {
            auto temp = one<arithmetic_type>();
            for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
                const auto ind = tidx + idx * mat.stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    if (col_idx == tidx) {
                        temp = one<arithmetic_type>() / val[ind];
                    }
                }
            }
            work_[tidx] = temp;
        }
        __syncthreads();
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by dynamic_work_size.
     */
    template <typename MatValues>
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::csr::batch_item<
            const value_type, const index_type, MatValues>& mat,
        arithmetic_type* const __restrict__ work)
    {
        work_ = work;
        constexpr auto warp_size = config::warp_size;
        const auto tile =
            group::tiled_partition<warp_size>(group::this_thread_block());
        const int tile_rank = threadIdx.x / warp_size;
        const int num_tiles = (blockDim.x - 1) / warp_size + 1;
        for (int irow = tile_rank; irow < mat.num_rows; irow += num_tiles) {
            for (int j = mat.row_ptrs[irow] + tile.thread_rank();
                 j < mat.row_ptrs[irow + 1]; j += warp_size) {
                if (mat.col_idxs[j] == irow) {
                    const arithmetic_type diag = mat.values[j];
                    work_[irow] = (diag != zero<arithmetic_type>())
                                      ? one<arithmetic_type>() / diag
                                      : one<arithmetic_type>();
                }
            }
        }
        __syncthreads();
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by dynamic_work_size.
     */
    template <typename MatValues>
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::dense::batch_item<const value_type,
                                                    MatValues>& mat,
        arithmetic_type* const __restrict__ work)
    {
        work_ = work;
        constexpr auto warp_size = config::warp_size;
        const auto tile =
            group::tiled_partition<warp_size>(group::this_thread_block());
        const int tile_rank = threadIdx.x / warp_size;
        const int num_tiles = (blockDim.x - 1) / warp_size + 1;
        for (int irow = tile_rank; irow < mat.num_rows; irow += num_tiles) {
            const int iz = irow * static_cast<int>(mat.stride) + irow;
            const arithmetic_type diag = mat.values[iz];
            work_[irow] = (diag != zero<arithmetic_type>())
                              ? one<arithmetic_type>() / diag
                              : one<arithmetic_type>();
        }
        __syncthreads();
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const arithmetic_type* const r,
                                          arithmetic_type* const z) const
    {
        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            z[i] = work_[i] * r[i];
        }
    }

private:
    arithmetic_type* __restrict__ work_;
};


}  // namespace batch_preconditioner
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
