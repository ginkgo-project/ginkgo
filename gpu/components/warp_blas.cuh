/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_GPU_COMPONENTS_WARP_BLAS_CUH_
#define GKO_GPU_COMPONENTS_WARP_BLAS_CUH_


#include "gpu/components/reduction.cuh"


#include <cassert>


namespace gko {
namespace kernels {
namespace gpu {
namespace warp {


/**
 * @internal
 *
 * Applies a Gauss-Jordan transformation (single step of Gauss-Jordan
 * elimination) to a `max_problem_size`-by-`max_problem_size` matrix using
 * `subwarp_size` threads (restrictions from `gpu::warp::reduce` apply, and
 * `max_problem_size` must not be greater than `subwarp_size`.
 * Each thread contributes one `row` of the matrix, and the routine uses warp
 * shuffles to exchange data between rows. The transform is performed by using
 * the `key_row`-th row and `key_col`-th column of the matrix.
 *
 * @note assumes that block dimensions are in "standard format":
 *       (subwarp_size, warp_size / subwarp_size, z)
 */
template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void apply_gauss_jordan_transform(int32 key_row,
                                                             int32 key_col,
                                                             ValueType *row)
{
    static_assert(max_problem_size <= subwarp_size,
                  "max_problem_size cannot be larger than subwarp_size");
    auto key_col_elem = shuffle(row[key_col], key_row, subwarp_size);
    if (key_col_elem == zero<ValueType>()) {
        // TODO: implement error handling for GPUs to be able to properly
        //       report it here
        return;
    }
    if (threadIdx.x == key_row) {
        key_col_elem = one<ValueType>() / key_col_elem;
    } else {
        key_col_elem = -row[key_col] / key_col_elem;
    }
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        const auto key_row_elem = shuffle(row[i], key_row, subwarp_size);
        if (threadIdx.x == key_row) {
            row[i] = zero<ValueType>();
        }
        row[i] += key_col_elem * key_row_elem;
    }
    row[key_col] = key_col_elem;
}


/**
 * @internal
 *
 * Inverts a matrix using Gauss-Jordan elimination. The inversion is
 * done in-place, so the original matrix will be overridden with the inverse.
 * The inversion routine uses implicit pivoting, so the returned matrix will be
 * a permuted inverse (from both sides). To obtain the correct inverse, the
 * rows of the result should be permuted with \f$P\f$, and the columns with
 * \f$ P^T \f$ (i.e.
 * \f$ A^{-1} = P X P \f$, where \f$ X \f$ is the returned matrix). These
 * permutation matrices are returned compressed as vectors `perm` and `tperm`,
 * respectively. `i`-th value of each of the vectors is returned to sub-warp
 * thread with index `i`.
 *
 * @tparam max_problem_size  the maximum problem size that will be passed to the
 *                           inversion routine (a tighter bound results in
 *                           faster code
 * @tparam subwarp_size  the size of the sub-warp used to invert the block,
 *                       cannot be smaller than `max_problem_size`
 * @tparam ValueType  type of values stored in the matrix
 *
 * @param problem_size  the actual size of the matrix (cannot be larger than
 *                      max_problem_size)
 * @param row  a pointer to the matrix row (i-th thread in the subwarp should
 *             pass the pointer to the i-th row), has to have at least
 *             max_problem_size elements
 * @param perm  a value to hold an element of permutation matrix \f$ P \f$
 * @param tperm  a value to hold an element of permutation matrix \f$ P^T \f$
 *
 * @note assumes that block dimensions are in "standard format":
 *       (subwarp_size, warp_size / subwarp_size, z)
 */
template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void invert_block(uint32 problem_size,
                                             ValueType *__restrict__ row,
                                             uint32 &perm, uint32 &tperm)
{
    static_assert(max_problem_size <= subwarp_size,
                  "max_problem_size cannot be larger than subwarp_size");
    assert(problem_size <= max_problem_size);
    // prevent rows after problem_size to become pivots
    auto pivoted = threadIdx.x >= problem_size;
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        const auto piv = choose_pivot<subwarp_size>(row[i], pivoted);
        if (threadIdx.x == piv) {
            perm = i;
            pivoted = true;
        }
        if (threadIdx.x == i) {
            tperm = piv;
        }
        apply_gauss_jordan_transform<max_problem_size, subwarp_size>(piv, i,
                                                                     row);
    }
}


template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void copy_matrix(
    uint32 problem_size, const ValueType *__restrict__ source_row,
    uint32 increment, uint32 rperm, uint32 cperm,
    ValueType *__restrict__ destination, size_type padding)
{
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        const auto idx = warp::shuffle(cperm, i, subwarp_size);
        if (threadIdx.x < problem_size) {
            destination[idx * padding + rperm] = source_row[i * increment];
        }
    }
}


template <int max_problem_size, int subwarp_size, typename ValueType>
__device__ __forceinline__ void multiply_transposed_vec(
    uint32 problem_size, const ValueType &__restrict__ vec,
    const ValueType *__restrict__ mtx_row, uint32 mtx_increment,
    ValueType *__restrict__ res, uint32 res_increment)
{
    auto mtx_elem = zero<ValueType>();
    for (uint32 i = 0; i < problem_size; ++i) {
        if (threadIdx.x < problem_size) {
            mtx_elem = mtx_row[i * mtx_increment];
        }
        const auto out = reduce<subwarp_size>(
            mtx_elem * vec, [](ValueType x, ValueType y) { return x + y; });
        if (threadIdx.x == 0) {
            res[i * res_increment] = out;
        }
    }
}


}  // namespace warp
}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_COMPONENTS_WARP_BLAS_CUH_
