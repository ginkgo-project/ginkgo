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

#ifndef GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_
#define GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_


#include "cuda/base/math.hpp"
#include "cuda/components/reduction.cuh"


#include <cassert>


namespace gko {
namespace kernels {
namespace cuda {


/**
 * @internal
 *
 * Applies a Gauss-Jordan transformation (single step of Gauss-Jordan
 * elimination) to a `max_problem_size`-by-`max_problem_size` matrix using
 * usinge the thread group `group.  Each thread contributes one `row` of the
 * matrix, and the routine uses warp shuffles to exchange data between rows. The
 * transform is performed by using the `key_row`-th row and `key_col`-th column
 * of the matrix.
 */
template <
    int max_problem_size, typename Group, typename ValueType,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ void apply_gauss_jordan_transform(const Group &group,
                                                             int32 key_row,
                                                             int32 key_col,
                                                             ValueType *row)
{
    auto key_col_elem = group.shfl(row[key_col], key_row);
    if (key_col_elem == zero<ValueType>()) {
        // TODO: implement error handling for GPUs to be able to properly
        //       report it here
        return;
    }
    if (group.thread_rank() == key_row) {
        key_col_elem = one<ValueType>() / key_col_elem;
    } else {
        key_col_elem = -row[key_col] / key_col_elem;
    }
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        const auto key_row_elem = group.shfl(row[i], key_row);
        if (group.thread_rank() == key_row) {
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
 * permutation matrices are returned compressed as vectors `perm` and
 * `trans_perm`, respectively. `i`-th value of each of the vectors is returned
 * to thread of the group with rank `i`.
 *
 * @tparam max_problem_size  the maximum problem size that will be passed to the
 *                           inversion routine (a tighter bound results in
 *                           faster code
 * @tparam Group  type of the group of threads
 * @tparam ValueType  type of values stored in the matrix
 *
 * @param group  the group of threads which participate in the inversion
 * @param problem_size  the actual size of the matrix (cannot be larger than
 *                      max_problem_size)
 * @param row  a pointer to the matrix row (i-th thread in the group should
 *             pass the pointer to the i-th row), has to have at least
 *             max_problem_size elements
 * @param perm  a value to hold an element of permutation matrix \f$ P \f$
 * @param trans_perm  a value to hold an element of permutation matrix \f$ P^T
 * \f$
 */
template <
    int max_problem_size, typename Group, typename ValueType,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ void invert_block(const Group &group,
                                             uint32 problem_size,
                                             ValueType *__restrict__ row,
                                             uint32 &__restrict__ perm,
                                             uint32 &__restrict__ trans_perm)
{
    GKO_ASSERT(problem_size <= max_problem_size);
    // prevent rows after problem_size to become pivots
    auto pivoted = group.thread_rank() >= problem_size;
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        const auto piv = choose_pivot(group, row[i], pivoted);
        if (group.thread_rank() == piv) {
            perm = i;
            pivoted = true;
        }
        if (group.thread_rank() == i) {
            trans_perm = piv;
        }
        apply_gauss_jordan_transform<max_problem_size>(group, piv, i, row);
    }
}


/**
 * @internal
 *
 * Copies a matrix stored as a collection of rows in different threads of the
 * warp in a block of memory accessible by all threads in row-major order.
 * Optionally permutes rows and columns of the matrix in the process.
 *
 * @tparam max_problem_size  maximum problem size passed to the routine
 * @tparam Group  type of the group of threads
 * @tparam ValueType  type of values stored in the matrix
 *
 * @param group  group of threads participating in the copy
 * @param problem_size  actual size of the matrix
 *                      (`problem_size <= max_problem_size`)
 * @param source_row  pointer to memory used to store a row of the source matrix
 *                    `i`-th thread of the sub-warp should pass in the `i`-th
 *                    row of the matrix
 * @param increment  offset between two consecutive elements of the row
 * @param row_perm  permutation vector to apply on the rows of the matrix
 *                  (thread `i` supplies the `i`-th value of the vector)
 * @param col_perm  permutation vector to apply on the column of the matrix
 *                  (thread `i` supplies the `i`-th value of the vector)
 * @param destination  pointer to memory where the result will be stored
 *                     (all threads supply the same value)
 * @param stride  offset between two consecutive rows of the matrix
 */
template <
    int max_problem_size, typename Group, typename ValueType,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ void copy_matrix(
    const Group &group, uint32 problem_size,
    const ValueType *__restrict__ source_row, uint32 increment, uint32 row_perm,
    uint32 col_perm, ValueType *__restrict__ destination, size_type stride)
{
    GKO_ASSERT(problem_size <= max_problem_size);
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        const auto idx = group.shfl(col_perm, i);
        if (group.thread_rank() < problem_size) {
            destination[idx * stride + row_perm] = source_row[i * increment];
        }
    }
}


/**
 * @internal
 *
 * Multiplies a transposed vector and a matrix stored in column-major order.
 *
 * In mathematical terms, performs the operation \f$ res^T = vec^T \cdot mtx\f$.
 *
 * @tparam max_problem_size  maximum problem size passed to the routine
 * @tparam Group  type of the group of threads
 * @tparam ValueType  type of values stored in matrix and vectors
 *
 * @param group  group of threads participating in the operation
 * @param problem_size  actual size of the matrix
 *                      (`problem_size <= max_problem_size`)
 * @param vec  input vector to multiply (thread `i` supplies the `i`-th value of
 *             the vector)
 * @param mtx_row  pointer to memory used to store a row of the input matrix,
 *                    `i`-th thread of the sub-warp should pass in the
 *                    `i`-th row of the matrix
 * @param mtx_increment  offset between two consecutive elements of the row
 * @param res  pointer to a block of memory where the result will be written
 *             (only thread 0 of the group has to supply a valid value)
 * @param mtx_increment  offset between two consecutive elements of the result
 */
template <
    int max_problem_size, typename Group, typename ValueType,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ void multiply_transposed_vec(
    const Group &group, uint32 problem_size, const ValueType &__restrict__ vec,
    const ValueType *__restrict__ mtx_row, uint32 mtx_increment,
    ValueType *__restrict__ res, uint32 res_increment)
{
    GKO_ASSERT(problem_size <= max_problem_size);
    auto mtx_elem = zero<ValueType>();
#pragma unroll
    for (int32 i = 0; i < max_problem_size; ++i) {
        if (i >= problem_size) {
            break;
        }
        if (group.thread_rank() < problem_size) {
            mtx_elem = mtx_row[i * mtx_increment];
        }
        const auto out = reduce(group, mtx_elem * vec,
                                [](ValueType x, ValueType y) { return x + y; });
        if (group.thread_rank() == 0) {
            res[i * res_increment] = out;
        }
    }
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_WARP_BLAS_CUH_
