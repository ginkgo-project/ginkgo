// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/device_limits.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/merging.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "common/cuda_hip/components/warp_blas.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Isai preconditioner namespace.
 * @ref Isai
 */
namespace isai {


constexpr int subwarp_size{row_size_limit};
constexpr int subwarps_per_block{2};
constexpr int default_block_size{subwarps_per_block * subwarp_size};


namespace kernel {


/**
 * @internal
 *
 * This kernel supports at most `subwarp_size` (< `warp_size`) elements per row.
 * If there are more elements, they are simply ignored. Only the first
 * `subwarp_size` elements are considered both for the values and for the
 * sparsity pattern.
 */
template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType, typename Callable>
__forceinline__ __device__ void generic_generate(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values,
    IndexType* __restrict__ excess_rhs_sizes,
    IndexType* __restrict__ excess_nnz, Callable direct_solve)
{
    static_assert(subwarp_size >= row_size_limit, "incompatible subwarp_size");
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    if (row >= num_rows) {
        return;
    }

    const auto i_row_begin = i_row_ptrs[row];
    const auto i_row_size = i_row_ptrs[row + 1] - i_row_begin;

    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int local_id = subwarp.thread_rank();

    if (i_row_size > subwarp_size) {
        // defer long rows: store their nnz and number of matches
        IndexType count{};
        for (IndexType nz = 0; nz < i_row_size; ++nz) {
            auto col = i_col_idxs[i_row_begin + nz];
            auto m_row_begin = m_row_ptrs[col];
            auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
            // extract the sparse submatrix consisting of the entries whose
            // columns/rows match column indices from this row
            group_match<subwarp_size>(
                m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
                i_row_size, subwarp,
                [&](IndexType, IndexType, IndexType,
                    config::lane_mask_type matchmask,
                    bool) { count += popcnt(matchmask); });
        }
        // store the dim and nnz of this sparse block
        if (local_id == 0) {
            excess_rhs_sizes[row] = i_row_size;
            excess_nnz[row] = count;
        }
    } else {
        // handle short rows directly: no excess
        if (local_id == 0) {
            excess_rhs_sizes[row] = 0;
            excess_nnz[row] = 0;
        }

        // subwarp_size^2 storage per subwarp
        __shared__ uninitialized_array<ValueType, subwarp_size * subwarp_size *
                                                      subwarps_per_block>
            storage;

        auto dense_system_ptr = storage + (threadIdx.x / subwarp_size) *
                                              subwarp_size * subwarp_size;
        // row-major accessor
        auto dense_system = [&](IndexType row, IndexType col) -> ValueType& {
            return dense_system_ptr[row * subwarp_size + col];
        };

#pragma unroll
        for (int i = 0; i < subwarp_size; ++i) {
            dense_system(i, local_id) = zero<ValueType>();
        }

        subwarp.sync();

        IndexType rhs_one_idx{};
        for (IndexType nz = 0; nz < i_row_size; ++nz) {
            auto col = i_col_idxs[i_row_begin + nz];
            auto m_row_begin = m_row_ptrs[col];
            auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
            // extract the dense submatrix consisting of the entries whose
            // columns/rows match column indices from this row within the
            // sparsity pattern of the original matrix (matches outside of that
            // are zero)
            group_match<subwarp_size>(
                m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
                i_row_size, subwarp,
                [&](IndexType, IndexType m_idx, IndexType i_idx,
                    config::lane_mask_type, bool valid) {
                    if (valid) {
                        dense_system(nz, i_idx) = m_values[m_row_begin + m_idx];
                    }
                });
            const auto i_transposed_row_begin = i_row_ptrs[col];
            const auto i_transposed_row_size =
                i_row_ptrs[col + 1] - i_transposed_row_begin;
            // Loop over all matches that are within the sparsity pattern of
            // the inverse to find the index of the one in the right-hand-side
            group_match<subwarp_size>(
                i_col_idxs + i_transposed_row_begin, i_transposed_row_size,
                i_col_idxs + i_row_begin, i_row_size, subwarp,
                [&](IndexType, IndexType m_idx, IndexType i_idx,
                    config::lane_mask_type, bool valid) {
                    rhs_one_idx += popcnt(group::ballot(
                        subwarp,
                        valid &&
                            i_col_idxs[i_transposed_row_begin + m_idx] < row &&
                            col == row));
                });
        }

        subwarp.sync();

        // Now, read a full col of `dense_system` into local registers, which
        // will be row elements after this (implicit) transpose
        ValueType local_row[subwarp_size];
#pragma unroll
        for (int i = 0; i < subwarp_size; ++i) {
            local_row[i] = dense_system(i, local_id);
        }

        const auto rhs =
            direct_solve(i_row_size, local_row, subwarp, rhs_one_idx);

        // Write back:
        if (local_id < i_row_size) {
            const auto idx = i_row_begin + local_id;
            if (is_finite(rhs)) {
                i_values[idx] = rhs;
            } else {
                i_values[idx] = i_col_idxs[idx] == row ? one<ValueType>()
                                                       : zero<ValueType>();
            }
        }
    }
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_l_inverse(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values,
    IndexType* __restrict__ excess_rhs_sizes,
    IndexType* __restrict__ excess_nnz)
{
    auto trs_solve =
        [](IndexType num_elems, const ValueType* __restrict__ local_row,
           group::thread_block_tile<subwarp_size>& subwarp, size_type) {
            const int local_id = subwarp.thread_rank();
            ValueType rhs = local_id == num_elems - 1 ? one<ValueType>()
                                                      : zero<ValueType>();
            // Solve Triangular system
            for (int d_col = num_elems - 1; d_col >= 0; --d_col) {
                const auto elem = local_row[d_col];
                if (d_col == local_id) {
                    rhs /= elem;
                }

                const ValueType bot = subwarp.shfl(rhs, d_col);
                if (local_id < d_col) {
                    rhs -= bot * elem;
                }
            }

            return rhs;
        };

    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, excess_rhs_sizes, excess_nnz, trs_solve);
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_u_inverse(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values,
    IndexType* __restrict__ excess_rhs_sizes,
    IndexType* __restrict__ excess_nnz)
{
    auto trs_solve = [](IndexType num_elems,
                        const ValueType* __restrict__ local_row,
                        group::thread_block_tile<subwarp_size>& subwarp,
                        size_type) {
        const int local_id = subwarp.thread_rank();
        ValueType rhs = local_id == 0 ? one<ValueType>() : zero<ValueType>();
        // Solve Triangular system
        for (int d_col = 0; d_col < num_elems; ++d_col) {
            const auto elem = local_row[d_col];
            if (d_col == local_id) {
                rhs /= elem;
            }

            const ValueType top = subwarp.shfl(rhs, d_col);
            if (d_col < local_id) {
                rhs -= top * elem;
            }
        }

        return rhs;
    };

    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, excess_rhs_sizes, excess_nnz, trs_solve);
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_general_inverse(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values,
    IndexType* __restrict__ excess_rhs_sizes,
    IndexType* __restrict__ excess_nnz, bool spd)
{
    // __device__ is important to hip. Without __device__, hip might consider it
    // is a normal function first such that it ends up some ambiguous sqrt from
    // the system header for bfloat16. Using __device__ helps only search the
    // device part and we will not use float/long double from the system for
    // bfloat16.
    auto general_solve =
        [spd] __device__(IndexType num_elems, ValueType* __restrict__ local_row,
                         group::thread_block_tile<subwarp_size> & subwarp,
                         size_type rhs_one_idx) {
            const int local_id = subwarp.thread_rank();
            ValueType rhs =
                local_id == rhs_one_idx ? one<ValueType>() : zero<ValueType>();
            size_type perm = local_id;
            auto pivoted = subwarp.thread_rank() >= num_elems;
            auto status = true;
            for (size_type i = 0; i < num_elems; i++) {
                const auto piv = choose_pivot(subwarp, local_row[i], pivoted);
                if (local_id == piv) {
                    pivoted = true;
                }
                if (local_id == i) {
                    perm = piv;
                }

                apply_gauss_jordan_transform_with_rhs<subwarp_size>(
                    subwarp, piv, i, local_row, &rhs, status);
            }

            ValueType sol = subwarp.shfl(rhs, perm);

            if (spd) {
                auto diag = subwarp.shfl(sol, num_elems - 1);
                sol /= sqrt(diag);
            }

            return sol;
        };

    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, excess_rhs_sizes, excess_nnz, general_solve);
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_excess_system(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs,
    const IndexType* __restrict__ excess_rhs_ptrs,
    const IndexType* __restrict__ excess_nz_ptrs,
    IndexType* __restrict__ excess_row_ptrs,
    IndexType* __restrict__ excess_col_idxs,
    ValueType* __restrict__ excess_values, ValueType* __restrict__ excess_rhs,
    size_type e_start, size_type e_end)
{
    const auto row =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>() + e_start;

    if (row >= e_end) {
        return;
    }

    const auto i_row_begin = i_row_ptrs[row];
    const auto i_row_size = i_row_ptrs[row + 1] - i_row_begin;

    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int local_id = subwarp.thread_rank();
    const auto prefix_mask = (config::lane_mask_type{1} << local_id) - 1;

    if (row == e_start && local_id == 0) {
        excess_row_ptrs[0] = 0;
    }

    if (i_row_size <= subwarp_size) {
        return;
    }

    auto excess_rhs_begin = excess_rhs_ptrs[row];
    auto excess_nz_begin = excess_nz_ptrs[row];

    auto out_nz_begin = excess_nz_begin - excess_nz_ptrs[e_start];
    auto out_ptrs_begin = excess_rhs_begin - excess_rhs_ptrs[e_start];

    // defer long rows: store their nnz and number of matches
    for (IndexType nz = 0; nz < i_row_size; ++nz) {
        auto col = i_col_idxs[i_row_begin + nz];
        auto m_row_begin = m_row_ptrs[col];
        auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
        // extract the sparse submatrix consisting of the entries whose
        // columns/rows match column indices from this row
        group_match<subwarp_size>(
            m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
            i_row_size, subwarp,
            [&](IndexType col, IndexType m_idx, IndexType i_idx,
                config::lane_mask_type mask, bool valid) {
                // dense_system(nz, i_idx) = m_values[m_row_begin + m_idx]
                // only in sparse :)
                if (valid) {
                    auto nz = out_nz_begin + popcnt(mask & prefix_mask);
                    excess_col_idxs[nz] = out_ptrs_begin + i_idx;
                    excess_values[nz] = m_values[m_row_begin + m_idx];
                }
                out_nz_begin += popcnt(mask);
            });
        if (local_id == 0) {
            // build right-hand side: 1 for diagonal entry, 0 else
            excess_rhs[out_ptrs_begin + nz] =
                row == col ? one<ValueType>() : zero<ValueType>();
            // store row pointers
            excess_row_ptrs[out_ptrs_begin + nz + 1] = out_nz_begin;
        }
    }
}

template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void scale_excess_solution(
    const IndexType* __restrict__ excess_block_ptrs,
    ValueType* __restrict__ excess_solution, size_type e_start, size_type e_end)
{
    const auto warp_id = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int local_id = subwarp.thread_rank();
    const auto row = warp_id + e_start;

    if (row >= e_end) {
        return;
    }

    const IndexType offset = excess_block_ptrs[e_start];
    const IndexType block_begin = excess_block_ptrs[row] - offset;
    const IndexType block_end = excess_block_ptrs[row + 1] - offset;
    if (block_end == block_begin) {
        return;
    }
    const auto diag = excess_solution[block_end - 1];
    const ValueType scal = one<ValueType>() / sqrt(diag);

    for (size_type i = block_begin + local_id; i < block_end;
         i += subwarp_size) {
        excess_solution[i] *= scal;
    }
}

template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void copy_excess_solution(
    IndexType num_rows, const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ excess_rhs_ptrs,
    const ValueType* __restrict__ excess_solution,
    ValueType* __restrict__ i_values, size_type e_start, size_type e_end)
{
    const auto excess_row =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    const auto row = excess_row + e_start;

    if (row >= e_end) {
        return;
    }

    auto local_id = threadIdx.x % subwarp_size;

    const auto i_row_begin = i_row_ptrs[row];

    const auto excess_begin = excess_rhs_ptrs[row];
    const auto excess_size = excess_rhs_ptrs[row + 1] - excess_begin;

    // if it was handled separately:
    if (excess_size > 0) {
        // copy the values for this row
        for (IndexType nz = local_id; nz < excess_size; nz += subwarp_size) {
            i_values[nz + i_row_begin] =
                excess_solution[nz + excess_begin - excess_rhs_ptrs[e_start]];
        }
    }
}


constexpr int spd_t1_block_size = 128;
constexpr int spd_t2_block_size = 256;
// longest row handled in shared memory (if the device has enough of it)
constexpr int spd_max_t1_size = 128;


__host__ __device__ __forceinline__ int64 spd_packed(int64 a, int64 b)
{
    return a * (a + 1) / 2 + b;
}


/**
 * @internal
 *
 * SPD ISAI rows with at most `subwarp_size` entries, one subwarp per row:
 * M(a, b) = conj(A(S_a, S_b)) in a column-major shared tile, M = C C^H,
 * C^H z = e_k. Longer rows are skipped.
 */
template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__
__launch_bounds__(subwarp_size* subwarps_per_block) void spd_inverse_t0(
    IndexType num_rows, const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto begin = i_row_ptrs[row];
    const auto size = i_row_ptrs[row + 1] - begin;
    if (size > subwarp_size) {
        return;
    }
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int lane = subwarp.thread_rank();
    __shared__ uninitialized_array<ValueType, subwarp_size * subwarp_size *
                                                  subwarps_per_block>
        storage;
    ValueType* tile =
        storage + (threadIdx.x / subwarp_size) * subwarp_size * subwarp_size;
    // tile[col * subwarp_size + row] holds M(row, col)
    for (int i = 0; i < subwarp_size; ++i) {
        tile[i * subwarp_size + lane] = zero<ValueType>();
    }
    subwarp.sync();
    const auto pattern = i_col_idxs + begin;
    for (IndexType a = 0; a < size; ++a) {
        const auto m_begin = m_row_ptrs[pattern[a]];
        // M(b, a) = A(S_a, S_b) on the lane that matched S_b
        group_match<subwarp_size>(
            m_col_idxs + m_begin, m_row_ptrs[pattern[a] + 1] - m_begin, pattern,
            size, subwarp,
            [&](IndexType, IndexType m_idx, IndexType b, config::lane_mask_type,
                bool valid) {
                if (valid) {
                    tile[a * subwarp_size + b] = m_values[m_begin + m_idx];
                }
            });
    }
    subwarp.sync();
    bool valid = true;
    for (int j = 0; j < size && valid; ++j) {
        const auto diag = real(tile[j * subwarp_size + j]);
        valid = diag > zero<remove_complex<ValueType>>() && is_finite(diag);
        if (valid) {
            const auto c = sqrt(diag);
            subwarp.sync();
            ValueType l_ij{};
            if (lane > j && lane < size) {
                l_ij = tile[j * subwarp_size + lane] / c;
                tile[j * subwarp_size + lane] = l_ij;
            }
            if (lane == j) {
                tile[j * subwarp_size + j] = c;
            }
            subwarp.sync();
            if (lane > j && lane < size) {
                for (int l = j + 1; l <= lane; ++l) {
                    tile[l * subwarp_size + lane] -=
                        l_ij * conj(tile[j * subwarp_size + l]);
                }
            }
            subwarp.sync();
        }
    }
    // row-oriented backward solve, lane i keeps z_i
    ValueType z{};
    if (valid) {
        for (int i = size - 1; i >= 0; --i) {
            const auto partial = lane > i && lane < size
                                     ? conj(tile[i * subwarp_size + lane]) * z
                                     : zero<ValueType>();
            const auto sum =
                reduce(subwarp, partial,
                       [](ValueType x, ValueType y) { return x + y; });
            const auto z_i =
                ((i == size - 1 ? one<ValueType>() : zero<ValueType>()) - sum) /
                tile[i * subwarp_size + i];
            if (lane == i) {
                z = z_i;
            }
        }
        valid = !subwarp.any(lane < size && !is_finite(z));
    }
    if (lane < size) {
        i_values[begin + lane] =
            valid
                ? z
                : (pattern[lane] == row ? one<ValueType>() : zero<ValueType>());
    }
}


/**
 * @internal
 *
 * One SPD ISAI row by a whole block: packed lower M in `m`, the solution in
 * `z` (both shared or global memory), `pattern` the row's column indices.
 */
template <int block_size, typename ValueType, typename IndexType>
__device__ __forceinline__ void spd_inverse_block_row(
    IndexType row, IndexType size, const IndexType* pattern,
    const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values, ValueType* m, ValueType* z,
    ValueType* __restrict__ out, int* invalid)
{
    constexpr int subwarp_size =
        config::warp_size < 32 ? config::warp_size : 32;
    constexpr int num_subwarps = block_size / subwarp_size;
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subwarp_id = threadIdx.x / subwarp_size;
    const int lane = subwarp.thread_rank();
    const auto packed_size = spd_packed(size, 0);
    for (int64 t = threadIdx.x; t < packed_size; t += block_size) {
        m[t] = zero<ValueType>();
    }
    for (IndexType t = threadIdx.x; t < size; t += block_size) {
        z[t] = t == size - 1 ? one<ValueType>() : zero<ValueType>();
    }
    if (threadIdx.x == 0) {
        *invalid = 0;
    }
    __syncthreads();
    // M(a, b) = conj(A(S_a, S_b)) for b <= a, rows a by subwarps
    for (IndexType a = subwarp_id; a < size; a += num_subwarps) {
        const auto m_begin = m_row_ptrs[pattern[a]];
        group_match<subwarp_size>(
            m_col_idxs + m_begin, m_row_ptrs[pattern[a] + 1] - m_begin, pattern,
            a + 1, subwarp,
            [&](IndexType, IndexType m_idx, IndexType b, config::lane_mask_type,
                bool valid) {
                if (valid) {
                    m[spd_packed(a, b)] = conj(m_values[m_begin + m_idx]);
                }
            });
    }
    __syncthreads();
    bool valid = true;
    for (IndexType j = 0; j < size && valid; ++j) {
        const auto diag = real(m[spd_packed(j, j)]);
        valid = diag > zero<remove_complex<ValueType>>() && is_finite(diag);
        if (valid) {
            const auto c = sqrt(diag);
            for (IndexType i = j + 1 + threadIdx.x; i < size; i += block_size) {
                m[spd_packed(i, j)] /= c;
            }
            __syncthreads();
            // rows i by subwarps, columns l in (j, i] by lanes
            for (IndexType i = j + 1 + subwarp_id; i < size;
                 i += num_subwarps) {
                const auto l_ij = m[spd_packed(i, j)];
                for (IndexType l = j + 1 + lane; l <= i; l += subwarp_size) {
                    m[spd_packed(i, l)] -= l_ij * conj(m[spd_packed(l, j)]);
                }
            }
            if (threadIdx.x == 0) {
                m[spd_packed(j, j)] = c;
            }
            __syncthreads();
        }
    }
    // column-oriented backward solve of C^H z = e_k
    if (valid) {
        for (IndexType i = size - 1; i >= 0; --i) {
            const auto z_i = z[i] / m[spd_packed(i, i)];
            for (IndexType l = threadIdx.x; l < i; l += block_size) {
                z[l] -= conj(m[spd_packed(i, l)]) * z_i;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                z[i] = z_i;
            }
        }
        __syncthreads();
        for (IndexType t = threadIdx.x; t < size; t += block_size) {
            if (!is_finite(z[t])) {
                *invalid = 1;
            }
        }
    }
    __syncthreads();
    valid = valid && *invalid == 0;
    for (IndexType t = threadIdx.x; t < size; t += block_size) {
        out[t] =
            valid ? z[t]
                  : (pattern[t] == row ? one<ValueType>() : zero<ValueType>());
    }
    // the next row may reuse m and z
    __syncthreads();
}


// the column indices first, padded so that the values after them are aligned
template <typename IndexType>
__host__ __device__ size_type spd_t1_index_bytes(int64 size)
{
    return (static_cast<size_type>(size) * sizeof(IndexType) + 15) / 16 * 16;
}


template <typename ValueType, typename IndexType>
__host__ __device__ size_type spd_t1_shared_bytes(int64 size)
{
    return spd_t1_index_bytes<IndexType>(size) +
           static_cast<size_type>(spd_packed(size, 0) + size) *
               sizeof(ValueType);
}


// room left for the kernels' static shared memory
constexpr size_type spd_static_shared_bytes = 64;


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spd_t1_block_size) void spd_inverse_t1(
    const IndexType* __restrict__ rows,
    const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values)
{
    extern __shared__ __align__(16) char spd_shared[];
    __shared__ int invalid;
    const auto row = rows[blockIdx.x];
    const auto begin = i_row_ptrs[row];
    const auto size = i_row_ptrs[row + 1] - begin;
    auto pattern = reinterpret_cast<IndexType*>(spd_shared);
    auto m = reinterpret_cast<ValueType*>(spd_shared +
                                          spd_t1_index_bytes<IndexType>(size));
    auto z = m + spd_packed(size, 0);
    for (IndexType t = threadIdx.x; t < size; t += spd_t1_block_size) {
        pattern[t] = i_col_idxs[begin + t];
    }
    __syncthreads();
    spd_inverse_block_row<spd_t1_block_size>(row, size, pattern, m_row_ptrs,
                                             m_col_idxs, m_values, m, z,
                                             i_values + begin, &invalid);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spd_t2_block_size) void spd_inverse_t2(
    const IndexType* __restrict__ rows, IndexType num_t2_rows,
    const IndexType* __restrict__ m_row_ptrs,
    const IndexType* __restrict__ m_col_idxs,
    const ValueType* __restrict__ m_values,
    const IndexType* __restrict__ i_row_ptrs,
    const IndexType* __restrict__ i_col_idxs, ValueType* __restrict__ i_values,
    ValueType* __restrict__ workspace, int64 slot_size)
{
    __shared__ int invalid;
    auto m = workspace + blockIdx.x * slot_size;
    for (IndexType idx = blockIdx.x; idx < num_t2_rows; idx += gridDim.x) {
        const auto row = rows[idx];
        const auto begin = i_row_ptrs[row];
        const auto size = i_row_ptrs[row + 1] - begin;
        spd_inverse_block_row<spd_t2_block_size>(
            row, size, i_col_idxs + begin, m_row_ptrs, m_col_idxs, m_values, m,
            m + spd_packed(size, 0), i_values + begin, &invalid);
    }
}


/**
 * @internal
 *
 * 1 in t1_flags[row] / t2_flags[row] for rows longer than 32 / than k1,
 * the longest such row in max_t2_size; entry num_rows of both is 0.
 */
template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void classify_spd_rows(
    IndexType num_rows, IndexType k1, const IndexType* __restrict__ row_ptrs,
    IndexType* __restrict__ t1_flags, IndexType* __restrict__ t2_flags,
    IndexType* __restrict__ max_t2_size)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row > num_rows) {
        return;
    }
    const auto size = row < num_rows ? row_ptrs[row + 1] - row_ptrs[row] : 0;
    t1_flags[row] = size > row_size_limit && size <= k1;
    t2_flags[row] = size > k1;
    if (size > k1) {
        atomic_max(max_t2_size, size);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_spd_row_lists(
    IndexType num_rows, const IndexType* __restrict__ t1_ptrs,
    const IndexType* __restrict__ t2_ptrs, IndexType* __restrict__ t1_rows,
    IndexType* __restrict__ t2_rows)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    if (t1_ptrs[row + 1] > t1_ptrs[row]) {
        t1_rows[t1_ptrs[row]] = row;
    }
    if (t2_ptrs[row + 1] > t2_ptrs[row]) {
        t2_rows[t2_ptrs[row]] = row;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void generate_tri_inverse(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<ValueType, IndexType> inverse, IndexType* excess_rhs_ptrs,
    IndexType* excess_nz_ptrs, bool lower)
{
    const auto num_rows = input.size[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(num_rows, block / subwarp_size);
    if (grid > 0) {
        if (lower) {
            kernel::generate_l_inverse<subwarp_size, subwarps_per_block>
                <<<grid, block, 0, exec->get_stream()>>>(
                    static_cast<IndexType>(num_rows), input.row_ptrs,
                    input.col_idxs, as_device_type(input.values),
                    inverse.row_ptrs, inverse.col_idxs,
                    as_device_type(inverse.values), excess_rhs_ptrs,
                    excess_nz_ptrs);
        } else {
            kernel::generate_u_inverse<subwarp_size, subwarps_per_block>
                <<<grid, block, 0, exec->get_stream()>>>(
                    static_cast<IndexType>(num_rows), input.row_ptrs,
                    input.col_idxs, as_device_type(input.values),
                    inverse.row_ptrs, inverse.col_idxs,
                    as_device_type(inverse.values), excess_rhs_ptrs,
                    excess_nz_ptrs);
        }
    }
    components::prefix_sum_nonnegative(exec, excess_rhs_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, excess_nz_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_general_inverse(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<ValueType, IndexType> inverse, IndexType* excess_rhs_ptrs,
    IndexType* excess_nz_ptrs, bool spd)
{
    const auto num_rows = input.size[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(num_rows, block / subwarp_size);
    if (grid > 0) {
        kernel::generate_general_inverse<subwarp_size, subwarps_per_block>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), input.row_ptrs,
                input.col_idxs, as_device_type(input.values), inverse.row_ptrs,
                inverse.col_idxs, as_device_type(inverse.values),
                excess_rhs_ptrs, excess_nz_ptrs, spd);
    }
    components::prefix_sum_nonnegative(exec, excess_rhs_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, excess_nz_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<const ValueType, const IndexType> inverse,
    const IndexType* excess_rhs_ptrs, const IndexType* excess_nz_ptrs,
    matrix::view::csr<ValueType, IndexType> excess_system,
    matrix::view::dense<ValueType> excess_rhs, size_type e_start,
    size_type e_end)
{
    const auto num_rows = input.size[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::generate_excess_system<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), input.row_ptrs,
                input.col_idxs, as_device_type(input.values), inverse.row_ptrs,
                inverse.col_idxs, excess_rhs_ptrs, excess_nz_ptrs,
                excess_system.row_ptrs, excess_system.col_idxs,
                as_device_type(excess_system.values),
                as_device_type(excess_rhs.values), e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scale_excess_solution(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* excess_block_ptrs,
                           matrix::view::dense<ValueType> excess_solution,
                           size_type e_start, size_type e_end)
{
    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::scale_excess_solution<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                excess_block_ptrs, as_device_type(excess_solution.values),
                e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(
    std::shared_ptr<const DefaultExecutor> exec,
    const IndexType* excess_rhs_ptrs,
    matrix::view::dense<const ValueType> excess_solution,
    matrix::view::csr<ValueType, IndexType> inverse, size_type e_start,
    size_type e_end)
{
    const auto num_rows = inverse.size[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::copy_excess_solution<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), inverse.row_ptrs,
                excess_rhs_ptrs, as_device_type(excess_solution.values),
                as_device_type(inverse.values), e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


template <typename ValueType, typename IndexType>
void generate_spd_inverse(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<ValueType, IndexType> inverse, size_type workspace_bytes)
{
    using device_value = device_type<ValueType>;
    const auto num_rows = static_cast<IndexType>(inverse.size[0]);
    if (num_rows == 0) {
        return;
    }
    const auto m_vals = as_device_type(input.values);
    const auto i_vals = as_device_type(inverse.values);
    // T0: rows with at most 32 entries, one subwarp each
    kernel::spd_inverse_t0<subwarp_size, subwarps_per_block>
        <<<ceildiv(num_rows, subwarps_per_block), default_block_size, 0,
           exec->get_stream()>>>(num_rows, input.row_ptrs, input.col_idxs,
                                 m_vals, inverse.row_ptrs, inverse.col_idxs,
                                 i_vals);
    // T1 up to the longest row whose system fits the shared memory
    const auto shared_limit =
        static_cast<size_type>(max_shared_memory_per_block_optin(exec));
    IndexType k1 = row_size_limit;
    for (IndexType k = kernel::spd_max_t1_size; k > row_size_limit; --k) {
        if (kernel::spd_t1_shared_bytes<device_value, IndexType>(k) +
                kernel::spd_static_shared_bytes <=
            shared_limit) {
            k1 = k;
            break;
        }
    }
    array<IndexType> t1_ptrs{exec, static_cast<size_type>(num_rows) + 1};
    array<IndexType> t2_ptrs{exec, static_cast<size_type>(num_rows) + 1};
    array<IndexType> max_t2_size{exec, {IndexType{}}};
    kernel::classify_spd_rows<<<ceildiv(num_rows + 1, default_block_size),
                                default_block_size, 0, exec->get_stream()>>>(
        num_rows, k1, inverse.row_ptrs, t1_ptrs.get_data(), t2_ptrs.get_data(),
        max_t2_size.get_data());
    components::prefix_sum_nonnegative(exec, t1_ptrs.get_data(),
                                       t1_ptrs.get_size());
    components::prefix_sum_nonnegative(exec, t2_ptrs.get_data(),
                                       t2_ptrs.get_size());
    const auto num_t1 = get_element(t1_ptrs, num_rows);
    const auto num_t2 = get_element(t2_ptrs, num_rows);
    if (num_t1 + num_t2 == 0) {
        return;
    }
    array<IndexType> t1_rows{exec, static_cast<size_type>(num_t1)};
    array<IndexType> t2_rows{exec, static_cast<size_type>(num_t2)};
    kernel::fill_spd_row_lists<<<ceildiv(num_rows, default_block_size),
                                 default_block_size, 0, exec->get_stream()>>>(
        num_rows, t1_ptrs.get_const_data(), t2_ptrs.get_const_data(),
        t1_rows.get_data(), t2_rows.get_data());
    if (num_t1 > 0) {
        const auto shared_bytes =
            kernel::spd_t1_shared_bytes<device_value, IndexType>(k1);
        allow_dynamic_shared_memory(
            kernel::spd_inverse_t1<device_value, IndexType>,
            static_cast<int>(shared_bytes));
        kernel::spd_inverse_t1<<<num_t1, kernel::spd_t1_block_size,
                                 shared_bytes, exec->get_stream()>>>(
            t1_rows.get_const_data(), input.row_ptrs, input.col_idxs, m_vals,
            inverse.row_ptrs, inverse.col_idxs, i_vals);
    }
    if (num_t2 > 0) {
        // one global workspace slot per block, each block loops over rows
        const int64 max_size = get_element(max_t2_size, 0);
        const auto slot_size = kernel::spd_packed(max_size, 0) + max_size;
        const auto budget =
            workspace_bytes != 0
                ? workspace_bytes
                : std::min<size_type>(size_type{1} << 30,
                                      free_device_memory(exec) / 4);
        const auto num_slots = std::max<int64>(
            1, std::min<int64>(num_t2, static_cast<int64>(budget) /
                                           (slot_size * sizeof(ValueType))));
        array<ValueType> workspace{
            exec, static_cast<size_type>(num_slots * slot_size)};
        kernel::spd_inverse_t2<<<num_slots, kernel::spd_t2_block_size, 0,
                                 exec->get_stream()>>>(
            t2_rows.get_const_data(), num_t2, input.row_ptrs, input.col_idxs,
            m_vals, inverse.row_ptrs, inverse.col_idxs, i_vals,
            as_device_type(workspace.get_data()), slot_size);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_SPD_INVERSE_KERNEL);


}  // namespace isai
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
