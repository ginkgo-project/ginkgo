// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
                    rhs_one_idx += popcnt(subwarp.ballot(
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
    auto general_solve = [spd](IndexType num_elems,
                               ValueType* __restrict__ local_row,
                               group::thread_block_tile<subwarp_size>& subwarp,
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


}  // namespace kernel


template <typename ValueType, typename IndexType>
void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* input,
                          matrix::Csr<ValueType, IndexType>* inverse,
                          IndexType* excess_rhs_ptrs, IndexType* excess_nz_ptrs,
                          bool lower)
{
    const auto num_rows = input->get_size()[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(num_rows, block / subwarp_size);
    if (grid > 0) {
        if (lower) {
            kernel::generate_l_inverse<subwarp_size, subwarps_per_block>
                <<<grid, block, 0, exec->get_stream()>>>(
                    static_cast<IndexType>(num_rows),
                    input->get_const_row_ptrs(), input->get_const_col_idxs(),
                    as_device_type(input->get_const_values()),
                    inverse->get_row_ptrs(), inverse->get_col_idxs(),
                    as_device_type(inverse->get_values()), excess_rhs_ptrs,
                    excess_nz_ptrs);
        } else {
            kernel::generate_u_inverse<subwarp_size, subwarps_per_block>
                <<<grid, block, 0, exec->get_stream()>>>(
                    static_cast<IndexType>(num_rows),
                    input->get_const_row_ptrs(), input->get_const_col_idxs(),
                    as_device_type(input->get_const_values()),
                    inverse->get_row_ptrs(), inverse->get_col_idxs(),
                    as_device_type(inverse->get_values()), excess_rhs_ptrs,
                    excess_nz_ptrs);
        }
    }
    components::prefix_sum_nonnegative(exec, excess_rhs_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, excess_nz_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_general_inverse(std::shared_ptr<const DefaultExecutor> exec,
                              const matrix::Csr<ValueType, IndexType>* input,
                              matrix::Csr<ValueType, IndexType>* inverse,
                              IndexType* excess_rhs_ptrs,
                              IndexType* excess_nz_ptrs, bool spd)
{
    const auto num_rows = input->get_size()[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(num_rows, block / subwarp_size);
    if (grid > 0) {
        kernel::generate_general_inverse<subwarp_size, subwarps_per_block>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), input->get_const_row_ptrs(),
                input->get_const_col_idxs(),
                as_device_type(input->get_const_values()),
                inverse->get_row_ptrs(), inverse->get_col_idxs(),
                as_device_type(inverse->get_values()), excess_rhs_ptrs,
                excess_nz_ptrs, spd);
    }
    components::prefix_sum_nonnegative(exec, excess_rhs_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, excess_nz_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* input,
                            const matrix::Csr<ValueType, IndexType>* inverse,
                            const IndexType* excess_rhs_ptrs,
                            const IndexType* excess_nz_ptrs,
                            matrix::Csr<ValueType, IndexType>* excess_system,
                            matrix::Dense<ValueType>* excess_rhs,
                            size_type e_start, size_type e_end)
{
    const auto num_rows = input->get_size()[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::generate_excess_system<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), input->get_const_row_ptrs(),
                input->get_const_col_idxs(),
                as_device_type(input->get_const_values()),
                inverse->get_const_row_ptrs(), inverse->get_const_col_idxs(),
                excess_rhs_ptrs, excess_nz_ptrs, excess_system->get_row_ptrs(),
                excess_system->get_col_idxs(),
                as_device_type(excess_system->get_values()),
                as_device_type(excess_rhs->get_values()), e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scale_excess_solution(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* excess_block_ptrs,
                           matrix::Dense<ValueType>* excess_solution,
                           size_type e_start, size_type e_end)
{
    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::scale_excess_solution<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                excess_block_ptrs,
                as_device_type(excess_solution->get_values()), e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(std::shared_ptr<const DefaultExecutor> exec,
                             const IndexType* excess_rhs_ptrs,
                             const matrix::Dense<ValueType>* excess_solution,
                             matrix::Csr<ValueType, IndexType>* inverse,
                             size_type e_start, size_type e_end)
{
    const auto num_rows = inverse->get_size()[0];

    const auto block = default_block_size;
    const auto grid = ceildiv(e_end - e_start, block / subwarp_size);
    if (grid > 0) {
        kernel::copy_excess_solution<subwarp_size>
            <<<grid, block, 0, exec->get_stream()>>>(
                static_cast<IndexType>(num_rows), inverse->get_const_row_ptrs(),
                excess_rhs_ptrs,
                as_device_type(excess_solution->get_const_values()),
                as_device_type(inverse->get_values()), e_start, e_end);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);
