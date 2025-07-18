// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/factorization_kernels.hpp"

#include <thrust/logical.h>

#include <ginkgo/core/base/array.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/searching.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/factorization/factorization_helpers.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


constexpr int default_block_size{512};


namespace kernel {


namespace detail {


// Default implementation for the unsorted case
template <bool IsSorted>
struct find_helper {
    template <typename Group, typename IndexType>
    static __forceinline__ __device__ bool find(Group subwarp_grp,
                                                const IndexType* first,
                                                const IndexType* last,
                                                IndexType value)
    {
        auto subwarp_idx = subwarp_grp.thread_rank();
        bool found{false};
        for (auto curr_start = first; curr_start < last;
             curr_start += subwarp_grp.size()) {
            const auto curr = curr_start + subwarp_idx;
            found = (curr < last && *curr == value);
            found = subwarp_grp.any(found);
            if (found) {
                break;
            }
        }
        return found;
    }
};


// Improved version in case the CSR matrix is sorted
template <>
struct find_helper<true> {
    template <typename Group, typename IndexType>
    static __forceinline__ __device__ bool find(Group subwarp_grp,
                                                const IndexType* first,
                                                const IndexType* last,
                                                IndexType value)
    {
        const auto length = static_cast<IndexType>(last - first);
        const auto pos =
            group_wide_search(IndexType{}, length, subwarp_grp,
                              [&](IndexType i) { return first[i] >= value; });
        return pos < length && first[pos] == value;
    }
};


}  // namespace detail


// SubwarpSize needs to be a power of 2
// Each subwarp works on one row
template <bool IsSorted, int SubwarpSize, typename IndexType>
__global__
__launch_bounds__(default_block_size) void find_missing_diagonal_elements(
    IndexType num_rows, IndexType num_cols,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs,
    IndexType* __restrict__ elements_to_add_per_row,
    bool* __restrict__ changes_required)
{
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<SubwarpSize, IndexType>();
    const auto begin_row =
        thread::get_subwarp_id_flat<SubwarpSize, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<SubwarpSize>(thread_block);
    const auto subwarp_idx = subwarp_grp.thread_rank();

    bool local_change{false};
    for (auto row = begin_row; row < num_rows; row += total_subwarp_count) {
        if (row >= num_cols) {
            if (subwarp_idx == 0) {
                elements_to_add_per_row[row] = 0;
            }
            continue;
        }
        const auto* start_cols = col_idxs + row_ptrs[row];
        const auto* end_cols = col_idxs + row_ptrs[row + 1];
        if (detail::find_helper<IsSorted>::find(subwarp_grp, start_cols,
                                                end_cols, row)) {
            if (subwarp_idx == 0) {
                elements_to_add_per_row[row] = 0;
            }
        } else {
            if (subwarp_idx == 0) {
                elements_to_add_per_row[row] = 1;
            }
            local_change = true;
        }
    }
    // Could also be reduced (not sure if that leads to a performance benefit)
    if (local_change && subwarp_idx == 0) {
        *changes_required = true;
    }
}


// SubwarpSize needs to be a power of 2
// Each subwarp works on one row
template <int SubwarpSize, typename ValueType, typename IndexType>
__global__
__launch_bounds__(default_block_size) void add_missing_diagonal_elements(
    IndexType num_rows, const ValueType* __restrict__ old_values,
    const IndexType* __restrict__ old_col_idxs,
    const IndexType* __restrict__ old_row_ptrs,
    ValueType* __restrict__ new_values, IndexType* __restrict__ new_col_idxs,
    const IndexType* __restrict__ row_ptrs_addition)
{
    // Precaution in case not enough threads were created
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<SubwarpSize, IndexType>();
    const auto begin_row =
        thread::get_subwarp_id_flat<SubwarpSize, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<SubwarpSize>(thread_block);
    const auto subwarp_idx = subwarp_grp.thread_rank();

    for (auto row = begin_row; row < num_rows; row += total_subwarp_count) {
        const IndexType old_row_start{old_row_ptrs[row]};
        const IndexType old_row_end{old_row_ptrs[row + 1]};
        const IndexType new_row_start{old_row_start + row_ptrs_addition[row]};
        const IndexType new_row_end{old_row_end + row_ptrs_addition[row + 1]};

        // if no element needs to be added, do a simple copy of the whole row
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            for (IndexType i = subwarp_idx; i < new_row_end - new_row_start;
                 i += SubwarpSize) {
                const IndexType new_idx = new_row_start + i;
                const IndexType old_idx = old_row_start + i;
                new_values[new_idx] = old_values[old_idx];
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
        } else {
            IndexType new_idx = new_row_start + subwarp_idx;
            bool diagonal_added{false};
            for (IndexType old_idx_start = old_row_start;
                 old_idx_start < old_row_end;
                 old_idx_start += SubwarpSize, new_idx += SubwarpSize) {
                const auto old_idx = old_idx_start + subwarp_idx;
                bool thread_is_active = old_idx < old_row_end;
                const auto col_idx =
                    thread_is_active ? old_col_idxs[old_idx] : IndexType{};
                // automatically false if thread is not active
                bool diagonal_add_required = !diagonal_added && row < col_idx;
                auto ballot = group::ballot(subwarp_grp, diagonal_add_required);

                if (ballot) {
                    auto first_subwarp_idx = ffs(ballot) - 1;
                    if (first_subwarp_idx == subwarp_idx) {
                        new_values[new_idx] = zero<ValueType>();
                        new_col_idxs[new_idx] = row;
                    }
                    if (thread_is_active) {
                        // if diagonal was inserted in a thread below this one,
                        // add it to the new_idx.
                        bool is_thread_after_diagonal =
                            (first_subwarp_idx <= subwarp_idx);
                        new_idx += is_thread_after_diagonal;
                        new_values[new_idx] = old_values[old_idx];
                        new_col_idxs[new_idx] = col_idx;
                        // if diagonal is inserted in a thread after this one,
                        // it needs to be considered after writing the values
                        new_idx += !is_thread_after_diagonal;
                    }
                    diagonal_added = true;
                } else if (thread_is_active) {
                    new_values[new_idx] = old_values[old_idx];
                    new_col_idxs[new_idx] = col_idx;
                }
            }
            if (!diagonal_added && subwarp_idx == 0) {
                new_idx = new_row_end - 1;
                new_values[new_idx] = zero<ValueType>();
                new_col_idxs[new_idx] = row;
            }
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void update_row_ptrs(
    IndexType num_rows, IndexType* __restrict__ row_ptrs,
    IndexType* __restrict__ row_ptr_addition)
{
    const auto total_thread_count = thread::get_thread_num_flat<IndexType>();
    const auto begin_row = thread::get_thread_id_flat<IndexType>();

    for (auto row = begin_row; row < num_rows; row += total_thread_count) {
        row_ptrs[row] += row_ptr_addition[row];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_l_u_row(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, IndexType* __restrict__ l_nnz_row,
    IndexType* __restrict__ u_nnz_row)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row < num_rows) {
        IndexType l_row_nnz{};
        IndexType u_row_nnz{};
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            auto col = col_idxs[idx];
            // skip diagonal
            l_row_nnz += (col < row);
            u_row_nnz += (row < col);
        }
        // add the diagonal entry
        l_nnz_row[row] = l_row_nnz + 1;
        u_nnz_row[row] = u_row_nnz + 1;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_l_row(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, IndexType* __restrict__ l_nnz_row)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row < num_rows) {
        IndexType l_row_nnz{};
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            auto col = col_idxs[idx];
            // skip the diagonal entry
            l_row_nnz += col < row;
        }
        // add the diagonal entry
        l_nnz_row[row] = l_row_nnz + 1;
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void symbolic_validate(
    const IndexType* __restrict__ mtx_row_ptrs,
    const IndexType* __restrict__ mtx_cols,
    const IndexType* __restrict__ factor_row_ptrs,
    const IndexType* __restrict__ factor_cols, size_type size,
    const IndexType* __restrict__ storage_offsets,
    const int64* __restrict__ row_descs, const int32* __restrict__ storage,
    bool* __restrict__ found, bool* __restrict__ missing)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= size) {
        return;
    }
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        factor_row_ptrs, factor_cols, storage_offsets,
        storage,         row_descs,   static_cast<size_type>(row)};
    const auto mtx_begin = mtx_row_ptrs[row];
    const auto mtx_end = mtx_row_ptrs[row + 1];
    const auto factor_begin = factor_row_ptrs[row];
    const auto factor_end = factor_row_ptrs[row + 1];
    bool local_missing = false;
    const auto mark_found = [&](IndexType col) {
        const auto local_idx = lookup[col];
        const auto idx = local_idx + factor_begin;
        if (local_idx == invalid_index<IndexType>()) {
            local_missing = true;
            return;
        }
        found[idx] = true;
    };
    // check the original matrix is part of the factors
    for (auto nz = mtx_begin + lane; nz < mtx_end; nz += config::warp_size) {
        mark_found(mtx_cols[nz]);
    }
    // check the diagonal is part of the factors
    if (lane == 0) {
        mark_found(row);
    }
    // check it is a valid factorization
    for (auto nz = factor_begin; nz < factor_end; nz++) {
        const auto dep = factor_cols[nz];
        if (dep >= row) {
            continue;
        }
        // for every lower triangular entry
        const auto dep_begin = factor_row_ptrs[dep];
        const auto dep_end = factor_row_ptrs[dep + 1];
        for (auto dep_nz = dep_begin + lane; dep_nz < dep_end;
             dep_nz += config::warp_size) {
            const auto col = factor_cols[dep_nz];
            // check every upper triangular entry thereof is part of the
            // factorization
            if (col > dep) {
                mark_found(col);
            }
        }
    }
    local_missing = warp.any(local_missing);
    if (lane == 0) {
        missing[row] = local_missing;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec,
                           matrix::Csr<ValueType, IndexType>* mtx,
                           bool is_sorted)
{
    // TODO: Runtime can be optimized by choosing a appropriate size for the
    //       subwarp dependent on the matrix properties
    constexpr int subwarp_size = config::warp_size;
    auto mtx_size = mtx->get_size();
    auto num_rows = static_cast<IndexType>(mtx_size[0]);
    auto num_cols = static_cast<IndexType>(mtx_size[1]);
    size_type row_ptrs_size = num_rows + 1;
    if (num_rows == 0) {
        return;
    }

    array<IndexType> row_ptrs_addition(exec, row_ptrs_size);
    array<bool> needs_change_host{exec->get_master(), 1};
    needs_change_host.get_data()[0] = false;
    array<bool> needs_change_device{exec, 1};
    needs_change_device = needs_change_host;

    auto old_values = as_device_type(mtx->get_const_values());
    auto old_col_idxs = mtx->get_const_col_idxs();
    auto old_row_ptrs = mtx->get_row_ptrs();
    auto row_ptrs_add = row_ptrs_addition.get_data();

    const auto block_dim = default_block_size;
    const auto grid_dim =
        static_cast<uint32>(ceildiv(num_rows, block_dim / subwarp_size));
    if (is_sorted) {
        kernel::find_missing_diagonal_elements<true, subwarp_size>
            <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                num_rows, num_cols, old_col_idxs, old_row_ptrs, row_ptrs_add,
                needs_change_device.get_data());
    } else {
        kernel::find_missing_diagonal_elements<false, subwarp_size>
            <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                num_rows, num_cols, old_col_idxs, old_row_ptrs, row_ptrs_add,
                needs_change_device.get_data());
    }
    needs_change_host = needs_change_device;
    if (!needs_change_host.get_const_data()[0]) {
        return;
    }

    components::prefix_sum_nonnegative(exec, row_ptrs_add, row_ptrs_size);
    exec->synchronize();

    auto total_additions = get_element(row_ptrs_addition, row_ptrs_size - 1);
    size_type new_num_elems = static_cast<size_type>(total_additions) +
                              mtx->get_num_stored_elements();


    array<ValueType> new_value_array{exec, new_num_elems};
    array<IndexType> new_col_idx_array{exec, new_num_elems};
    auto new_values = as_device_type(new_value_array.get_data());
    auto new_col_idxs = new_col_idx_array.get_data();

    kernel::add_missing_diagonal_elements<subwarp_size>
        <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
            num_rows, old_values, old_col_idxs, old_row_ptrs, new_values,
            new_col_idxs, row_ptrs_add);

    const auto grid_dim_row_ptrs_update =
        static_cast<uint32>(ceildiv(num_rows, block_dim));
    kernel::update_row_ptrs<<<grid_dim_row_ptrs_update, block_dim, 0,
                              exec->get_stream()>>>(num_rows + 1, old_row_ptrs,
                                                    row_ptrs_add);

    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_value_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idx_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs, IndexType* u_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};

    const auto block_size = default_block_size;
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size));
    const auto grid_dim = number_blocks;

    if (grid_dim > 0) {
        kernel::count_nnz_per_l_u_row<<<grid_dim, block_size, 0,
                                        exec->get_stream()>>>(
            num_rows, system_matrix->get_const_row_ptrs(),
            system_matrix->get_const_col_idxs(),
            as_device_type(system_matrix->get_const_values()), l_row_ptrs,
            u_row_ptrs);
    }

    components::prefix_sum_nonnegative(exec, l_row_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* system_matrix,
                    matrix::Csr<ValueType, IndexType>* csr_l,
                    matrix::Csr<ValueType, IndexType>* csr_u)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    using namespace gko::factorization;

    if (grid_dim > 0) {
        auto l_closure = triangular_mtx_closure(
            [] __device__(auto val) { return one(val); }, identity{});
        auto u_closure = triangular_mtx_closure(identity{}, identity{});
        helpers::
            initialize_l_u<<<grid_dim, block_size, 0, exec->get_stream()>>>(
                num_rows, system_matrix->get_const_row_ptrs(),
                system_matrix->get_const_col_idxs(),
                as_device_type(system_matrix->get_const_values()),
                csr_l->get_const_row_ptrs(), csr_l->get_col_idxs(),
                as_device_type(csr_l->get_values()),
                csr_u->get_const_row_ptrs(), csr_u->get_col_idxs(),
                as_device_type(csr_u->get_values()), l_closure, u_closure);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    IndexType* l_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};

    const auto block_size = default_block_size;
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size));
    const auto grid_dim = number_blocks;

    if (grid_dim > 0) {
        kernel::count_nnz_per_l_row<<<grid_dim, block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, system_matrix->get_const_row_ptrs(),
            system_matrix->get_const_col_idxs(),
            as_device_type(system_matrix->get_const_values()), l_row_ptrs);
    }

    components::prefix_sum_nonnegative(exec, l_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* system_matrix,
                  matrix::Csr<ValueType, IndexType>* csr_l, bool diag_sqrt)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const auto block_size = helpers::default_block_size;
    const auto grid_dim = static_cast<uint32>(
        ceildiv(num_rows, static_cast<size_type>(block_size)));

    if (grid_dim > 0) {
        using namespace gko::factorization;

        helpers::initialize_l<<<grid_dim, block_size, 0, exec->get_stream()>>>(
            num_rows, system_matrix->get_const_row_ptrs(),
            system_matrix->get_const_col_idxs(),
            as_device_type(system_matrix->get_const_values()),
            csr_l->get_const_row_ptrs(), csr_l->get_col_idxs(),
            as_device_type(csr_l->get_values()),
            triangular_mtx_closure(
                [diag_sqrt] __device__(auto val) {
                    if (diag_sqrt) {
                        val = sqrt(val);
                        if (!is_finite(val)) {
                            val = one(val);
                        }
                    }
                    return val;
                },
                identity{}));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL);


template <typename ValueType, typename IndexType>
void symbolic_validate(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const matrix::Csr<ValueType, IndexType>* factors,
    const matrix::csr::lookup_data<IndexType>& factors_lookup, bool& valid)
{
    const auto size = system_matrix->get_size()[0];
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto factor_row_ptrs = factors->get_const_row_ptrs();
    const auto factor_col_idxs = factors->get_const_col_idxs();
    // this stores for each factor nonzero whether it occurred as part of the
    // factorization.
    array<bool> found(exec, factors->get_num_stored_elements());
    components::fill_array(exec, found.get_data(), found.get_size(), false);
    // this stores for each row whether there were any elements missing
    array<bool> missing(exec, size);
    components::fill_array(exec, missing.get_data(), missing.get_size(), false);
    if (size > 0) {
        const auto num_blocks =
            ceildiv(size, default_block_size / config::warp_size);
        kernel::symbolic_validate<<<num_blocks, default_block_size>>>(
            row_ptrs, col_idxs, factor_row_ptrs, factor_col_idxs, size,
            factors_lookup.storage_offsets.get_const_data(),
            factors_lookup.row_descs.get_const_data(),
            factors_lookup.storage.get_const_data(), found.get_data(),
            missing.get_data());
    }
    valid = thrust::all_of(thrust_policy(exec), found.get_const_data(),
                           found.get_const_data() + found.get_size(),
                           thrust::identity<bool>{}) &&
            !thrust::any_of(thrust_policy(exec), missing.get_const_data(),
                            missing.get_const_data() + missing.get_size(),
                            thrust::identity<bool>{});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_SYMBOLIC_VALIDATE_KERNEL);


}  // namespace factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
