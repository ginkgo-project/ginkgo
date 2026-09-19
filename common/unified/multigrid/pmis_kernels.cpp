// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Pmis namespace.
 *
 */
namespace pmis {


// the number of threads working on the same row
constexpr int width = 32;


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal,
                        remove_complex<ValueType>* row_maxabs)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto has_diagonal, auto prev_maxabs,
                      auto row_ptrs, auto col_idxs, auto values) {
            // seeded from the previous value so a second call extends the
            // maximum. Reading the output is safe: run_kernel_row_reduction
            // writes it only in its final pass
            auto maxabs = prev_maxabs[row];
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (has_diagonal && row == col_idxs[idx]) {
                    continue;
                }
                maxabs = gko::max(maxabs, abs(values[idx]));
            }
            return maxabs;
        },
        GKO_KERNEL_REDUCE_MAX(remove_complex<ValueType>), row_maxabs, 1,
        dim<2>{csr->get_size()[0], width}, has_diagonal, row_maxabs,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* csr,
                            bool has_diagonal,
                            const remove_complex<ValueType>* row_maxabs,
                            remove_complex<ValueType> strength_threshold,
                            IndexType* sparsity_rows)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto has_diagonal, auto row_maxabs,
                      auto strength_threshold, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto max_abs = row_maxabs[row];
            auto count = zero<IndexType>();
            if (max_abs == zero(max_abs)) {
                return count;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (has_diagonal && row == col_idxs[idx]) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), sparsity_rows, 1,
        dim<2>{csr->get_size()[0], width}, has_diagonal, row_maxabs,
        strength_threshold, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal,
                        const remove_complex<ValueType>* row_maxabs,
                        remove_complex<ValueType> strength_threshold,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    // we handle this by one thread per row. It might get improved if we use a
    // warp with popcount and prefix for a row.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto row_maxabs,
                      auto strength_threshold, auto row_ptrs, auto col_idxs,
                      auto values, auto dep_row_ptrs, auto dep_col_idxs) {
            auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) {
                return;
            }
            auto d_idx = dep_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (has_diagonal && row == col) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    dep_col_idxs[d_idx] = col;
                    d_idx++;
                }
            }
        },
        csr->get_size()[0], has_diagonal, row_maxabs, strength_threshold,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), strong_dep->get_const_row_ptrs(),
        strong_dep->get_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void initialize_weight_and_status(std::shared_ptr<const DefaultExecutor> exec,
                                  size_type num, const LocalIndexType* counts,
                                  const GlobalIndexType* global_idx,
                                  ValueType* weight, int* status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto counts, auto global_idx, auto weight,
                      auto status) {
            using type = device_type<ValueType>;
            const auto count = counts[row];
            status[row] = (count == zero(count) ? kernels::pmis::fine
                                                : kernels::pmis::unassigned);
            const auto draw = kernels::pmis::random_weight_from_index(
                static_cast<uint64>(global_idx[row]));
            // the draw is below 1, but a narrow weight type could round it
            // up to 1 and shift the node by a whole in-degree
            weight[row] =
                static_cast<type>(draw * 0.99f + static_cast<float>(count));
        },
        num, counts, global_idx, weight, status);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void classify_select(
    std::shared_ptr<const DefaultExecutor> exec,
    const remove_complex<ValueType>* weight, const GlobalIndexType* global_idx,
    LocalIndexType col_offset,
    const matrix::SparsityCsr<ValueType, LocalIndexType>* strong_dep,
    const int* status, int* new_status)
{
    static_assert(kernels::pmis::unassigned < kernels::pmis::coarse,
                  "we use min reduction to mark local maximum as coarse");
    // seeded from new_status so a second call can only downgrade; see
    // compute_row_maxabs for why reading the reduction output is safe
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto col_offset, auto status,
                      auto new_status, auto weight, auto global_idx,
                      auto row_ptrs, auto col_idxs) {
            if (status[row] != kernels::pmis::unassigned ||
                new_status[row] != kernels::pmis::coarse) {
                return new_status[row];
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                const auto col = col_offset + col_idxs[idx];
                if (status[col] == kernels::pmis::unassigned &&
                    device_std::tie(weight[col], global_idx[col]) >
                        device_std::tie(weight[row], global_idx[row])) {
                    return kernels::pmis::unassigned;
                }
            }
            return kernels::pmis::coarse;
        },
        [] GKO_KERNEL(auto a, auto b) { return a < b ? a : b; } /* minimum */,
        [] GKO_KERNEL(auto a) { return a; }, kernels::pmis::coarse, new_status,
        1, dim<2>{strong_dep->get_size()[0], width}, col_offset, status,
        new_status, weight, global_idx, strong_dep->get_const_row_ptrs(),
        strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_CLASSIFY_SELECT_KERNEL);


template <typename ValueType, typename IndexType>
void classify_mark_fine(
    std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    int* new_status)
{
    // TODO: using warp vote function if implement in native way.
    static_assert(kernels::pmis::fine > kernels::pmis::unassigned,
                  "we use max reduction to mark new fine by any strong coarse");
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto col_offset, auto new_status,
                      auto row_ptrs, auto col_idxs) {
            if (new_status[row] != kernels::pmis::unassigned) {
                return new_status[row];
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                // we will only update new_status from -1 to 0 or keep -1, so
                // grabbing this value is fine no matter if it is updated or
                // not.
                if (new_status[col_offset + col_idxs[idx]] ==
                    kernels::pmis::coarse) {
                    return kernels::pmis::fine;
                }
            }
            return kernels::pmis::unassigned;
        },
        [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; } /* maximum */,
        [] GKO_KERNEL(auto a) { return a; }, kernels::pmis::unassigned,
        new_status, 1, dim<2>{strong_dep->get_size()[0], width}, col_offset,
        new_status, strong_dep->get_const_row_ptrs(),
        strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_CLASSIFY_MARK_FINE_KERNEL);


void classify_seed(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                   const int* status, int* new_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto status, auto new_status) {
            new_status[row] = (status[row] == kernels::pmis::unassigned)
                                  ? kernels::pmis::coarse
                                  : status[row];
        },
        num, status, new_status);
}


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    array<size_type> d_result(exec, 1);
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto status) {
            return static_cast<size_type>(status[i] ==
                                          kernels::pmis::unassigned);
        },
        GKO_KERNEL_REDUCE_SUM(size_type), d_result.get_data(), num, status);
    *num_unassigned = get_element(d_result, 0);
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_count)
{
    // seeded from prolong_row_count so a second call adds to the first
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto col_offset, auto prev_count,
                      auto status, auto row_ptrs, auto col_idxs) {
            // the caller seeds the coarse row's identity entry
            auto count = tid == 0 ? prev_count[row] : zero<IndexType>();
            if (status[row] == kernels::pmis::coarse) {
                return count;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (status[col_offset + col_idxs[idx]] ==
                    kernels::pmis::coarse) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), prolong_row_count, 1,
        dim<2>{strong_dep->get_size()[0], width}, col_offset, prolong_row_count,
        status, strong_dep->get_const_row_ptrs(),
        strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT);


template <typename LocalIndexType, typename GlobalIndexType>
void coarse_global_index(std::shared_ptr<const DefaultExecutor> exec,
                         size_type num, GlobalIndexType offset,
                         const int* status, const LocalIndexType* coarse_map,
                         GlobalIndexType* coarse_global)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto offset, auto status, auto coarse_map,
                      auto coarse_global) {
            using global_index_type = decltype(offset);
            coarse_global[i] =
                status[i] == kernels::pmis::coarse
                    ? offset + static_cast<global_index_type>(coarse_map[i])
                    : invalid_index<global_index_type>();
        },
        num, offset, status, coarse_map, coarse_global);
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_COARSE_GLOBAL_INDEX);


// run_kernel maps its arguments to device types, but does not map the
// members of a struct
template <typename ValueType, typename IndexType>
kernels::pmis::interpolation_workspace<device_type<ValueType>, IndexType>
as_device_workspace(
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws)
{
    return {as_device_type(ws.pos),  as_device_type(ws.pos_divisor),
            as_device_type(ws.neg),  as_device_type(ws.neg_divisor),
            as_device_type(ws.diag), ws.enable_pos,
            ws.enable_neg,           ws.cursor};
}


template <typename ValueType, typename IndexType>
void direct_interpolation_accumulate(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,
    IndexType col_offset, const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold, const int* status,
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto col_offset,
                      auto row_maxabs, auto strength_threshold, auto status,
                      auto row_ptrs, auto col_idxs, auto values, auto ws) {
            if (status[row] == kernels::pmis::coarse) {
                return;
            }
            const auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) {
                return;
            }
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto val = values[idx];
                const auto col = col_idxs[idx];
                if (has_diagonal && col == row) {
                    ws.diag[row] = val;
                    continue;
                }
                const bool is_coarse =
                    status[col_offset + col] == kernels::pmis::coarse;
                const bool is_strong = abs(val) >= strength_threshold * max_abs;
                if (real(val) >= zero(real(val))) {
                    ws.pos[row] += val;
                    if (is_coarse && is_strong) {
                        ws.pos_divisor[row] += val;
                        ws.enable_pos[row] = 1;
                    }
                } else {
                    ws.neg[row] += val;
                    if (is_coarse && is_strong) {
                        ws.neg_divisor[row] += val;
                        ws.enable_neg[row] = 1;
                    }
                }
            }
        },
        csr->get_size()[0], has_diagonal, col_offset, row_maxabs,
        strength_threshold, status, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(),
        as_device_workspace(ws));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ACCUMULATE);


template <typename ValueType, typename IndexType>
void direct_interpolation_emit(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,
    IndexType col_offset, const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold, const int* status,
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto col_offset,
                      auto row_maxabs, auto strength_threshold, auto status,
                      auto row_ptrs, auto col_idxs, auto values, auto ws,
                      auto prolong_col_idxs, auto prolong_values) {
            if (status[row] == kernels::pmis::coarse) {
                return;
            }
            const auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) {
                return;
            }
            if (!ws.enable_pos[row] && !ws.enable_neg[row]) {
                return;
            }
            const auto alpha = safe_divide(ws.pos[row], ws.pos_divisor[row]);
            const auto beta = safe_divide(ws.neg[row], ws.neg_divisor[row]);
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto val = values[idx];
                const auto col = col_idxs[idx];
                if ((has_diagonal && col == row) ||
                    abs(val) < strength_threshold * max_abs ||
                    status[col_offset + col] != kernels::pmis::coarse) {
                    continue;
                }
                if (real(val) >= zero(real(val)) && ws.enable_pos[row]) {
                    prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                    prolong_values[ws.cursor[row]] =
                        -alpha * val / ws.diag[row];
                    ws.cursor[row]++;
                }
                if (real(val) < zero(real(val)) && ws.enable_neg[row]) {
                    prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                    prolong_values[ws.cursor[row]] = -beta * val / ws.diag[row];
                    ws.cursor[row]++;
                }
            }
        },
        csr->get_size()[0], has_diagonal, col_offset, row_maxabs,
        strength_threshold, status, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(),
        as_device_workspace(ws), prolong_col_idxs, prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_EMIT);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill_coarse_rows(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,
    const int* status, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto status, auto prolong_row_ptrs,
                      auto prolong_col_idxs, auto prolong_values) {
            if (status[row] == kernels::pmis::coarse) {
                const auto idx = prolong_row_ptrs[row];
                prolong_col_idxs[idx] = row;
                prolong_values[idx] = one<device_type<ValueType>>();
            }
        },
        num_rows, status, prolong_row_ptrs, prolong_col_idxs, prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL_COARSE_ROWS);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
