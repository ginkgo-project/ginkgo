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


// one thread per row: three per-row scalars at once do not map onto one
// reduction, and three reductions would read the matrix three times
template <typename ValueType, typename IndexType>
void compute_row_scales(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal, remove_complex<ValueType>* neg_scale,
                        remove_complex<ValueType>* row_sum,
                        remove_complex<ValueType>* diag_abs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto row_ptrs, auto col_idxs,
                      auto values, auto neg_scale, auto row_sum,
                      auto diag_abs) {
            // accumulating: a second call on the off-diagonal block extends
            // both
            auto scale = neg_scale[row];
            auto sum = row_sum[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto val = real(values[idx]);
                sum += val;
                if (has_diagonal && col_idxs[idx] == row) {
                    // only the block carrying the diagonal writes diag_abs
                    diag_abs[row] = abs(val);
                    continue;
                }
                scale = gko::max(scale, -val);
            }
            // floored at zero: with the strict > in is_strong_dep, a row with
            // no negative off-diagonal has no strong dependency
            neg_scale[row] = gko::max(scale, zero(scale));
            row_sum[row] = sum;
        },
        csr->get_size()[0], has_diagonal, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(), neg_scale, row_sum,
        diag_abs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_ROW_SCALES_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* csr,
                            bool has_diagonal,
                            const remove_complex<ValueType>* neg_scale,
                            const remove_complex<ValueType>* row_sum,
                            const remove_complex<ValueType>* diag_abs,
                            remove_complex<ValueType> strength_threshold,
                            remove_complex<ValueType> max_row_sum,
                            IndexType* sparsity_rows)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto has_diagonal, auto neg_scale,
                      auto row_sum, auto diag_abs, auto strength_threshold,
                      auto max_row_sum, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto count = zero<IndexType>();
            // a fully summed row has every dependency weakened; all `width`
            // threads see the same condition
            if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                   max_row_sum)) {
                return count;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (has_diagonal && row == col_idxs[idx]) {
                    continue;
                }
                if (kernels::pmis::is_strong_dep(values[idx], neg_scale[row],
                                                 strength_threshold)) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), sparsity_rows, 1,
        dim<2>{csr->get_size()[0], width}, has_diagonal, neg_scale, row_sum,
        diag_abs, strength_threshold, max_row_sum, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal,
                        const remove_complex<ValueType>* neg_scale,
                        const remove_complex<ValueType>* row_sum,
                        const remove_complex<ValueType>* diag_abs,
                        remove_complex<ValueType> strength_threshold,
                        remove_complex<ValueType> max_row_sum,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    // we handle this by one thread per row. It might get improved if we use a
    // warp with popcount and prefix for a row.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto neg_scale, auto row_sum,
                      auto diag_abs, auto strength_threshold, auto max_row_sum,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto dep_row_ptrs, auto dep_col_idxs) {
            // compute_strong_dep_row has already given a fully summed row an
            // empty range
            if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                   max_row_sum)) {
                return;
            }
            auto d_idx = dep_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (has_diagonal && row == col) {
                    continue;
                }
                if (kernels::pmis::is_strong_dep(values[idx], neg_scale[row],
                                                 strength_threshold)) {
                    dep_col_idxs[d_idx] = col;
                    d_idx++;
                }
            }
        },
        csr->get_size()[0], has_diagonal, neg_scale, row_sum, diag_abs,
        strength_threshold, max_row_sum, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(),
        strong_dep->get_const_row_ptrs(), strong_dep->get_col_idxs());
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
            // scaled by an exact power of two, see the reference kernel
            weight[row] =
                static_cast<type>(draw * 0.5f + static_cast<float>(count));
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


void count_unassigned(std::shared_ptr<const DefaultExecutor> exec,
                      size_type num, const int* status,
                      size_type* num_unassigned)
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
    IndexType col_offset, const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto col_offset,
                      auto neg_scale, auto row_sum, auto diag_abs,
                      auto strength_threshold, auto max_row_sum, auto status,
                      auto row_ptrs, auto col_idxs, auto values, auto ws) {
            if (status[row] == kernels::pmis::coarse) {
                return;
            }
            // no strong dependence, so no interpolation entry; the emit pass
            // still reads ws.diag and the pos/neg sums
            if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                   max_row_sum)) {
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
                const bool is_strong = kernels::pmis::is_strong_dep(
                    val, neg_scale[row], strength_threshold);
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
        csr->get_size()[0], has_diagonal, col_offset, neg_scale, row_sum,
        diag_abs, strength_threshold, max_row_sum, status,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), as_device_workspace(ws));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ACCUMULATE);


template <typename ValueType, typename IndexType>
void direct_interpolation_emit(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,
    IndexType col_offset, const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto has_diagonal, auto col_offset,
                      auto neg_scale, auto row_sum, auto diag_abs,
                      auto strength_threshold, auto max_row_sum, auto status,
                      auto row_ptrs, auto col_idxs, auto values, auto ws,
                      auto prolong_col_idxs, auto prolong_values) {
            if (status[row] == kernels::pmis::coarse) {
                return;
            }
            if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                   max_row_sum)) {
                return;
            }
            if (!ws.enable_pos[row] && !ws.enable_neg[row]) {
                return;
            }
            const auto alpha = safe_divide(ws.neg[row], ws.neg_divisor[row]);
            const auto beta = safe_divide(ws.pos[row], ws.pos_divisor[row]);
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto val = values[idx];
                const auto col = col_idxs[idx];
                if ((has_diagonal && col == row) ||
                    !kernels::pmis::is_strong_dep(val, neg_scale[row],
                                                  strength_threshold) ||
                    status[col_offset + col] != kernels::pmis::coarse) {
                    continue;
                }
                if (real(val) >= zero(real(val)) && ws.enable_pos[row]) {
                    prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                    prolong_values[ws.cursor[row]] = -beta * val / ws.diag[row];
                    ws.cursor[row]++;
                }
                if (real(val) < zero(real(val)) && ws.enable_neg[row]) {
                    prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                    prolong_values[ws.cursor[row]] =
                        -alpha * val / ws.diag[row];
                    ws.cursor[row]++;
                }
            }
        },
        csr->get_size()[0], has_diagonal, col_offset, neg_scale, row_sum,
        diag_abs, strength_threshold, max_row_sum, status,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), as_device_workspace(ws), prolong_col_idxs,
        prolong_values);
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


// random access a_{row, col} by binary search; GKO_ATTRIBUTES/GKO_INLINE so it
// is valid device code
template <typename IndexType, typename ValueType>
GKO_ATTRIBUTES GKO_INLINE ValueType lookup_value(const IndexType* row_ptrs,
                                                 const IndexType* col_idxs,
                                                 const ValueType* values,
                                                 IndexType row, IndexType col)
{
    auto begin = row_ptrs[row];
    auto end = row_ptrs[row + 1];
    while (begin < end) {
        const auto mid = begin + (end - begin) / 2;
        if (col_idxs[mid] < col) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return (begin < row_ptrs[row + 1] && col_idxs[begin] == col)
               ? values[begin]
               : zero<ValueType>();
}


// Classical (Ruge-Stueben) interpolation, one thread per row;
// reference/multigrid/pmis_kernels.cpp is the specification.
template <typename ValueType, typename IndexType>
void classical_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    IndexType* prolong_row_count)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto neg_scale, auto row_sum, auto diag_abs,
                      auto strength_threshold, auto max_row_sum, auto status,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto prolong_row_count) {
            if (status[row] == kernels::pmis::coarse) {
                prolong_row_count[row] = one<IndexType>();
                return;
            }
            auto count = zero<IndexType>();
            if (!kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                    max_row_sum)) {
                for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                    const auto col = col_idxs[idx];
                    if (col == row) {
                        continue;
                    }
                    if (kernels::pmis::is_strong_dep(
                            values[idx], neg_scale[row], strength_threshold) &&
                        status[col] == kernels::pmis::coarse) {
                        count++;
                    }
                }
            }
            prolong_row_count[row] = count;
        },
        csr->get_size()[0], neg_scale, row_sum, diag_abs, strength_threshold,
        max_row_sum, status, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(), prolong_row_count);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CLASSICAL_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void classical_interpolation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    const IndexType* coarse_map, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    // built host-side: remove_complex<ValueType>{...} is not
    // device-constructible for gko::half
    const remove_complex<ValueType> stability_factor{0.1};
    const remove_complex<ValueType> absolute_gate_factor{1e-3};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto neg_scale, auto row_sum,
                      auto diag_abs, auto strength_threshold, auto max_row_sum,
                      auto stability_factor, auto absolute_gate_factor,
                      auto status, auto coarse_map, auto row_ptrs,
                      auto col_idxs, auto values, auto prolong_row_ptrs,
                      auto prolong_col_idxs, auto prolong_values) {
            using value_type = device_type<ValueType>;
            using index_type = device_type<IndexType>;
            const auto out_begin = prolong_row_ptrs[row];
            const auto out_end = prolong_row_ptrs[row + 1];
            if (status[row] == kernels::pmis::coarse) {
                prolong_col_idxs[out_begin] = coarse_map[row];
                prolong_values[out_begin] = one<value_type>();
                return;
            }
            // the magnitude scale the gate at the end of this loop compares
            // against
            const auto max_abs = row_maxabs[row];
            if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                   max_row_sum)) {
                return;
            }

            const auto row_idx = static_cast<index_type>(row);
            const auto diag =
                lookup_value(row_ptrs, col_idxs, values, row_idx, row_idx);
            // a coupling is distributed only when its sign opposes the
            // diagonal's. Real part only.
            auto distributable = [&](value_type val) {
                return real(val) * real(diag) < zero(real(diag));
            };

            // pass 1: classify the row's off-diagonal entries. The column is
            // encoded as `col` for a distributable slot and -(col + 1)
            // otherwise, so pass 2 can tell them apart in place.
            auto weak_sum = zero<value_type>();
            auto absorbed = zero<value_type>();
            auto out_idx = out_begin;
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                const auto val = values[idx];
                if (col == row) {
                    continue;
                }
                if (kernels::pmis::is_strong_dep(val, neg_scale[row],
                                                 strength_threshold)) {
                    if (status[col] == kernels::pmis::coarse) {
                        if (distributable(val)) {
                            prolong_col_idxs[out_idx] = col;
                            prolong_values[out_idx] = val;
                        } else {
                            prolong_col_idxs[out_idx] =
                                -(col + one<index_type>());
                            prolong_values[out_idx] = zero<value_type>();
                            absorbed += val;
                        }
                        out_idx++;
                    }
                } else {
                    weak_sum += val;
                }
            }
            auto denominator = diag + weak_sum + absorbed;

            // pass 2: distribute each strong F-neighbour k's coupling over the
            // slots where both a_{row,m} and a_{k,m} oppose diag, the set s_k
            // sums
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                const auto val = values[idx];
                if (col == row ||
                    !kernels::pmis::is_strong_dep(val, neg_scale[row],
                                                  strength_threshold) ||
                    status[col] == kernels::pmis::coarse) {
                    continue;
                }
                const auto k = col;
                const auto a_ik = val;
                if (!distributable(a_ik)) {
                    denominator += a_ik;
                    continue;
                }
                auto s_k = zero<value_type>();
                for (auto o = out_begin; o < out_end; o++) {
                    if (prolong_col_idxs[o] < zero(prolong_col_idxs[o])) {
                        continue;  // non-distributable slot, excluded
                    }
                    const auto a_km = lookup_value(row_ptrs, col_idxs, values,
                                                   k, prolong_col_idxs[o]);
                    if (!distributable(a_km)) {
                        continue;  // wrong-signed a_km, excluded from s_k
                    }
                    s_k += a_km;
                }
                // nowhere to distribute a_ik, so absorb it into the diagonal
                // and treat k as weak
                if (s_k == zero(s_k)) {
                    denominator += a_ik;
                    continue;
                }
                for (auto o = out_begin; o < out_end; o++) {
                    if (prolong_col_idxs[o] < zero(prolong_col_idxs[o])) {
                        continue;
                    }
                    const auto a_km = lookup_value(row_ptrs, col_idxs, values,
                                                   k, prolong_col_idxs[o]);
                    if (!distributable(a_km)) {
                        continue;
                    }
                    prolong_values[o] += a_ik * a_km / s_k;
                }
            }

            // the denominator may have absorbed same-sign couplings, so require
            // it to keep diag's sign and a tenth of its magnitude
            const auto stable =
                ((real(denominator) >= zero(real(denominator))) ==
                 (real(diag) >= zero(real(diag)))) &&
                abs(denominator) >= stability_factor * abs(diag);
            auto final_denominator = denominator;
            if (!stable) {
                // fall back to the direct formula, which divides by diag alone
                auto pos = zero<value_type>();
                auto neg = zero<value_type>();
                auto pos_divisor = zero<value_type>();
                auto neg_divisor = zero<value_type>();
                for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                    const auto col = col_idxs[idx];
                    if (col == row) {
                        continue;
                    }
                    const auto val = values[idx];
                    const auto is_coarse_and_strong =
                        status[col] == kernels::pmis::coarse &&
                        kernels::pmis::is_strong_dep(val, neg_scale[row],
                                                     strength_threshold);
                    if (real(val) >= zero(real(val))) {
                        pos += val;
                        if (is_coarse_and_strong) {
                            pos_divisor += val;
                        }
                    } else {
                        neg += val;
                        if (is_coarse_and_strong) {
                            neg_divisor += val;
                        }
                    }
                }
                const auto alpha = safe_divide(pos, pos_divisor);
                const auto beta = safe_divide(neg, neg_divisor);
                for (auto o = out_begin; o < out_end; o++) {
                    const auto encoded = prolong_col_idxs[o];
                    const auto j = encoded >= zero(encoded)
                                       ? encoded
                                       : -encoded - one<index_type>();
                    const auto a_ij =
                        lookup_value(row_ptrs, col_idxs, values, row_idx, j);
                    prolong_values[o] = real(a_ij) >= zero(real(a_ij))
                                            ? alpha * a_ij
                                            : beta * a_ij;
                }
                final_denominator = diag;
            }

            // absolute gate: the relative check can pass while the divisor is
            // still tiny. The 1e-3 floor sits at or below epsilon at gko::half
            // and gko::bfloat16.
            const auto too_small =
                abs(final_denominator) < absolute_gate_factor * max_abs;
            for (auto o = out_begin; o < out_end; o++) {
                const auto encoded = prolong_col_idxs[o];
                const auto j = encoded >= zero(encoded)
                                   ? encoded
                                   : -encoded - one<index_type>();
                prolong_col_idxs[o] = coarse_map[j];
                prolong_values[o] =
                    too_small ? zero<value_type>()
                              : -prolong_values[o] / final_denominator;
            }
        },
        csr->get_size()[0], row_maxabs, neg_scale, row_sum, diag_abs,
        strength_threshold, max_row_sum, stability_factor, absolute_gate_factor,
        status, coarse_map, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(), prolong_row_ptrs,
        prolong_col_idxs, prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CLASSICAL_INTERPOLATION_FILL);


// Extended+i interpolation, one thread per row;
// reference/multigrid/pmis_kernels.cpp is the specification. The difference is
// scratch: the row's own output range doubles as the reference's sorted,
// deduplicated Chat_i.


// the smallest member of Chat_row strictly greater than `prev`, or -1; feeding
// each answer back walks the set in ascending order without a sort
template <typename IndexType, typename ValueType, typename RealType>
GKO_ATTRIBUTES GKO_INLINE IndexType ext_i_next_extended_c(
    const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* values, const RealType* neg_scale, const RealType* row_sum,
    const RealType* diag_abs, RealType strength_threshold, RealType max_row_sum,
    const int* status, IndexType row, IndexType prev)
{
    auto best = -one<IndexType>();
    if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                           max_row_sum)) {
        return best;
    }
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        const auto col = col_idxs[idx];
        if (col == row ||
            !kernels::pmis::is_strong_dep(values[idx], neg_scale[row],
                                          strength_threshold)) {
            continue;
        }
        if (status[col] == kernels::pmis::coarse) {
            if (col > prev && (best < zero<IndexType>() || col < best)) {
                best = col;
            }
            continue;
        }
        const auto k = col;
        if (kernels::pmis::row_is_fully_summed(row_sum[k], diag_abs[k],
                                               max_row_sum)) {
            continue;
        }
        for (auto idx2 = row_ptrs[k]; idx2 < row_ptrs[k + 1]; idx2++) {
            const auto col2 = col_idxs[idx2];
            if (col2 == k || status[col2] != kernels::pmis::coarse ||
                !kernels::pmis::is_strong_dep(values[idx2], neg_scale[k],
                                              strength_threshold)) {
                continue;
            }
            if (col2 > prev && (best < zero<IndexType>() || col2 < best)) {
                best = col2;
            }
        }
    }
    return best;
}


// position of `node` in the extended set parked in cols[begin, end), or -1
template <typename IndexType>
GKO_ATTRIBUTES GKO_INLINE IndexType ext_i_slot_of(const IndexType* cols,
                                                  IndexType begin,
                                                  IndexType end, IndexType node)
{
    auto lo = begin;
    auto hi = end;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (cols[mid] < node) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < end && cols[lo] == node) ? lo : -one<IndexType>();
}


template <typename ValueType, typename IndexType>
void ext_i_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    IndexType* prolong_row_count)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto neg_scale, auto row_sum, auto diag_abs,
                      auto strength_threshold, auto max_row_sum, auto status,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto prolong_row_count) {
            using index_type = device_type<IndexType>;
            if (status[row] == kernels::pmis::coarse) {
                prolong_row_count[row] = one<index_type>();
                return;
            }
            const auto row_idx = static_cast<index_type>(row);
            // |Chat_row|, counted by walking the set in ascending order
            auto count = zero<index_type>();
            auto prev = -one<index_type>();
            while (true) {
                const auto next = ext_i_next_extended_c(
                    row_ptrs, col_idxs, values, neg_scale, row_sum, diag_abs,
                    strength_threshold, max_row_sum, status, row_idx, prev);
                if (next < zero<index_type>()) {
                    break;
                }
                count++;
                prev = next;
            }
            prolong_row_count[row] = count;
        },
        csr->get_size()[0], neg_scale, row_sum, diag_abs, strength_threshold,
        max_row_sum, status, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(), prolong_row_count);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_EXT_I_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void ext_i_interpolation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
    const IndexType* coarse_map, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    // built host-side, like strength_threshold: not device-constructible for
    // gko::half
    const remove_complex<ValueType> absolute_gate_factor{1e-3};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto neg_scale, auto row_sum,
                      auto diag_abs, auto strength_threshold, auto max_row_sum,
                      auto absolute_gate_factor, auto status, auto coarse_map,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto prolong_row_ptrs, auto prolong_col_idxs,
                      auto prolong_values) {
            using value_type = device_type<ValueType>;
            using index_type = device_type<IndexType>;
            const auto out_begin = prolong_row_ptrs[row];
            const auto out_end = prolong_row_ptrs[row + 1];
            if (status[row] == kernels::pmis::coarse) {
                prolong_col_idxs[out_begin] = coarse_map[row];
                prolong_values[out_begin] = one<value_type>();
                return;
            }
            if (out_begin >= out_end) {
                return;
            }
            const auto i = static_cast<index_type>(row);

            // rebuild Chat_i, ascending, into the row's output range; the count
            // pass sized it with the same helper, so no bound is needed
            {
                auto prev = -one<index_type>();
                for (auto o = out_begin; o < out_end; o++) {
                    prev = ext_i_next_extended_c(row_ptrs, col_idxs, values,
                                                 neg_scale, row_sum, diag_abs,
                                                 strength_threshold,
                                                 max_row_sum, status, i, prev);
                    prolong_col_idxs[o] = prev;
                    prolong_values[o] = zero<value_type>();
                }
            }

            // d starts at a_ii, looked up explicitly since the diagonal need
            // not be first
            auto d = lookup_value(row_ptrs, col_idxs, values, i, i);

            for (auto idx = row_ptrs[i]; idx < row_ptrs[i + 1]; idx++) {
                const auto j = col_idxs[idx];
                const auto a_ij = values[idx];
                if (j == i) {
                    continue;
                }
                const auto slot =
                    ext_i_slot_of(prolong_col_idxs, out_begin, out_end, j);
                if (slot >= zero<index_type>()) {
                    prolong_values[slot] += a_ij;
                    continue;
                }
                const bool is_strong_f_neighbour =
                    status[j] != kernels::pmis::coarse &&
                    kernels::pmis::is_strong_dep(a_ij, neg_scale[i],
                                                 strength_threshold);
                if (!is_strong_f_neighbour) {
                    // weak, or a C-point Chat_i does not contain: lumped onto
                    // the diagonal
                    d += a_ij;
                    continue;
                }
                const auto k = j;
                const auto a_ik = a_ij;
                // the sign of row k's own diagonal, not row i's
                const auto a_kk =
                    lookup_value(row_ptrs, col_idxs, values, k, k);
                const auto sgn_k = real(a_kk) < zero(real(a_kk))
                                       ? -one(real(a_kk))
                                       : one(real(a_kk));
                // one predicate for both the sigma_k sum and the distribution.
                // Real part only.
                auto admissible = [&](index_type m, value_type a_km) {
                    return m != k &&
                           (m == i ||
                            ext_i_slot_of(prolong_col_idxs, out_begin, out_end,
                                          m) >= zero<index_type>()) &&
                           sgn_k * real(a_km) < zero(real(a_km));
                };
                auto sigma_k = zero<value_type>();
                for (auto idx2 = row_ptrs[k]; idx2 < row_ptrs[k + 1]; idx2++) {
                    if (admissible(col_idxs[idx2], values[idx2])) {
                        sigma_k += values[idx2];
                    }
                }
                // exact, with no tolerance: see above
                if (sigma_k == zero(sigma_k)) {
                    // nowhere to distribute a_ik, so treat k as weak for this
                    // row
                    d += a_ik;
                    continue;
                }
                const auto tau_k = a_ik / sigma_k;
                for (auto idx2 = row_ptrs[k]; idx2 < row_ptrs[k + 1]; idx2++) {
                    const auto m = col_idxs[idx2];
                    const auto a_km = values[idx2];
                    if (!admissible(m, a_km)) {
                        continue;
                    }
                    if (m == i) {
                        // the "+i" feedback: while scanning row k, test m == i
                        d += tau_k * a_km;
                    } else {
                        prolong_values[ext_i_slot_of(prolong_col_idxs,
                                                     out_begin, out_end, m)] +=
                            tau_k * a_km;
                    }
                }
            }

            // d mixes signs and can cancel to near zero, so the guard is
            // relative to row_maxabs
            const auto usable =
                abs(d) >= absolute_gate_factor * row_maxabs[row];
            for (auto o = out_begin; o < out_end; o++) {
                prolong_col_idxs[o] = coarse_map[prolong_col_idxs[o]];
                prolong_values[o] =
                    usable ? -prolong_values[o] / d : zero<value_type>();
            }
        },
        csr->get_size()[0], row_maxabs, neg_scale, row_sum, diag_abs,
        strength_threshold, max_row_sum, absolute_gate_factor, status,
        coarse_map, csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), prolong_row_ptrs, prolong_col_idxs,
        prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_EXT_I_INTERPOLATION_FILL);


// truncation of an assembled prolongation; see core/multigrid/pmis_kernels.hpp
// for the two-pass contract


// shared by both kernels so they cannot drift apart; a plain function, so it is
// found unqualified inside a GKO_KERNEL lambda
GKO_ATTRIBUTES GKO_INLINE size_type truncated_row_nnz(size_type row_nnz,
                                                      size_type p_max_elmts)
{
    return row_nnz < p_max_elmts ? row_nnz : p_max_elmts;
}


// IndexType-only: it never touches a value, only row_ptrs differences
template <typename IndexType>
void truncate_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,
    size_type p_max_elmts, const IndexType* in_row_ptrs,
    IndexType* out_row_ptrs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto p_max_elmts, auto in_row_ptrs,
                      auto out_row_ptrs) {
            using index_type = device_type<IndexType>;
            const auto row_nnz =
                static_cast<size_type>(in_row_ptrs[row + 1] - in_row_ptrs[row]);
            out_row_ptrs[row] = static_cast<index_type>(
                truncated_row_nnz(row_nnz, p_max_elmts));
        },
        num_rows, p_max_elmts, in_row_ptrs, out_row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_TRUNCATE_INTERPOLATION_ROW_COUNT);


// out-of-place: truncation shrinks rows
template <typename ValueType, typename IndexType>
void truncate_interpolation_fill(std::shared_ptr<const DefaultExecutor> exec,
                                 size_type num_rows, size_type p_max_elmts,
                                 const IndexType* in_row_ptrs,
                                 const IndexType* in_col_idxs,
                                 const ValueType* in_values,
                                 const IndexType* out_row_ptrs,
                                 IndexType* out_col_idxs, ValueType* out_values)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto p_max_elmts, auto in_row_ptrs,
                      auto in_col_idxs, auto in_values, auto out_row_ptrs,
                      auto out_col_idxs, auto out_values) {
            using value_type = device_type<ValueType>;
            const auto begin = in_row_ptrs[row];
            const auto end = in_row_ptrs[row + 1];
            const auto row_nnz = static_cast<size_type>(end - begin);
            const auto out_begin = out_row_ptrs[row];
            // hard bound: a NaN makes every magnitude comparison false, so
            // every position would rank 0 and overrun a slot sized for
            // p_max_elmts
            const auto out_end = out_row_ptrs[row + 1];
            auto out_idx = out_begin;
            if (truncated_row_nnz(row_nnz, p_max_elmts) == row_nnz) {
                // untouched: nothing is dropped, so the row sum is preserved
                for (auto idx = begin; idx < end; idx++, out_idx++) {
                    out_col_idxs[out_idx] = in_col_idxs[idx];
                    out_values[out_idx] = in_values[idx];
                }
                return;
            }
            auto original_row_sum = zero<value_type>();
            for (auto idx = begin; idx < end; idx++) {
                original_row_sum += in_values[idx];
            }
            // rank-and-keep in one ascending pass: keep a position if fewer
            // than p_max_elmts others outrank it. Writing in this scan gives
            // ascending column order for free.
            auto kept_sum = zero<value_type>();
            for (auto idx = begin; idx < end; idx++) {
                const auto abs_val = abs(in_values[idx]);
                auto rank = zero(p_max_elmts);
                for (auto idx2 = begin; idx2 < end; idx2++) {
                    if (idx2 == idx) {
                        continue;
                    }
                    const auto abs_val2 = abs(in_values[idx2]);
                    if (abs_val2 > abs_val ||
                        (abs_val2 == abs_val && idx2 < idx)) {
                        rank++;
                    }
                }
                if (rank < p_max_elmts && out_idx < out_end) {
                    out_col_idxs[out_idx] = in_col_idxs[idx];
                    out_values[out_idx] = in_values[idx];
                    kept_sum += in_values[idx];
                    out_idx++;
                }
            }
            // rescale the survivors to the original row sum, unless they sum to
            // exactly zero
            if (kept_sum != zero(kept_sum)) {
                const auto scale = original_row_sum / kept_sum;
                for (auto o = out_begin; o < out_idx; o++) {
                    out_values[o] *= scale;
                }
            }
        },
        num_rows, p_max_elmts, in_row_ptrs, in_col_idxs, in_values,
        out_row_ptrs, out_col_idxs, out_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_TRUNCATE_INTERPOLATION_FILL);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
