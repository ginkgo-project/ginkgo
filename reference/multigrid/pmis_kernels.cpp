// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <tuple>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PMIS solver namespace.
 *
 */
namespace pmis {


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal,
                        remove_complex<ValueType>* row_maxabs)
{
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // accumulates, so that a second call on the off-diagonal block
        // extends the maximum
        auto max_abs = row_maxabs[row];
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (has_diagonal && col_idxs[idx] == row) {
                continue;
            }
            max_abs = std::max(max_abs, abs(vals[idx]));
        }
        row_maxabs[row] = max_abs;
    }
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
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // count the number of strongest neighbor
        IndexType count = 0;
        auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            sparsity_rows[row] = zero<IndexType>();
            continue;
        }
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (has_diagonal && col_idxs[idx] == row) {
                continue;
            }

            if (abs(vals[idx]) >= strength_threshold * max_abs) {
                count++;
            }
        }
        sparsity_rows[row] = count;
    }
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
    const auto vals = csr->get_const_values();
    for (IndexType row = 0; row < csr->get_size()[0]; row++) {
        auto s_idx = strong_dep->get_const_row_ptrs()[row];
        auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            continue;
        }
        for (auto idx = csr->get_const_row_ptrs()[row];
             idx < csr->get_const_row_ptrs()[row + 1]; idx++) {
            if (has_diagonal && csr->get_const_col_idxs()[idx] == row) {
                continue;
            }
            if (abs(vals[idx]) >= strength_threshold * max_abs) {
                strong_dep->get_col_idxs()[s_idx] =
                    csr->get_const_col_idxs()[idx];
                s_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void initialize_weight_and_status(std::shared_ptr<const DefaultExecutor> exec,
                                  size_type num, const LocalIndexType* counts,
                                  const GlobalIndexType* global_idx,
                                  ValueType* weight, int* status)
{
    for (size_type row = 0; row < num; row++) {
        const auto count = counts[row];
        status[row] =
            (count == zero<LocalIndexType>() ? kernels::pmis::fine
                                             : kernels::pmis::unassigned);
        const auto draw = kernels::pmis::random_weight_from_index(
            static_cast<uint64>(global_idx[row]));
        // the draw is below 1, but a narrow weight type could round it up to
        // 1 and shift the node by a whole in-degree
        weight[row] =
            static_cast<ValueType>(draw * 0.99f + static_cast<float>(count));
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename IndexType>
void add_at_indices(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                    const IndexType* idxs, const IndexType* values,
                    IndexType* out)
{
    for (size_type i = 0; i < num; i++) {
        out[idxs[i]] += values ? values[i] : IndexType{1};
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PMIS_ADD_AT_INDICES);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void classify_select(
    std::shared_ptr<const DefaultExecutor> exec,
    const remove_complex<ValueType>* weight, const GlobalIndexType* global_idx,
    LocalIndexType col_offset,
    const matrix::SparsityCsr<ValueType, LocalIndexType>* strong_dep,
    const int* status, int* new_status)
{
    const auto nrows = static_cast<LocalIndexType>(strong_dep->get_size()[0]);
    const auto row_ptrs = strong_dep->get_const_row_ptrs();
    const auto col_idxs = strong_dep->get_const_col_idxs();

    for (LocalIndexType row = 0; row < nrows; row++) {
        // already decided, or already downgraded by the other block
        if (status[row] != kernels::pmis::unassigned ||
            new_status[row] != kernels::pmis::coarse) {
            continue;
        }
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto col = col_offset + col_idxs[idx];
            if (status[col] == kernels::pmis::unassigned &&
                std::tie(weight[col], global_idx[col]) >
                    std::tie(weight[row], global_idx[row])) {
                new_status[row] = kernels::pmis::unassigned;
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_CLASSIFY_SELECT_KERNEL);


template <typename ValueType, typename IndexType>
void classify_mark_fine(
    std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    int* new_status)
{
    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto row_ptrs = strong_dep->get_const_row_ptrs();
    const auto col_idxs = strong_dep->get_const_col_idxs();

    for (IndexType row = 0; row < nrows; row++) {
        if (new_status[row] != kernels::pmis::unassigned) {
            continue;
        }
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (new_status[col_offset + col_idxs[idx]] ==
                kernels::pmis::coarse) {
                new_status[row] = kernels::pmis::fine;
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_CLASSIFY_MARK_FINE_KERNEL);


void classify_seed(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                   const int* status, int* new_status)
{
    for (size_type row = 0; row < num; row++) {
        new_status[row] = (status[row] == kernels::pmis::unassigned)
                              ? kernels::pmis::coarse
                              : status[row];
    }
}


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    size_type ans = 0;
    for (size_type i = 0; i < num; i++) {
        if (status[i] == kernels::pmis::unassigned) {
            ans++;
        }
    }
    *num_unassigned = ans;
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_count)
{
    const auto row_ptrs = strong_dep->get_const_row_ptrs();
    const auto col_idxs = strong_dep->get_const_col_idxs();
    for (size_type row = 0; row < strong_dep->get_size()[0]; row++) {
        // the caller seeds the coarse row's identity entry
        if (status[row] == kernels::pmis::coarse) {
            continue;
        }
        IndexType num = 0;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (status[col_offset + col_idxs[idx]] == kernels::pmis::coarse) {
                num++;
            }
        }
        prolong_row_count[row] += num;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT);


template <typename LocalIndexType, typename GlobalIndexType>
void coarse_global_index(std::shared_ptr<const DefaultExecutor> exec,
                         size_type num, GlobalIndexType offset,
                         const int* status, const LocalIndexType* coarse_map,
                         GlobalIndexType* coarse_global)
{
    for (size_type i = 0; i < num; ++i) {
        coarse_global[i] =
            status[i] == kernels::pmis::coarse
                ? offset + static_cast<GlobalIndexType>(coarse_map[i])
                : invalid_index<GlobalIndexType>();
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PMIS_COARSE_GLOBAL_INDEX);


template <typename ValueType, typename IndexType>
void direct_interpolation_accumulate(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,
    IndexType col_offset, const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold, const int* status,
    kernels::pmis::interpolation_workspace<ValueType, IndexType> ws)
{
    const auto values = csr->get_const_values();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto row_ptrs = csr->get_const_row_ptrs();
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (status[row] == kernels::pmis::coarse) {
            continue;
        }
        // no strong dependence, so no interpolation entry, matching
        // compute_strong_dep{,_row} and direct_interpolation_row_count
        const auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            continue;
        }
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto val = values[idx];
            const auto col = col_idxs[idx];
            if (has_diagonal && col == static_cast<IndexType>(row)) {
                ws.diag[row] = val;
                continue;
            }
            const bool is_coarse =
                status[col_offset + col] == kernels::pmis::coarse;
            const bool is_strong = abs(val) >= strength_threshold * max_abs;
            if (real(val) >= 0) {
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
    }
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
    const auto values = csr->get_const_values();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto row_ptrs = csr->get_const_row_ptrs();
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (status[row] == kernels::pmis::coarse) {
            continue;
        }
        const auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            continue;
        }
        if (!ws.enable_pos[row] && !ws.enable_neg[row]) {
            continue;
        }
        const auto alpha = safe_divide(ws.pos[row], ws.pos_divisor[row]);
        const auto beta = safe_divide(ws.neg[row], ws.neg_divisor[row]);
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto val = values[idx];
            const auto col = col_idxs[idx];
            if ((has_diagonal && col == static_cast<IndexType>(row)) ||
                abs(val) < strength_threshold * max_abs ||
                status[col_offset + col] != kernels::pmis::coarse) {
                continue;
            }
            if (real(val) >= 0 && ws.enable_pos[row]) {
                prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                prolong_values[ws.cursor[row]] = -alpha * val / ws.diag[row];
                ws.cursor[row]++;
            }
            if (real(val) < 0 && ws.enable_neg[row]) {
                prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                prolong_values[ws.cursor[row]] = -beta * val / ws.diag[row];
                ws.cursor[row]++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_EMIT);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill_coarse_rows(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,
    const int* status, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    for (size_type row = 0; row < num_rows; row++) {
        if (status[row] == kernels::pmis::coarse) {
            const auto idx = prolong_row_ptrs[row];
            prolong_col_idxs[idx] = static_cast<IndexType>(row);
            prolong_values[idx] = one<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL_COARSE_ROWS);


}  // namespace pmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
