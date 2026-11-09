// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

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
void compute_row_scales(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        bool has_diagonal, remove_complex<ValueType>* neg_scale,
                        remove_complex<ValueType>* row_sum,
                        remove_complex<ValueType>* diag_abs)
{
    using real_type = remove_complex<ValueType>;
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // accumulating: a second call on the off-diagonal block extends both
        auto scale = neg_scale[row];
        auto sum = row_sum[row];
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto val = real(vals[idx]);
            sum += val;
            if (has_diagonal && col_idxs[idx] == row) {
                // only the block carrying the diagonal writes diag_abs
                diag_abs[row] = abs(val);
                continue;
            }
            scale = std::max(scale, -val);
        }
        // floored at zero: with the strict > in is_strong_dep, a row with no
        // negative off-diagonal has no strong dependency
        neg_scale[row] = std::max(scale, zero<real_type>());
        row_sum[row] = sum;
    }
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
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // a nearly fully summed row has every dependency made weak
        if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                               max_row_sum)) {
            sparsity_rows[row] = zero<IndexType>();
            continue;
        }
        IndexType count = 0;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (has_diagonal && col_idxs[idx] == row) {
                continue;
            }

            if (kernels::pmis::is_strong_dep(vals[idx], neg_scale[row],
                                             strength_threshold)) {
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
                        const remove_complex<ValueType>* neg_scale,
                        const remove_complex<ValueType>* row_sum,
                        const remove_complex<ValueType>* diag_abs,
                        remove_complex<ValueType> strength_threshold,
                        remove_complex<ValueType> max_row_sum,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    const auto vals = csr->get_const_values();
    for (IndexType row = 0; row < csr->get_size()[0]; row++) {
        auto s_idx = strong_dep->get_const_row_ptrs()[row];
        // compute_strong_dep_row has already given a fully summed row an empty
        // range
        if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                               max_row_sum)) {
            continue;
        }
        for (auto idx = csr->get_const_row_ptrs()[row];
             idx < csr->get_const_row_ptrs()[row + 1]; idx++) {
            if (has_diagonal && csr->get_const_col_idxs()[idx] == row) {
                continue;
            }
            if (kernels::pmis::is_strong_dep(vals[idx], neg_scale[row],
                                             strength_threshold)) {
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
        // Scaled below 1: a narrow weight type could otherwise round the
        // draw up to 1 and shift the node by a whole in-degree. An exact power
        // of two, so fusing the multiply-add gives the same value as not
        // fusing.
        weight[row] =
            static_cast<ValueType>(draw * 0.5f + static_cast<float>(count));
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


void count_unassigned(std::shared_ptr<const DefaultExecutor> exec,
                      size_type num, const int* status,
                      size_type* num_unassigned)
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
    IndexType col_offset, const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
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
        if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                               max_row_sum)) {
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
            const bool is_strong = kernels::pmis::is_strong_dep(
                val, neg_scale[row], strength_threshold);
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
    IndexType col_offset, const remove_complex<ValueType>* neg_scale,
    const remove_complex<ValueType>* row_sum,
    const remove_complex<ValueType>* diag_abs,
    const remove_complex<ValueType> strength_threshold,
    const remove_complex<ValueType> max_row_sum, const int* status,
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
        if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                               max_row_sum)) {
            continue;
        }
        if (!ws.enable_pos[row] && !ws.enable_neg[row]) {
            continue;
        }
        const auto alpha = safe_divide(ws.neg[row], ws.neg_divisor[row]);
        const auto beta = safe_divide(ws.pos[row], ws.pos_divisor[row]);
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto val = values[idx];
            const auto col = col_idxs[idx];
            if ((has_diagonal && col == static_cast<IndexType>(row)) ||
                !kernels::pmis::is_strong_dep(val, neg_scale[row],
                                              strength_threshold) ||
                status[col_offset + col] != kernels::pmis::coarse) {
                continue;
            }
            if (real(val) >= 0 && ws.enable_pos[row]) {
                prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                prolong_values[ws.cursor[row]] = -beta * val / ws.diag[row];
                ws.cursor[row]++;
            }
            if (real(val) < 0 && ws.enable_neg[row]) {
                prolong_col_idxs[ws.cursor[row]] = col_offset + col;
                prolong_values[ws.cursor[row]] = -alpha * val / ws.diag[row];
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


// Classical (Ruge-Stueben) interpolation; re-derives strong(row, col) inline
// rather than consuming a pre-built strength matrix.
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
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto values = csr->get_const_values();
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (status[row] == kernels::pmis::coarse) {
            prolong_row_count[row] = one<IndexType>();
            continue;
        }
        IndexType count = zero<IndexType>();
        if (!kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                                max_row_sum)) {
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (col == static_cast<IndexType>(row)) {
                    continue;
                }
                if (kernels::pmis::is_strong_dep(values[idx], neg_scale[row],
                                                 strength_threshold) &&
                    status[col] == kernels::pmis::coarse) {
                    count++;
                }
            }
        }
        prolong_row_count[row] = count;
    }
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
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto values = csr->get_const_values();

    // random access a_{row, col}, assuming columns are sorted per row
    auto lookup = [&](IndexType row, IndexType col) -> ValueType {
        const auto begin = col_idxs + row_ptrs[row];
        const auto end = col_idxs + row_ptrs[row + 1];
        const auto it = std::lower_bound(begin, end, col);
        return (it != end && *it == col) ? values[it - col_idxs]
                                         : zero<ValueType>();
    };

    using real_type = remove_complex<ValueType>;
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        auto out_idx = prolong_row_ptrs[row];
        if (status[row] == kernels::pmis::coarse) {
            prolong_col_idxs[out_idx] = coarse_map[row];
            prolong_values[out_idx] = one<ValueType>();
            continue;
        }
        // the magnitude scale the stability gate at the end of this loop
        // compares against
        const auto max_abs = row_maxabs[row];
        if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                               max_row_sum)) {
            continue;
        }

        const auto diag =
            lookup(static_cast<IndexType>(row), static_cast<IndexType>(row));
        // a coupling is distributed only when its sign opposes the diagonal's;
        // a same-sign one is folded into the denominator instead. Real part
        // only.
        auto distributable = [&](ValueType val) {
            return real(val) * real(diag) < zero<real_type>();
        };

        ValueType weak_sum = zero<ValueType>();
        std::vector<IndexType> strong_c_cols;
        std::vector<ValueType> strong_c_vals;
        std::vector<bool> strong_c_distributable;
        std::vector<IndexType> strong_f_cols;
        std::vector<ValueType> strong_f_vals;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            const auto col = col_idxs[idx];
            const auto val = values[idx];
            if (col == static_cast<IndexType>(row)) {
                continue;
            }
            if (kernels::pmis::is_strong_dep(val, neg_scale[row],
                                             strength_threshold)) {
                if (status[col] == kernels::pmis::coarse) {
                    strong_c_cols.push_back(col);
                    strong_c_vals.push_back(val);
                    strong_c_distributable.push_back(distributable(val));
                } else {
                    strong_f_cols.push_back(col);
                    strong_f_vals.push_back(val);
                }
            } else {
                weak_sum += val;
            }
        }

        auto denominator = diag + weak_sum;
        // running w_ij: a_ij if distributable, zero if not
        std::vector<ValueType> w(strong_c_cols.size(), zero<ValueType>());
        for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
            if (strong_c_distributable[jj]) {
                w[jj] = strong_c_vals[jj];
            } else {
                denominator += strong_c_vals[jj];
            }
        }

        std::vector<ValueType> a_km(strong_c_cols.size());
        std::vector<bool> a_km_distributable(strong_c_cols.size());
        for (size_type kk = 0; kk < strong_f_cols.size(); kk++) {
            const auto k = strong_f_cols[kk];
            const auto a_ik = strong_f_vals[kk];
            if (!distributable(a_ik)) {
                denominator += a_ik;
                continue;
            }
            // s_k sums row k's entries at the row's distributable strong
            // C-neighbours where a_{k,m} also opposes diag. One sign per
            // summand makes sum_m(a_km/s_k) == 1 over that set, which preserves
            // the row sum.
            std::fill(a_km_distributable.begin(), a_km_distributable.end(),
                      false);
            ValueType s_k = zero<ValueType>();
            for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
                if (!strong_c_distributable[jj]) {
                    continue;
                }
                const auto val = lookup(k, strong_c_cols[jj]);
                if (!distributable(val)) {
                    continue;
                }
                a_km[jj] = val;
                a_km_distributable[jj] = true;
                s_k += val;
            }
            // an F-neighbour with no distributable C-neighbour in common is
            // absorbed into the diagonal, which keeps the row-sum identity
            if (s_k == zero<ValueType>()) {
                denominator += a_ik;
                continue;
            }
            for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
                if (!a_km_distributable[jj]) {
                    continue;
                }
                w[jj] += a_ik * a_km[jj] / s_k;
            }
        }

        // the denominator may have absorbed same-sign couplings, so require it
        // to keep diag's sign and a tenth of its magnitude
        const auto stable = ((real(denominator) >= zero<real_type>()) ==
                             (real(diag) >= zero<real_type>())) &&
                            abs(denominator) >= real_type{0.1} * abs(diag);
        auto final_denominator = denominator;
        if (!stable) {
            // fall back to the direct formula, which divides by diag alone
            ValueType pos = zero<ValueType>();
            ValueType neg = zero<ValueType>();
            ValueType pos_divisor = zero<ValueType>();
            ValueType neg_divisor = zero<ValueType>();
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (col == static_cast<IndexType>(row)) {
                    continue;
                }
                const auto val = values[idx];
                const auto is_coarse_and_strong =
                    status[col] == kernels::pmis::coarse &&
                    kernels::pmis::is_strong_dep(val, neg_scale[row],
                                                 strength_threshold);
                if (real(val) >= zero<real_type>()) {
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
            for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
                w[jj] = real(strong_c_vals[jj]) >= zero<real_type>()
                            ? alpha * strong_c_vals[jj]
                            : beta * strong_c_vals[jj];
            }
            final_denominator = diag;
        }

        // absolute gate: the relative check above can pass while the divisor is
        // still tiny. The 1e-3 floor sits at or below epsilon at gko::half and
        // gko::bfloat16.
        if (abs(final_denominator) < real_type{1e-3} * max_abs) {
            // a near-singular row emits zero weights rather than dividing
            for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
                prolong_col_idxs[out_idx] = coarse_map[strong_c_cols[jj]];
                prolong_values[out_idx] = zero<ValueType>();
                out_idx++;
            }
        } else {
            for (size_type jj = 0; jj < strong_c_cols.size(); jj++) {
                prolong_col_idxs[out_idx] = coarse_map[strong_c_cols[jj]];
                prolong_values[out_idx] = -w[jj] / final_denominator;
                out_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CLASSICAL_INTERPOLATION_FILL);


// Extended+i interpolation. For an F-row i, with S the sign-aware strength
// graph (no diagonal):
//
//   C_i = S_i n C,  F_i = S_i n F,  Chat_i = C_i u union_{k in F_i} (S_k n C)
//
// Chat_i is P's row pattern, one hop further out than classical, so it does
// not rely on the common-C-point assumption and can hold a node j with
// a_ij == 0.
//
// Per k in F_i, with sgn_k the sign of a_kk:
//
//   A_k(i) = { m != k : (m in Chat_i or m == i) and sgn_k * a_km < 0 }
//   sigma_k = sum_{m in A_k(i)} a_km
//
// "m == i" is the "+i" term. Every term of sigma_k passes the same sign
// test, so it cannot near-cancel and the test against zero needs no
// tolerance. Real part only, so for a complex instantiation sigma_k can
// still be small but non-zero through imaginary cancellation.
namespace {


// builds Chat_i and F_i for row `row`; both passes call it, so they cannot
// disagree. extended_c comes back sorted and deduplicated.
template <typename ValueType, typename IndexType>
void ext_i_build_sets(const matrix::Csr<ValueType, IndexType>* csr,
                      const remove_complex<ValueType>* neg_scale,
                      const remove_complex<ValueType>* row_sum,
                      const remove_complex<ValueType>* diag_abs,
                      const remove_complex<ValueType> strength_threshold,
                      const remove_complex<ValueType> max_row_sum,
                      const int* status, IndexType row,
                      std::vector<IndexType>& extended_c,
                      std::vector<IndexType>& strong_f)
{
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto values = csr->get_const_values();
    extended_c.clear();
    strong_f.clear();
    // a fully summed row has an empty S_i, hence an empty C_i and union over
    // F_i
    if (kernels::pmis::row_is_fully_summed(row_sum[row], diag_abs[row],
                                           max_row_sum)) {
        return;
    }
    // S_k n C, appended to extended_c
    auto append_strong_c = [&](IndexType scan) {
        if (kernels::pmis::row_is_fully_summed(row_sum[scan], diag_abs[scan],
                                               max_row_sum)) {
            return;
        }
        for (auto idx = row_ptrs[scan]; idx < row_ptrs[scan + 1]; idx++) {
            const auto col = col_idxs[idx];
            if (col == scan) {
                continue;
            }
            if (status[col] == kernels::pmis::coarse &&
                kernels::pmis::is_strong_dep(values[idx], neg_scale[scan],
                                             strength_threshold)) {
                extended_c.push_back(col);
            }
        }
    };
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        const auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        if (!kernels::pmis::is_strong_dep(values[idx], neg_scale[row],
                                          strength_threshold)) {
            continue;
        }
        if (status[col] == kernels::pmis::coarse) {
            extended_c.push_back(col);
        } else {
            strong_f.push_back(col);
        }
    }
    for (const auto k : strong_f) {
        append_strong_c(k);
    }
    std::sort(extended_c.begin(), extended_c.end());
    extended_c.erase(std::unique(extended_c.begin(), extended_c.end()),
                     extended_c.end());
}


}  // namespace


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
    std::vector<IndexType> extended_c;
    std::vector<IndexType> strong_f;
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (status[row] == kernels::pmis::coarse) {
            prolong_row_count[row] = one<IndexType>();
            continue;
        }
        ext_i_build_sets(csr, neg_scale, row_sum, diag_abs, strength_threshold,
                         max_row_sum, status, static_cast<IndexType>(row),
                         extended_c, strong_f);
        prolong_row_count[row] = static_cast<IndexType>(extended_c.size());
    }
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
    using real_type = remove_complex<ValueType>;
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto values = csr->get_const_values();

    // random access a_{r, col}: the diagonal need not be a row's first entry.
    // Assumes sorted columns.
    auto lookup = [&](IndexType r, IndexType col) -> ValueType {
        const auto begin = col_idxs + row_ptrs[r];
        const auto end = col_idxs + row_ptrs[r + 1];
        const auto it = std::lower_bound(begin, end, col);
        return (it != end && *it == col) ? values[it - col_idxs]
                                         : zero<ValueType>();
    };

    std::vector<IndexType> extended_c;
    std::vector<IndexType> strong_f;
    std::vector<ValueType> w;
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        auto out_idx = prolong_row_ptrs[row];
        if (status[row] == kernels::pmis::coarse) {
            prolong_col_idxs[out_idx] = coarse_map[row];
            prolong_values[out_idx] = one<ValueType>();
            continue;
        }
        const auto i = static_cast<IndexType>(row);
        ext_i_build_sets(csr, neg_scale, row_sum, diag_abs, strength_threshold,
                         max_row_sum, status, i, extended_c, strong_f);
        if (extended_c.empty()) {
            continue;
        }
        // position of a node in Chat_i, or -1: the sweep classifies by
        // membership and never re-tests strength
        auto slot_of = [&](IndexType node) {
            const auto it =
                std::lower_bound(extended_c.begin(), extended_c.end(), node);
            return (it != extended_c.end() && *it == node)
                       ? static_cast<IndexType>(it - extended_c.begin())
                       : IndexType{-1};
        };

        w.assign(extended_c.size(), zero<ValueType>());
        auto d = lookup(i, i);

        for (auto idx = row_ptrs[i]; idx < row_ptrs[i + 1]; idx++) {
            const auto j = col_idxs[idx];
            const auto a_ij = values[idx];
            if (j == i) {
                continue;
            }
            const auto slot = slot_of(j);
            if (slot >= zero<IndexType>()) {
                w[slot] += a_ij;
                continue;
            }
            const bool is_strong_f_neighbour =
                status[j] != kernels::pmis::coarse &&
                kernels::pmis::is_strong_dep(a_ij, neg_scale[i],
                                             strength_threshold);
            if (!is_strong_f_neighbour) {
                // weak, or a C-point Chat_i does not contain: lumped onto the
                // diagonal
                d += a_ij;
                continue;
            }
            const auto k = j;
            const auto a_ik = a_ij;
            // the sign of row k's own diagonal, not row i's
            const auto sgn_k = real(lookup(k, k)) < zero<real_type>()
                                   ? -one<real_type>()
                                   : one<real_type>();
            // one predicate for both the sigma_k sum and the distribution. Real
            // part only.
            auto admissible = [&](IndexType m, ValueType a_km) {
                return m != k && (m == i || slot_of(m) >= zero<IndexType>()) &&
                       sgn_k * real(a_km) < zero<real_type>();
            };
            ValueType sigma_k = zero<ValueType>();
            for (auto idx2 = row_ptrs[k]; idx2 < row_ptrs[k + 1]; idx2++) {
                if (admissible(col_idxs[idx2], values[idx2])) {
                    sigma_k += values[idx2];
                }
            }
            // exact, with no tolerance: see the note above ext_i_build_sets
            if (sigma_k == zero<ValueType>()) {
                // nowhere to distribute a_ik, so treat k as weak for this row
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
                    w[slot_of(m)] += tau_k * a_km;
                }
            }
        }

        // d mixes signs and can cancel to near zero, so the guard is relative
        // to row_maxabs rather than an exact d != 0 test
        const auto usable = abs(d) >= real_type{1e-3} * row_maxabs[row];
        // the count pass sized this row's slice through ext_i_build_sets too,
        // so the write needs no bound
        for (size_type jj = 0; jj < extended_c.size(); jj++) {
            prolong_col_idxs[out_idx] = coarse_map[extended_c[jj]];
            prolong_values[out_idx] = usable ? -w[jj] / d : zero<ValueType>();
            out_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_EXT_I_INTERPOLATION_FILL);


// truncation of an assembled prolongation; see core/multigrid/pmis_kernels.hpp
// for the two-pass contract
namespace {


// the single row_nnz vs p_max_elmts boundary both kernels must agree on
GKO_INLINE size_type truncated_row_nnz(size_type row_nnz, size_type p_max_elmts)
{
    return std::min(row_nnz, p_max_elmts);
}


}  // namespace


// IndexType-only: see the macro's declaration in
// core/multigrid/pmis_kernels.hpp
template <typename IndexType>
void truncate_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,
    size_type p_max_elmts, const IndexType* in_row_ptrs,
    IndexType* out_row_ptrs)
{
    for (size_type row = 0; row < num_rows; row++) {
        const auto row_nnz =
            static_cast<size_type>(in_row_ptrs[row + 1] - in_row_ptrs[row]);
        out_row_ptrs[row] =
            static_cast<IndexType>(truncated_row_nnz(row_nnz, p_max_elmts));
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_TRUNCATE_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void truncate_interpolation_fill(std::shared_ptr<const DefaultExecutor> exec,
                                 size_type num_rows, size_type p_max_elmts,
                                 const IndexType* in_row_ptrs,
                                 const IndexType* in_col_idxs,
                                 const ValueType* in_values,
                                 const IndexType* out_row_ptrs,
                                 IndexType* out_col_idxs, ValueType* out_values)
{
    std::vector<size_type> order;
    for (size_type row = 0; row < num_rows; row++) {
        const auto begin = in_row_ptrs[row];
        const auto end = in_row_ptrs[row + 1];
        const auto row_nnz = static_cast<size_type>(end - begin);
        auto out_idx = out_row_ptrs[row];
        if (truncated_row_nnz(row_nnz, p_max_elmts) == row_nnz) {
            // untouched: nothing is dropped, so the row sum is preserved
            for (auto idx = begin; idx < end; idx++, out_idx++) {
                out_col_idxs[out_idx] = in_col_idxs[idx];
                out_values[out_idx] = in_values[idx];
            }
            continue;
        }
        // rank by descending |value|, ties by ascending position, so the result
        // does not depend on the sort's tie behaviour
        order.resize(row_nnz);
        for (size_type i = 0; i < row_nnz; i++) {
            order[i] = i;
        }
        std::partial_sort(
            order.begin(),
            order.begin() + static_cast<std::ptrdiff_t>(p_max_elmts),
            order.end(), [&](size_type a, size_type b) {
                const auto abs_a =
                    abs(in_values[begin + static_cast<IndexType>(a)]);
                const auto abs_b =
                    abs(in_values[begin + static_cast<IndexType>(b)]);
                return abs_a > abs_b || (abs_a == abs_b && a < b);
            });
        order.resize(p_max_elmts);
        // restore ascending-column order: the local Galerkin path feeds this
        // straight into a Csr x Csr spgemm
        std::sort(order.begin(), order.end());

        auto original_row_sum = zero<ValueType>();
        for (auto idx = begin; idx < end; idx++) {
            original_row_sum += in_values[idx];
        }
        auto kept_sum = zero<ValueType>();
        for (const auto local : order) {
            kept_sum += in_values[begin + static_cast<IndexType>(local)];
        }
        // rescale the survivors to the original row sum, unless they sum to
        // exactly zero
        const auto scale = kept_sum == zero<ValueType>()
                               ? one<ValueType>()
                               : original_row_sum / kept_sum;
        for (const auto local : order) {
            const auto idx = begin + static_cast<IndexType>(local);
            out_col_idxs[out_idx] = in_col_idxs[idx];
            out_values[out_idx] = in_values[idx] * scale;
            out_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_TRUNCATE_INTERPOLATION_FILL);


}  // namespace pmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
