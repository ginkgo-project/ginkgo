// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <random>
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
 * @ingroup pmis
 */
namespace pmis {


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        remove_complex<ValueType>* row_maxabs)
{
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // get the max in the row except diagonal
        auto max_abs = zero<remove_complex<ValueType>>();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (col_idxs[idx] == row) {
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
            if (col_idxs[idx] == row) {
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
            if (csr->get_const_col_idxs()[idx] == row) {
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


template <typename ValueType>
void initialize_random_weight(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num, ValueType* weight)
{
    std::default_random_engine gen(42);
    std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
    for (size_type row = 0; row < num; row++) {
        weight[row] = dist(gen);
    }
}
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    // we can not use half, bfloat16 with random generator
    // generate it in double and then cast to corresponding type
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const auto nrows = static_cast<IndexType>(trans_strong_dep->get_size()[0]);
    const auto row_ptrs = trans_strong_dep->get_const_row_ptrs();

    for (size_type row = 0; row < nrows; row++) {
        weight[row] = static_cast<remove_complex<ValueType>>(row_ptrs[row + 1] -
                                                             row_ptrs[row]);
        status[row] = (weight[row] == 0 ? 0 : -1);
        weight[row] += static_cast<remove_complex<ValueType>>(dist(gen));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::Csr<ValueType, IndexType>* csr,
              const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
              const int* status, int* new_status)
{
    const auto nrows = static_cast<IndexType>(csr->get_size()[0]);
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();

    for (IndexType row = 0; row < nrows; row++) {
        // -1 is unassigned yet
        auto ans = status[row];
        if (status[row] == -1) {
            // works on the original graph
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            bool is_coarse = true;
            for (IndexType idx = row_start; idx < row_end; idx++) {
                auto col = col_idxs[idx];
                if (col == row) {
                    continue;
                }
                if (status[col] == -1 && weight[col] >= weight[row]) {
                    is_coarse = false;
                    break;
                }
            }
            if (is_coarse) {
                ans = 1;
            }
        }
        new_status[row] = ans;
    }
    // mark new fine point strongly influenced by the new coarse points
    for (IndexType row = 0; row < nrows; row++) {
        if (new_status[row] == 1 && new_status[row] != status[row]) {
            for (auto idx = trans_strong_dep->get_const_row_ptrs()[row];
                 idx < trans_strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
                // It is correct even if more than one threads might assign the
                // value
                auto col = trans_strong_dep->get_const_col_idxs()[idx];
                if (new_status[col] == -1) {
                    new_status[col] = 0;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_CLASSIFY_KERNEL);


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    size_type ans = 0;
    for (size_type i = 0; i < num; i++) {
        if (status[i] == -1) {
            ans++;
        }
    }
    *num_unassigned = ans;
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_ptr)
{
    for (size_type row = 0; row < strong_dep->get_size()[0]; row++) {
        IndexType num = 0;
        if (status[row] == 1) {
            prolong_row_ptr[row] = 1;
            continue;
        }
        for (auto idx = strong_dep->get_const_row_ptrs()[row];
             idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
            auto col = strong_dep->get_const_col_idxs()[idx];
            if (status[col] == 1) {
                num++;
            }
        }
        prolong_row_ptr[row] = num;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    auto csr_values = csr->get_const_values();
    auto csr_col_idxs = csr->get_const_col_idxs();
    auto csr_row_ptrs = csr->get_const_row_ptrs();
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (coarse_map[row] != coarse_map[row + 1]) {
            auto idx = prolong_row_ptrs[row];
            prolong_col_idxs[idx] = coarse_map[row];
            prolong_values[idx] = one<ValueType>();
            continue;
        }
        auto pos = zero<ValueType>();
        auto pos_divisor = zero<ValueType>();
        auto neg = zero<ValueType>();
        auto neg_divisor = zero<ValueType>();
        auto diag = zero<ValueType>();
        bool enable_neg = false;
        bool enable_pos = false;
        // first compute alpha/beta
        auto max_abs = row_maxabs[row];
        for (auto idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1]; idx++) {
            auto val = csr_values[idx];
            auto col = csr_col_idxs[idx];
            if (col == row) {
                diag = val;
                continue;
            }
            if (real(val) >= 0) {
                pos += val;
                if (coarse_map[col] != coarse_map[col + 1] &&
                    abs(val) >= strength_threshold * max_abs) {
                    pos_divisor += val;
                    enable_pos = true;
                }
            } else {
                neg += val;
                if (coarse_map[col] != coarse_map[col + 1] &&
                    abs(val) >= strength_threshold * max_abs) {
                    neg_divisor += val;
                    enable_neg = true;
                }
            }
        }
        pos = safe_divide(pos, pos_divisor);
        neg = safe_divide(neg, neg_divisor);
        if (!enable_neg && !enable_pos) {
            continue;
        }
        auto start = prolong_row_ptrs[row];
        for (auto idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1]; idx++) {
            auto val = csr_values[idx];
            auto col = csr_col_idxs[idx];
            if (col == row || abs(val) < strength_threshold * max_abs) {
                continue;
            }
            if (real(val) >= 0 && enable_pos &&
                coarse_map[col] != coarse_map[col + 1]) {
                prolong_col_idxs[start] = coarse_map[col];
                prolong_values[start] = -pos * val / diag;
                start++;
            }
            if (real(val) < 0 && enable_neg &&
                coarse_map[col] != coarse_map[col + 1]) {
                prolong_col_idxs[start] = coarse_map[col];
                prolong_values[start] = -neg * val / diag;
                start++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL);


}  // namespace pmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
