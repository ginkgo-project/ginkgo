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
    using rc = remove_complex<ValueType>;

    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; ++row) {
        // get the max in the row except diagonal
        rc max_abs = rc{0};
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
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
    using rc = remove_complex<ValueType>;

    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; ++row) {
        // count the number of strongest neighbor
        IndexType count = 0;
        auto max_abs = row_maxabs[row];
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] == row) {
                continue;
            }

            if (max_abs > zero<rc>() &&
                abs(vals[idx]) >= strength_threshold * max_abs) {
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


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    using rc = remove_complex<ValueType>;

    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto s_row_ptrs = strong_dep->get_const_row_ptrs();
    const auto s_col_idxs = strong_dep->get_const_col_idxs();

    for (auto r = 0; r < nrows; ++r) {
        weight[r] = rc{0};
    }

    for (auto r = 0; r < nrows; ++r) {
        for (auto p = s_row_ptrs[r]; p < s_row_ptrs[r + 1]; ++p) {
            auto c = s_col_idxs[p];
            weight[c] += rc{1};
        }
    }
    for (auto i = 0; i < nrows; ++i) {
        status[i] = (weight[i] == 0 ? 1 : 0);
        weight[i] += static_cast<rc>(dist(gen));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
              const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
              const int* status, int* new_status)
{
    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto s_row_ptrs = strong_dep->get_const_row_ptrs();
    const auto s_col_idxs = strong_dep->get_const_col_idxs();

    for (IndexType row = 0; row < nrows; row++) {
        // 0 is unassigned yet
        auto ans = status[row];
        if (status[row] == 0) {
            const auto row_start = s_row_ptrs[row];
            const auto row_end = s_row_ptrs[row + 1];
            bool is_coarse = true;
            for (IndexType j = row_start; j < row_end; ++j) {
                auto c = s_col_idxs[j];
                if (status[c] == 0 && weight[c] >= weight[row]) {
                    is_coarse = false;
                    break;
                }
            }
            if (is_coarse) {
                ans = 2;
            }
        }
        new_status[row] = ans;
    }
    // mark all points strongly influenced by the new coarse points to fine
    // group
    for (IndexType row = 0; row < nrows; row++) {
        if (new_status[row] == 2 && new_status[row] != status[row]) {
            for (auto idx = trans_strong_dep->get_const_row_ptrs()[row];
                 idx < trans_strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
                // It is correct even if more than one threads might assign the
                // value
                auto col = trans_strong_dep->get_const_col_idxs()[idx];
                if (new_status[col] == 0) {
                    new_status[col] = 1;
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
        if (status[i] == 0) {
            ans++;
        }
    }
    *num_unassigned = ans;
}


}  // namespace pmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
