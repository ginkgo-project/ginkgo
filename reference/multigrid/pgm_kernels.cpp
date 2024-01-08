// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"


#include <algorithm>
#include <memory>
#include <tuple>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PGM solver namespace.
 *
 * @ingroup pgm
 */
namespace pgm {


template <typename IndexType>
void match_edge(std::shared_ptr<const ReferenceExecutor> exec,
                const array<IndexType>& strongest_neighbor,
                array<IndexType>& agg)
{
    auto agg_vals = agg.get_data();
    auto strongest_neighbor_vals = strongest_neighbor.get_const_data();
    for (size_type i = 0; i < agg.get_size(); i++) {
        if (agg_vals[i] == -1) {
            auto neighbor = strongest_neighbor_vals[i];
            // i < neighbor always holds when neighbor is not -1
            if (neighbor != -1 && strongest_neighbor_vals[neighbor] == i &&
                i <= neighbor) {
                // Use the smaller index as agg point
                agg_vals[i] = i;
                agg_vals[neighbor] = i;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const ReferenceExecutor> exec,
                 const array<IndexType>& agg, IndexType* num_unagg)
{
    IndexType unagg = 0;
    for (size_type i = 0; i < agg.get_size(); i++) {
        unagg += (agg.get_const_data()[i] == -1);
    }
    *num_unagg = unagg;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const ReferenceExecutor> exec,
              array<IndexType>& agg, IndexType* num_agg)
{
    const auto num = agg.get_size();
    array<IndexType> agg_map(exec, num + 1);
    auto agg_vals = agg.get_data();
    auto agg_map_vals = agg_map.get_data();
    for (size_type i = 0; i < num + 1; i++) {
        agg_map_vals[i] = 0;
    }
    for (size_type i = 0; i < num; i++) {
        agg_map_vals[agg_vals[i]] = 1;
    }
    components::prefix_sum_nonnegative(exec, agg_map_vals, num + 1);
    for (size_type i = 0; i < num; i++) {
        agg_vals[i] = agg_map_vals[agg_vals[i]];
    }
    *num_agg = agg_map_vals[num];
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_RENUMBER_KERNEL);


template <typename IndexType>
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto it = detail::make_zip_iterator(row_idxs, col_idxs);
    std::sort(it, it + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


template <typename IndexType>
void map_row(std::shared_ptr<const DefaultExecutor> exec,
             size_type num_fine_row, const IndexType* fine_row_ptrs,
             const IndexType* agg, IndexType* row_idxs)
{
    for (size_type row = 0; row < num_fine_row; row++) {
        const auto coarse_row = agg[row];
        for (auto i = fine_row_ptrs[row]; i < fine_row_ptrs[row + 1]; i++) {
            row_idxs[i] = coarse_row;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MAP_ROW_KERNEL);


template <typename IndexType>
void map_col(std::shared_ptr<const DefaultExecutor> exec, size_type nnz,
             const IndexType* fine_col_idxs, const IndexType* agg,
             IndexType* col_idxs)
{
    for (size_type i = 0; i < nnz; i++) {
        col_idxs[i] = agg[fine_col_idxs[i]];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MAP_COL_KERNEL);


template <typename IndexType>
void count_unrepeated_nnz(std::shared_ptr<const DefaultExecutor> exec,
                          size_type nnz, const IndexType* row_idxs,
                          const IndexType* col_idxs, size_type* coarse_nnz)
{
    if (nnz > 1) {
        size_type result = 0;
        for (size_type i = 0; i < nnz - 1; i++) {
            if (row_idxs[i] != row_idxs[i + 1] ||
                col_idxs[i] != col_idxs[i + 1]) {
                result++;
            }
        }
        *coarse_nnz = result + 1;
    } else {
        *coarse_nnz = nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PGM_COUNT_UNREPEATED_NNZ_KERNEL);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* weight_mtx,
    const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg,
    array<IndexType>& strongest_neighbor)
{
    const auto row_ptrs = weight_mtx->get_const_row_ptrs();
    const auto col_idxs = weight_mtx->get_const_col_idxs();
    const auto vals = weight_mtx->get_const_values();
    const auto diag_vals = diag->get_const_values();
    for (size_type row = 0; row < agg.get_size(); row++) {
        auto max_weight_unagg = zero<ValueType>();
        auto max_weight_agg = zero<ValueType>();
        IndexType strongest_unagg = -1;
        IndexType strongest_agg = -1;
        if (agg.get_const_data()[row] == -1) {
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto col = col_idxs[idx];
                if (col == row) {
                    continue;
                }
                auto weight =
                    vals[idx] / max(abs(diag_vals[row]), abs(diag_vals[col]));
                if (agg.get_const_data()[col] == -1 &&
                    std::tie(weight, col) >
                        std::tie(max_weight_unagg, strongest_unagg)) {
                    max_weight_unagg = weight;
                    strongest_unagg = col;
                } else if (agg.get_const_data()[col] != -1 &&
                           std::tie(weight, col) >
                               std::tie(max_weight_agg, strongest_agg)) {
                    max_weight_agg = weight;
                    strongest_agg = col;
                }
            }

            if (strongest_unagg == -1 && strongest_agg != -1) {
                // all neighbor is agg, connect to the strongest agg
                agg.get_data()[row] = agg.get_data()[strongest_agg];
            } else if (strongest_unagg != -1) {
                // set the strongest neighbor in the unagg group
                strongest_neighbor.get_data()[row] = strongest_unagg;
            } else {
                // no neighbor
                strongest_neighbor.get_data()[row] = row;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_FIND_STRONGEST_NEIGHBOR);


template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const ReferenceExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* weight_mtx,
                         const matrix::Diagonal<ValueType>* diag,
                         array<IndexType>& agg,
                         array<IndexType>& intermediate_agg)
{
    const auto row_ptrs = weight_mtx->get_const_row_ptrs();
    const auto col_idxs = weight_mtx->get_const_col_idxs();
    const auto vals = weight_mtx->get_const_values();
    const auto agg_const_val = agg.get_const_data();
    auto agg_val = (intermediate_agg.get_size() > 0)
                       ? intermediate_agg.get_data()
                       : agg.get_data();
    const auto diag_vals = diag->get_const_values();
    for (IndexType row = 0; row < agg.get_size(); row++) {
        if (agg_const_val[row] != -1) {
            continue;
        }
        auto max_weight_agg = zero<ValueType>();
        IndexType strongest_agg = -1;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            auto col = col_idxs[idx];
            if (col == row) {
                continue;
            }
            auto weight =
                vals[idx] / max(abs(diag_vals[row]), abs(diag_vals[col]));
            if (agg_const_val[col] != -1 &&
                std::tie(weight, col) >
                    std::tie(max_weight_agg, strongest_agg)) {
                max_weight_agg = weight;
                strongest_agg = col;
            }
        }
        if (strongest_agg != -1) {
            agg_val[row] = agg_const_val[strongest_agg];
        } else {
            agg_val[row] = row;
        }
    }

    if (intermediate_agg.get_size() > 0) {
        // Copy the intermediate_agg to agg
        agg = intermediate_agg;
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_ASSIGN_TO_EXIST_AGG);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec, size_type nnz,
                    IndexType* row_idxs, IndexType* col_idxs, ValueType* vals)
{
    auto it = detail::make_zip_iterator(row_idxs, col_idxs, vals);
    std::stable_sort(it, it + nnz, [](auto a, auto b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) <
               std::tie(std::get<0>(b), std::get<1>(b));
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_SORT_ROW_MAJOR);


template <typename ValueType, typename IndexType>
void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,
                        size_type fine_nnz, const IndexType* row_idxs,
                        const IndexType* col_idxs, const ValueType* vals,
                        matrix::Coo<ValueType, IndexType>* coarse_coo)
{
    auto coarse_row = coarse_coo->get_row_idxs();
    auto coarse_col = coarse_coo->get_col_idxs();
    auto coarse_val = coarse_coo->get_values();
    IndexType row = 0;
    size_type idxs = 0;
    size_type coarse_idxs = 0;
    IndexType curr_row = row_idxs[0];
    IndexType curr_col = col_idxs[0];
    ValueType temp_val = vals[0];
    for (size_type idxs = 1; idxs < fine_nnz; idxs++) {
        if (curr_row != row_idxs[idxs] || curr_col != col_idxs[idxs]) {
            coarse_row[coarse_idxs] = curr_row;
            coarse_col[coarse_idxs] = curr_col;
            coarse_val[coarse_idxs] = temp_val;
            curr_row = row_idxs[idxs];
            curr_col = col_idxs[idxs];
            temp_val = vals[idxs];
            coarse_idxs++;
            continue;
        }
        temp_val += vals[idxs];
    }
    GKO_ASSERT(coarse_idxs + 1 == coarse_coo->get_num_stored_elements());
    coarse_row[coarse_idxs] = curr_row;
    coarse_col[coarse_idxs] = curr_col;
    coarse_val[coarse_idxs] = temp_val;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_COMPUTE_COARSE_COO);


}  // namespace pgm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
