// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Pgm namespace.
 *
 * @ingroup pgm
 */
namespace pgm {


template <typename IndexType>
void match_edge(std::shared_ptr<const DefaultExecutor> exec,
                const array<IndexType>& strongest_neighbor,
                array<IndexType>& agg)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto strongest_neighbor_vals, auto agg_vals) {
            if (agg_vals[tidx] != -1) {
                return;
            }
            auto neighbor = strongest_neighbor_vals[tidx];
            if (neighbor != -1 && strongest_neighbor_vals[neighbor] == tidx &&
                tidx <= neighbor) {
                // Use the smaller index as agg point
                agg_vals[tidx] = tidx;
                agg_vals[neighbor] = tidx;
            }
        },
        agg.get_size(), strongest_neighbor.get_const_data(), agg.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const DefaultExecutor> exec,
                 const array<IndexType>& agg, IndexType* num_unagg)
{
    array<IndexType> d_result(exec, 1);
    run_kernel_reduction(
        exec, [] GKO_KERNEL(auto i, auto array) { return array[i] == -1; },
        GKO_KERNEL_REDUCE_SUM(IndexType), d_result.get_data(), agg.get_size(),
        agg);

    *num_unagg = get_element(d_result, 0);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const DefaultExecutor> exec,
              array<IndexType>& agg, IndexType* num_agg)
{
    const auto num = agg.get_size();
    array<IndexType> agg_map(exec, num + 1);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto agg, auto agg_map) {
            // agg_vals[i] == i always holds in the aggregated group whose
            // identifier is
            // i because we use the index of element as the aggregated group
            // identifier.
            agg_map[tidx] = (agg[tidx] == tidx);
        },
        num, agg.get_const_data(), agg_map.get_data());

    components::prefix_sum_nonnegative(exec, agg_map.get_data(),
                                       agg_map.get_size());

    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto map, auto agg) {
            agg[tidx] = map[agg[tidx]];
        },
        num, agg_map.get_const_data(), agg.get_data());
    *num_agg = get_element(agg_map, num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_RENUMBER_KERNEL);


template <typename IndexType>
void map_row(std::shared_ptr<const DefaultExecutor> exec,
             size_type num_fine_row, const IndexType* fine_row_ptrs,
             const IndexType* agg, IndexType* row_idxs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto fine_row_ptrs, auto agg, auto row_idxs) {
            const auto coarse_row = agg[tidx];
            // TODO: when it is necessary, it can use warp per row to improve.
            for (auto i = fine_row_ptrs[tidx]; i < fine_row_ptrs[tidx + 1];
                 i++) {
                row_idxs[i] = coarse_row;
            }
        },
        num_fine_row, fine_row_ptrs, agg, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MAP_ROW_KERNEL);


template <typename IndexType>
void map_col(std::shared_ptr<const DefaultExecutor> exec, size_type nnz,
             const IndexType* fine_col_idxs, const IndexType* agg,
             IndexType* col_idxs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto fine_col_idxs, auto agg, auto col_idxs) {
            col_idxs[tidx] = agg[fine_col_idxs[tidx]];
        },
        nnz, fine_col_idxs, agg, col_idxs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MAP_COL_KERNEL);


template <typename IndexType>
void count_unrepeated_nnz(std::shared_ptr<const DefaultExecutor> exec,
                          size_type nnz, const IndexType* row_idxs,
                          const IndexType* col_idxs, size_type* coarse_nnz)
{
    if (nnz > 1) {
        array<IndexType> d_result(exec, 1);
        run_kernel_reduction(
            exec,
            [] GKO_KERNEL(auto i, auto row_idxs, auto col_idxs) {
                return row_idxs[i] != row_idxs[i + 1] ||
                       col_idxs[i] != col_idxs[i + 1];
            },
            GKO_KERNEL_REDUCE_SUM(IndexType), d_result.get_data(), nnz - 1,
            row_idxs, col_idxs);
        *coarse_nnz = static_cast<size_type>(get_element(d_result, 0) + 1);
    } else {
        *coarse_nnz = nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PGM_COUNT_UNREPEATED_NNZ_KERNEL);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* weight_mtx,
    const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg,
    array<IndexType>& strongest_neighbor)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto weight_vals,
                      auto diag, auto agg, auto strongest_neighbor) {
            using value_type = device_type<ValueType>;
            auto max_weight_unagg = zero<value_type>();
            auto max_weight_agg = zero<value_type>();
            IndexType strongest_unagg = -1;
            IndexType strongest_agg = -1;
            if (agg[row] != -1) {
                return;
            }
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto col = col_idxs[idx];
                if (col == row) {
                    continue;
                }
                auto weight =
                    weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
                if (agg[col] == -1 &&
                    device_std::tie(weight, col) >
                        device_std::tie(max_weight_unagg, strongest_unagg)) {
                    max_weight_unagg = weight;
                    strongest_unagg = col;
                } else if (agg[col] != -1 &&
                           device_std::tie(weight, col) >
                               device_std::tie(max_weight_agg, strongest_agg)) {
                    max_weight_agg = weight;
                    strongest_agg = col;
                }
            }

            if (strongest_unagg == -1 && strongest_agg != -1) {
                // all neighbor is agg, connect to the strongest agg
                // Also, no others will use this item as their
                // strongest_neighbor because they are already aggregated. Thus,
                // it is deterministic behavior
                agg[row] = agg[strongest_agg];
            } else if (strongest_unagg != -1) {
                // set the strongest neighbor in the unagg group
                strongest_neighbor[row] = strongest_unagg;
            } else {
                // no neighbor
                strongest_neighbor[row] = row;
            }
        },
        agg.get_size(), weight_mtx->get_const_row_ptrs(),
        weight_mtx->get_const_col_idxs(), weight_mtx->get_const_values(),
        diag->get_const_values(), agg.get_data(),
        strongest_neighbor.get_data());
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_FIND_STRONGEST_NEIGHBOR);

template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* weight_mtx,
                         const matrix::Diagonal<ValueType>* diag,
                         array<IndexType>& agg,
                         array<IndexType>& intermediate_agg)
{
    const auto num = agg.get_size();
    if (intermediate_agg.get_size() > 0) {
        // deterministic kernel
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs,
                          auto weight_vals, auto diag, auto agg_const_val,
                          auto agg_val) {
                if (agg_val[row] != -1) {
                    return;
                }
                using value_type = device_type<ValueType>;
                auto max_weight_agg = zero<value_type>();
                IndexType strongest_agg = -1;
                for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                    auto col = col_idxs[idx];
                    if (col == row) {
                        continue;
                    }
                    auto weight =
                        weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
                    if (agg_const_val[col] != -1 &&
                        device_std::tie(weight, col) >
                            device_std::tie(max_weight_agg, strongest_agg)) {
                        max_weight_agg = weight;
                        strongest_agg = col;
                    }
                }
                if (strongest_agg != -1) {
                    agg_val[row] = agg_const_val[strongest_agg];
                } else {
                    agg_val[row] = row;
                }
            },
            num, weight_mtx->get_const_row_ptrs(),
            weight_mtx->get_const_col_idxs(), weight_mtx->get_const_values(),
            diag->get_const_values(), agg.get_const_data(),
            intermediate_agg.get_data());
        // Copy the intermediate_agg to agg
        agg = intermediate_agg;
    } else {
        // undeterminstic kernel
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs,
                          auto weight_vals, auto diag, auto agg_val) {
                if (agg_val[row] != -1) {
                    return;
                }
                using value_type = device_type<ValueType>;
                auto max_weight_agg = zero<value_type>();
                IndexType strongest_agg = -1;
                for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                    auto col = col_idxs[idx];
                    if (col == row) {
                        continue;
                    }
                    auto weight =
                        weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
                    if (agg_val[col] != -1 &&
                        device_std::tie(weight, col) >
                            device_std::tie(max_weight_agg, strongest_agg)) {
                        max_weight_agg = weight;
                        strongest_agg = col;
                    }
                }
                if (strongest_agg != -1) {
                    agg_val[row] = agg_val[strongest_agg];
                } else {
                    agg_val[row] = row;
                }
            },
            num, weight_mtx->get_const_row_ptrs(),
            weight_mtx->get_const_col_idxs(), weight_mtx->get_const_values(),
            diag->get_const_values(), agg.get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_ASSIGN_TO_EXIST_AGG);


}  // namespace pgm
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
