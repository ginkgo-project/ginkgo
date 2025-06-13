// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"

#include <algorithm>
#include <memory>

#include <omp.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "core/base/iterator_factory.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The PGM solver namespace.
 *
 * @ingroup pgm
 */
namespace pgm {


template <typename IndexType>
void match_edge(std::shared_ptr<const DefaultExecutor> exec,
                const array<IndexType>& strongest_neighbor,
                array<IndexType>& agg)
{
    auto agg_vals = agg.get_data();
    auto strongest_neighbor_vals = strongest_neighbor.get_const_data();
#pragma omp parallel for
    for (int64 i = 0; i < static_cast<int64>(agg.get_size()); i++) {
        if (load(agg_vals + i) != -1) {
            continue;
        }
        auto neighbor = strongest_neighbor_vals[i];
        if (neighbor != -1 && strongest_neighbor_vals[neighbor] == i &&
            i <= neighbor) {
            store(agg_vals + i, i);
            store(agg_vals + neighbor, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto it = detail::make_zip_iterator(row_idxs, col_idxs);
    std::sort(it, it + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


template <typename ValueType, typename IndexType>
void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,
                        size_type fine_nnz, const IndexType* row_idxs,
                        const IndexType* col_idxs, const ValueType* vals,
                        matrix::Coo<ValueType, IndexType>* coarse_coo)
{
    auto coarse_row = coarse_coo->get_row_idxs();
    auto coarse_col = coarse_coo->get_col_idxs();
    auto coarse_val = coarse_coo->get_values();
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
}  // namespace omp
}  // namespace kernels
}  // namespace gko
