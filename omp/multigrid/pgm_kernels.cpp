// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"


#include <algorithm>
#include <memory>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>


#include "core/base/iterator_factory.hpp"


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
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto it = detail::make_zip_iterator(row_idxs, col_idxs);
    std::sort(it, it + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


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
