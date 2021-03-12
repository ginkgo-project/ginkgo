/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/multigrid/amgx_pgm_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The AMGX_PGM solver namespace.
 *
 * @ingroup amgx_pgm
 */
namespace amgx_pgm {


template <typename IndexType>
void match_edge(std::shared_ptr<const ReferenceExecutor> exec,
                const Array<IndexType> &strongest_neighbor,
                Array<IndexType> &agg)
{
    auto agg_vals = agg.get_data();
    auto strongest_neighbor_vals = strongest_neighbor.get_const_data();
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        if (agg_vals[i] == -1) {
            auto neighbor = strongest_neighbor_vals[i];
            if (neighbor != -1 && strongest_neighbor_vals[neighbor] == i) {
                agg_vals[i] = i;
                agg_vals[neighbor] = i;
                // Use the smaller index as agg point
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const ReferenceExecutor> exec,
                 const Array<IndexType> &agg, IndexType *num_unagg)
{
    IndexType unagg = 0;
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        unagg += (agg.get_const_data()[i] == -1);
    }
    *num_unagg = unagg;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const ReferenceExecutor> exec,
              Array<IndexType> &agg, IndexType *num_agg)
{
    const auto num = agg.get_num_elems();
    Array<IndexType> agg_map(exec, num + 1);
    auto agg_vals = agg.get_data();
    auto agg_map_vals = agg_map.get_data();
    for (size_type i = 0; i < num + 1; i++) {
        agg_map_vals[i] = 0;
    }
    for (size_type i = 0; i < num; i++) {
        agg_map_vals[agg_vals[i]] = 1;
    }
    components::prefix_sum(exec, agg_map_vals, num + 1);
    for (size_type i = 0; i < num; i++) {
        agg_vals[i] = agg_map_vals[agg_vals[i]];
    }
    *num_agg = agg_map_vals[num];
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *weight_mtx,
    const matrix::Diagonal<ValueType> *diag, Array<IndexType> &agg,
    Array<IndexType> &strongest_neighbor)
{
    const auto row_ptrs = weight_mtx->get_const_row_ptrs();
    const auto col_idxs = weight_mtx->get_const_col_idxs();
    const auto vals = weight_mtx->get_const_values();
    const auto diag_vals = diag->get_const_values();
    for (size_type row = 0; row < agg.get_num_elems(); row++) {
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
                    (weight > max_weight_unagg ||
                     (weight == max_weight_unagg && col > strongest_unagg))) {
                    max_weight_unagg = weight;
                    strongest_unagg = col;
                } else if (agg.get_const_data()[col] != -1 &&
                           (weight > max_weight_agg ||
                            (weight == max_weight_agg &&
                             col > strongest_agg))) {
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
    GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR);


template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const ReferenceExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *weight_mtx,
                         const matrix::Diagonal<ValueType> *diag,
                         Array<IndexType> &agg,
                         Array<IndexType> &intermediate_agg)
{
    const auto row_ptrs = weight_mtx->get_const_row_ptrs();
    const auto col_idxs = weight_mtx->get_const_col_idxs();
    const auto vals = weight_mtx->get_const_values();
    const auto agg_const_val = agg.get_const_data();
    auto agg_val = (intermediate_agg.get_num_elems() > 0)
                       ? intermediate_agg.get_data()
                       : agg.get_data();
    const auto diag_vals = diag->get_const_values();
    for (IndexType row = 0; row < agg.get_num_elems(); row++) {
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
                (weight > max_weight_agg ||
                 (weight == max_weight_agg && col > strongest_agg))) {
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

    if (intermediate_agg.get_num_elems() > 0) {
        // Copy the intermediate_agg to agg
        agg = intermediate_agg;
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG);


template <typename ValueType, typename IndexType>
void amgx_pgm_generate(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       const Array<IndexType> &agg,
                       matrix::Csr<ValueType, IndexType> *coarse)
{
    // agg[i] -> I, agg[j] -> J
    const auto coarse_nrows = coarse->get_size()[0];
    const auto source_nrows = source->get_size()[0];
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_vals = source->get_const_values();
    gko::vector<gko::map<IndexType, ValueType>> row_list(
        source_nrows, gko::map<IndexType, ValueType>{exec}, exec);
    for (size_type i = 0; i < source_nrows; i++) {
        IndexType row_idx = agg.get_const_data()[i];
        for (auto j = source_row_ptrs[i]; j < source_row_ptrs[i + 1]; j++) {
            const auto col = agg.get_const_data()[source_col_idxs[j]];
            const auto val = source_vals[j];
            row_list[row_idx][col] += val;
        }
    }
    auto coarse_row_ptrs = coarse->get_row_ptrs();
    for (size_type i = 0; i < coarse_nrows; i++) {
        coarse_row_ptrs[i] = row_list[i].size();
    }
    components::prefix_sum(exec, coarse_row_ptrs, coarse_nrows + 1);

    auto nnz = coarse_row_ptrs[coarse_nrows];
    matrix::CsrBuilder<ValueType, IndexType> coarse_builder{coarse};
    auto &coarse_col_idxs_array = coarse_builder.get_col_idx_array();
    auto &coarse_vals_array = coarse_builder.get_value_array();
    coarse_col_idxs_array.resize_and_reset(nnz);
    coarse_vals_array.resize_and_reset(nnz);
    auto coarse_col_idxs = coarse_col_idxs_array.get_data();
    auto coarse_vals = coarse_vals_array.get_data();

    for (size_type i = 0; i < coarse_nrows; i++) {
        auto ind = coarse_row_ptrs[i];
        for (auto pair : row_list[i]) {
            coarse_col_idxs[ind] = pair.first;
            coarse_vals[ind] = pair.second;
            ind++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_GENERATE);


}  // namespace amgx_pgm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
