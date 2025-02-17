// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/uniform_coarsening_kernels.hpp"

#include <memory>
#include <tuple>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>

#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace uniform_coarsening {


template <typename ValueType, typename IndexType>
void fill_restrict_op(std::shared_ptr<const DefaultExecutor> exec,
                      const array<IndexType>* coarse_rows,
                      matrix::Csr<ValueType, IndexType>* restrict_op)
{
    auto num_rows = restrict_op->get_size()[0];
    auto num_cols = restrict_op->get_size()[1];
    GKO_ASSERT(num_cols == coarse_rows->get_num_elems());
    GKO_ASSERT(num_cols >= num_rows);
    auto coarse_data = coarse_rows->get_const_data();

    for (IndexType j = 0; j < num_cols; ++j) {
        if (coarse_data[j] >= 0) {
            restrict_op->get_col_idxs()[coarse_data[j]] = j;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_RESTRICT_OP);


template <typename IndexType>
void fill_incremental_indices(std::shared_ptr<const DefaultExecutor> exec,
                              size_type coarse_skip,
                              array<IndexType>* coarse_rows)
{
    for (IndexType i = 0; i < coarse_rows->get_size(); i += coarse_skip) {
        coarse_rows->get_data()[i] = i / coarse_skip;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_INCREMENTAL_INDICES);


}  // namespace uniform_coarsening
}  // namespace reference
}  // namespace kernels
}  // namespace gko
