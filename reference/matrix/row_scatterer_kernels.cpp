// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/row_scatterer_kernels.hpp"

#include "core/base/mixed_precision_types.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace row_scatter {


template <typename ValueType, typename OutputType, typename IndexType>
void row_scatter(std::shared_ptr<const ReferenceExecutor> exec,
                 const array<IndexType>* row_idxs,
                 const matrix::Dense<ValueType>* orig,
                 matrix::Dense<OutputType>* target, bool& invalid_access)
{
    auto rows = row_idxs->get_const_data();
    for (size_type i = 0; i < row_idxs->get_size(); ++i) {
        if (rows[i] >= target->get_size()[0]) {
            invalid_access = true;
            return;
        }
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            target->at(rows[i], j) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_ROW_SCATTER_SIMPLE_APPLY);


}  // namespace row_scatter
}  // namespace reference
}  // namespace kernels
}  // namespace gko
