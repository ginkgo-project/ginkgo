// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_diagonal_kernels.hpp"


// Copyright (c) 2017-2023, the Ginkgo authors
#include <algorithm>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Diagonal matrix format namespace.
 * @ref Diagonal
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


#include "reference/matrix/batch_diagonal_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Diagonal<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto b_stride = b->get_common_size()[1];
    const auto x_stride = x->get_common_size()[1];
    const auto nrows = static_cast<int>(mat->get_common_size()[0]);
    for (size_type batch_id = 0; batch_id < b->get_num_batch_items();
         ++batch_id) {
        simple_apply(nrows, mat->get_const_values_for_item(batch_id),
                     x->get_common_size()[1], b_stride,
                     b->get_const_values_for_item(batch_id), x_stride,
                     x->get_values_for_item(batch_id));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIAGONAL_SIMPLE_APPLY_KERNEL);


}  // namespace batch_diagonal
}  // namespace reference
}  // namespace kernels
}  // namespace gko
