// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Coo matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto orig_values, auto orig_row_idxs,
                      auto orig_col_idxs, auto diag) {
            if (orig_row_idxs[tidx] == orig_col_idxs[tidx]) {
                diag[orig_row_idxs[tidx]] = orig_values[tidx];
            }
        },
        orig->get_num_stored_elements(), orig->get_const_values(),
        orig->get_const_row_idxs(), orig->get_const_col_idxs(),
        diag->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Coo<ValueType, IndexType>* orig,
                   matrix::Dense<ValueType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto orig_values, auto orig_row_idxs,
                      auto orig_col_idxs, auto result) {
            result(orig_row_idxs[tidx], orig_col_idxs[tidx]) =
                orig_values[tidx];
        },
        orig->get_num_stored_elements(), orig->get_const_values(),
        orig->get_const_row_idxs(), orig->get_const_col_idxs(), result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL);


}  // namespace coo
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
