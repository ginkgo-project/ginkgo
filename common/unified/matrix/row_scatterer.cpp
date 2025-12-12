// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/matrix/row_scatterer_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace row_scatter {


template <typename ValueType, typename OutputType, typename IndexType>
void row_scatter(std::shared_ptr<const DefaultExecutor> exec,
                 const array<IndexType>* row_idxs,
                 const matrix::Dense<ValueType>* orig,
                 matrix::Dense<OutputType>* target, bool& invalid_access)
{
    array<bool> invalid_access_arr{exec, {false}};
    run_kernel(
        exec,
        [num_rows = target->get_size()[0]] GKO_KERNEL(
            auto row, auto col, auto orig, auto rows, auto scattered,
            auto* invalid_access_ptr) {
            if (rows[row] >= num_rows) {
                *invalid_access_ptr = true;
                return;
            }
            scattered(rows[row], col) = orig(row, col);
        },
        dim<2>{row_idxs->get_size(), orig->get_size()[1]}, orig, *row_idxs,
        target, invalid_access_arr.get_data());
    invalid_access = exec->copy_val_to_host(invalid_access_arr.get_data());
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_ROW_SCATTER_SIMPLE_APPLY);


template <typename ValueType, typename OutputType, typename IndexType>
void advanced_row_scatter(std::shared_ptr<const DefaultExecutor> exec,
                          const array<IndexType>* row_idxs,
                          const matrix::Dense<ValueType>* alpha,
                          const matrix::Dense<ValueType>* orig,
                          const matrix::Dense<OutputType>* beta,
                          matrix::Dense<OutputType>* target,
                          bit_packed_span<bool, IndexType, uint32> mask,
                          bool& invalid_access) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_ROW_SCATTER_ADVANCED_APPLY);


}  // namespace row_scatter
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
