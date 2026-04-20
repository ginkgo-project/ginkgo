// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/rs_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "ginkgo/core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Rs namespace.
 *
 * @ingroup rs
 */
namespace rs {

template <typename ValueType, typename IndexType>
void compute_soc_and_run_rs(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* A,
                            remove_complex<ValueType> theta,
                            array<bool>& is_strong, array<IndexType>& lambda,
                            array<IndexType>& cf_marker, IndexType& coarse_size)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_coarse_and_compute_prolong_row_ptrs(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& cf_marker, array<IndexType>& coarse_rows,
    array<IndexType>& fine_to_coarse,
    const matrix::Csr<ValueType, IndexType>* A, const array<bool>& is_strong,
    array<IndexType>& row_ptrs)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_FILL_COARSE_AND_COMPUTE_PROLONG_ROW_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_interpolation(std::shared_ptr<const DefaultExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* A,
                           const bool* is_strong,
                           const array<IndexType>& cf_marker,
                           const IndexType* fine_to_coarse,
                           matrix::Csr<ValueType, IndexType>* P)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_INTERPOLATION_KERNEL);

}  // namespace rs
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
