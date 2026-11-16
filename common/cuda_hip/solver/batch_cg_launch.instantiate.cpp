// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/solver/batch_cg_launch.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/solver/batch_cg_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_cg {


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_cg::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const device_type<ValueType>* const __restrict__ b_values,
    device_type<ValueType>* const __restrict__ x_values,
    device_type<ValueType>* const __restrict__ workspace_data,
    const int& block_size, const size_t& shared_size)
{
    // the solver computes in the arithmetic type of the values
    const auto kernel = batch_single_kernels::get_apply_kernel<
        StopType, n_shared, prec_shared, PrecType, LogType, BatchMatrixType,
        device_type<ValueType>>();
    kernel<<<mat.num_batch_items, block_size, shared_size,
             exec->get_stream()>>>(
        sconf, settings.max_iterations,
        arithmetic<ValueType>(settings.residual_tol), logger, prec,
        arithmetic<ValueType>(mat),
        arithmetic<ValueType>(b_values, mat.num_batch_items * mat.num_rows),
        arithmetic<ValueType>(x_values, mat.num_batch_items * mat.num_rows),
        workspace_data);
}


// begin
GKO_INSTANTIATE_BATCH_CG_LAUNCH_0_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_1_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_2_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_3_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_4_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_5_FALSE;
// split
GKO_INSTANTIATE_BATCH_CG_LAUNCH_5_TRUE;
// end


}  // namespace batch_cg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
