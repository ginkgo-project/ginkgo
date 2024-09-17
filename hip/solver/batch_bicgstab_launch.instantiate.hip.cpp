// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/solver/batch_bicgstab_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "hip/solver/batch_bicgstab_launch.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_bicgstab {


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const hip_type<ValueType>* const __restrict__ b_values,
    hip_type<ValueType>* const __restrict__ x_values,
    hip_type<ValueType>* const __restrict__ workspace_data,
    const int& block_size, const size_t& shared_size)
{
    batch_single_kernels::apply_kernel<StopType, n_shared, prec_shared>
        <<<mat.num_batch_items, block_size, shared_size, exec->get_stream()>>>(
            sconf, settings.max_iterations, as_hip_type(settings.residual_tol),
            logger, prec, mat, b_values, x_values, workspace_data);
}


// begin
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_0_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_1_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_2_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_3_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_4_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_5_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_6_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_7_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_8_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_FALSE);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_TRUE);
// end


}  // namespace batch_bicgstab
}  // namespace hip
}  // namespace kernels
}  // namespace gko
