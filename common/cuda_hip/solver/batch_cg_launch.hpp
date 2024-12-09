// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_cg {


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


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
    const int& block_size, const size_t& shared_size);

#define GKO_DECLARE_BATCH_CG_LAUNCH(_vtype, _n_shared, _prec_shared, mat_t, \
                                    log_t, pre_t, stop_t)                   \
    void launch_apply_kernel<_vtype, _n_shared, _prec_shared,               \
                             stop_t<device_type<_vtype>>>(                  \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const gko::kernels::batch_cg::storage_config& sconf,                \
        const settings<remove_complex<_vtype>>& settings,                   \
        log_t<gko::remove_complex<device_type<_vtype>>>& logger,            \
        pre_t<device_type<_vtype>>& prec,                                   \
        const mat_t<const device_type<_vtype>>& mat,                        \
        const device_type<_vtype>* const __restrict__ b_values,             \
        device_type<_vtype>* const __restrict__ x_values,                   \
        device_type<_vtype>* const __restrict__ workspace_data,             \
        const int& block_size, const size_t& shared_size)

#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_0_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 0, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_1_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 1, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_2_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 2, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_3_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 3, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_4_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 4, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_5_FALSE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 5, false)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_5_TRUE \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 5, true)


}  // namespace batch_cg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
