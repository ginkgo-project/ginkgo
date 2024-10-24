// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_SOLVER_BATCH_CG_LAUNCH_HPP_
#define GKO_HIP_SOLVER_BATCH_CG_LAUNCH_HPP_


#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_cg {


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(std::shared_ptr<const DefaultExecutor> exec,
                         const gko::kernels::batch_cg::storage_config& sconf,
                         const settings<remove_complex<ValueType>>& settings,
                         LogType& logger, PrecType& prec,
                         const BatchMatrixType& mat,
                         const hip_type<ValueType>* const __restrict__ b_values,
                         hip_type<ValueType>* const __restrict__ x_values,
                         hip_type<ValueType>* const __restrict__ workspace_data,
                         const int& block_size, const size_t& shared_size);

#define GKO_DECLARE_BATCH_CG_LAUNCH(_vtype, _n_shared, _prec_shared, mat_t, \
                                    log_t, pre_t, stop_t)                   \
    void launch_apply_kernel<hip_type<_vtype>, _n_shared, _prec_shared,     \
                             stop_t<hip_type<_vtype>>>(                     \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const gko::kernels::batch_cg::storage_config& sconf,                \
        const settings<remove_complex<_vtype>>& settings,                   \
        log_t<hip_type<gko::remove_complex<hip_type<_vtype>>>>& logger,     \
        pre_t<hip_type<_vtype>>& prec,                                      \
        const mat_t<const hip_type<_vtype>>& mat,                           \
        const hip_type<_vtype>* const __restrict__ b_values,                \
        hip_type<_vtype>* const __restrict__ x_values,                      \
        hip_type<_vtype>* const __restrict__ workspace_data,                \
        const int& block_size, const size_t& shared_size)

#define GKO_DECLARE_BATCH_CG_LAUNCH_0_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 0, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_1_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 1, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_2_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 2, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_3_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 3, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_4_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 4, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_5_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 5, false)
#define GKO_DECLARE_BATCH_CG_LAUNCH_5_TRUE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, _vtype, 5, true)


}  // namespace batch_cg
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif
