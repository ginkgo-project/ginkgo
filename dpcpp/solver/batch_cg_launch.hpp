// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>

#include <ginkgo/core/solver/batch_cg.hpp>

#include "core/base/batch_instantiation.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_cg {


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename ValueType, typename StopType, const int subgroup_size,
          const int n_shared_total, typename PrecType, typename LogType,
          typename BatchMatrixType>
void launch_apply_kernel(std::shared_ptr<const DefaultExecutor> exec,
                         const gko::kernels::batch_cg::storage_config& sconf,
                         const settings<remove_complex<ValueType>>& settings,
                         LogType& logger, PrecType& prec,
                         const BatchMatrixType& mat,
                         const ValueType* const __restrict__ b_values,
                         ValueType* const __restrict__ x_values,
                         ValueType* const __restrict__ workspace,
                         const int& group_size, const int& shared_size);

#define GKO_DECLARE_BATCH_CG_LAUNCH(_vtype, _subgroup_size, _n_shared, mat_t, \
                                    log_t, pre_t, stop_t)                     \
    void                                                                      \
    launch_apply_kernel<_vtype, stop_t<_vtype>, _subgroup_size, _n_shared>(   \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const gko::kernels::batch_cg::storage_config& sconf,                  \
        const settings<remove_complex<_vtype>>& settings,                     \
        log_t<gko::remove_complex<_vtype>>& logger, pre_t<_vtype>& prec,      \
        const mat_t<const _vtype>& mat,                                       \
        const _vtype* const __restrict__ b_values,                            \
        _vtype* const __restrict__ x_values,                                  \
        _vtype* const __restrict__ workspace_data, const int& block_size,     \
        const int& shared_size)

#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_0 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 0)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_1 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 1)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_2 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 2)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_3 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 3)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_4 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 4)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_5 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 5)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_6 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 32, 6)
#define GKO_INSTANTIATE_BATCH_CG_LAUNCH_6_16 \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_LAUNCH, 16, 6)


}  // namespace batch_cg
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
