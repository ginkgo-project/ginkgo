// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_bicgstab {


template <typename T>
using settings = gko::kernels::batch_bicgstab::settings<T>;


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows);

#define GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK_(              \
    _vtype, mat_t, log_t, pre_t, stop_t)                                    \
    int get_num_threads_per_block<                                          \
        stop_t<cuda_type<_vtype>>, pre_t<cuda_type<_vtype>>,                \
        log_t<gko::remove_complex<_vtype>>, mat_t<const cuda_type<_vtype>>, \
        cuda_type<_vtype>>(std::shared_ptr<const DefaultExecutor> exec,     \
                           const int num_rows)

#define GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK(_vtype) \
    GKO_BATCH_INSTANTIATE(                                           \
        GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK_, _vtype)


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec);

#define GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY_(          \
    _vtype, mat_t, log_t, pre_t, stop_t)                                    \
    int get_max_dynamic_shared_memory<                                      \
        stop_t<cuda_type<_vtype>>, pre_t<cuda_type<_vtype>>,                \
        log_t<gko::remove_complex<_vtype>>, mat_t<const cuda_type<_vtype>>, \
        cuda_type<_vtype>>(std::shared_ptr<const DefaultExecutor> exec)

#define GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY(_vtype) \
    GKO_BATCH_INSTANTIATE(                                               \
        GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY_, _vtype)


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const ValueType* const __restrict__ b_values,
    ValueType* const __restrict__ x_values,
    ValueType* const __restrict__ workspace_data, const int& block_size,
    const size_t& shared_size);

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH(_vtype, _n_shared, _prec_shared, \
                                          mat_t, log_t, pre_t, stop_t)     \
    void launch_apply_kernel<cuda_type<_vtype>, _n_shared, _prec_shared,   \
                             stop_t<cuda_type<_vtype>>>(                   \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const gko::kernels::batch_bicgstab::storage_config& sconf,         \
        const settings<remove_complex<cuda_type<_vtype>>>& settings,       \
        log_t<gko::remove_complex<cuda_type<_vtype>>>& logger,             \
        pre_t<cuda_type<_vtype>>& prec,                                    \
        const mat_t<const cuda_type<_vtype>>& mat,                         \
        const cuda_type<_vtype>* const __restrict__ b_values,              \
        cuda_type<_vtype>* const __restrict__ x_values,                    \
        cuda_type<_vtype>* const __restrict__ workspace_data,              \
        const int& block_size, const size_t& shared_size)

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_0_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 0, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_1_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 1, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_2_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 2, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_3_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 3, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_4_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 4, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_5_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 5, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_6_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 6, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_7_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 7, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_8_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 8, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_FALSE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 9, false)
#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_9_TRUE(_vtype) \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_BICGSTAB_LAUNCH, _vtype, 9, true)


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
