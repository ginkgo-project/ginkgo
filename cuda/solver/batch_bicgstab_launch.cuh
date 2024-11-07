// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_SOLVER_BATCH_BICGSTAB_LAUNCH_CUH_
#define GKO_CUDA_SOLVER_BATCH_BICGSTAB_LAUNCH_CUH_


#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_bicgstab_launch.hpp"
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


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif
