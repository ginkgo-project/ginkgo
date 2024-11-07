// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_SOLVER_BATCH_CG_LAUNCH_CUH_
#define GKO_CUDA_SOLVER_BATCH_CG_LAUNCH_CUH_


#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/solver/batch_cg_launch.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_cg {


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows);

#define GKO_DECLARE_BATCH_CG_GET_NUM_THREADS_PER_BLOCK_(_vtype, mat_t, log_t, \
                                                        pre_t, stop_t)        \
    int get_num_threads_per_block<                                            \
        stop_t<cuda_type<_vtype>>, pre_t<cuda_type<_vtype>>,                  \
        log_t<gko::remove_complex<cuda_type<_vtype>>>,                        \
        mat_t<const cuda_type<_vtype>>, cuda_type<_vtype>>(                   \
        std::shared_ptr<const DefaultExecutor> exec, const int num_rows)

#define GKO_DECLARE_BATCH_CG_GET_NUM_THREADS_PER_BLOCK(_vtype)             \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_GET_NUM_THREADS_PER_BLOCK_, \
                          _vtype)


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec);

#define GKO_DECLARE_BATCH_CG_GET_MAX_DYNAMIC_SHARED_MEMORY_(                \
    _vtype, mat_t, log_t, pre_t, stop_t)                                    \
    int get_max_dynamic_shared_memory<                                      \
        stop_t<cuda_type<_vtype>>, pre_t<cuda_type<_vtype>>,                \
        log_t<gko::remove_complex<_vtype>>, mat_t<const cuda_type<_vtype>>, \
        cuda_type<_vtype>>(std::shared_ptr<const DefaultExecutor> exec)

#define GKO_DECLARE_BATCH_CG_GET_MAX_DYNAMIC_SHARED_MEMORY(_vtype)             \
    GKO_BATCH_INSTANTIATE(GKO_DECLARE_BATCH_CG_GET_MAX_DYNAMIC_SHARED_MEMORY_, \
                          _vtype)


}  // namespace batch_cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif
