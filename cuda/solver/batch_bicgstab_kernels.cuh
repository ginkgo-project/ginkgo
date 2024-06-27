// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <thrust/functional.h>
#include <thrust/transform.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "cuda/base/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/kernel_config.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows);

#define GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK_(              \
    _vtype, mat_t, log_t, pre_t, stop_t)                                    \
    int get_num_threads_per_block<stop_t<cuda_type<_vtype>>, pre_t<_vtype>, \
                                  log_t<_vtype>, mat_t<_vtype>, _vtype>(    \
        std::shared_ptr<const DefaultExecutor> exec, const int num_rows)

#define GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK(_vtype) \
    GKO_BATCH_INSTANTIATE(                                           \
        GKO_DECLARE_BATCH_BICGSTAB_GET_NUM_THREADS_PER_BLOCK_, _vtype)


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec);

#define GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY_( \
    _vtype, mat_t, log_t, pre_t, stop_t)                           \
    int get_max_dynamic_shared_memory(                             \
        std::shared_ptr<const DefaultExecutor> exec)

#define GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY(_vtype) \
    GKO_BATCH_INSTANTIATE(                                               \
        GKO_DECLARE_BATCH_BICGSTAB_GET_MAX_DYNAMIC_SHARED_MEMORY_, _vtype)


template <typename T>
using settings = gko::kernels::batch_bicgstab::settings<T>;


template <typename ValueType, int n_shared, bool prec_shared, typename StopType,
          typename PrecType, typename LogType, typename BatchMatrixType>
void launch_apply_kernel(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::kernels::batch_bicgstab::storage_config& sconf,
    const settings<remove_complex<ValueType>>& settings, LogType& logger,
    PrecType& prec, const BatchMatrixType& mat,
    const cuda_type<ValueType>* const __restrict__ b_values,
    cuda_type<ValueType>* const __restrict__ x_values,
    cuda_type<ValueType>* const __restrict__ workspace_data,
    const int& block_size, const size_t& shared_size);

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH(_vtype, _n_shared, _prec_shared, \
                                          mat_t, log_t, pre_t, stop_t)     \
    void launch_apply_kernel<_vtype, _n_shared, _prec_shared,              \
                             stop_t<cuda_type<_vtype>>>(                   \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const gko::kernels::batch_bicgstab::storage_config& sconf,         \
        const settings<remove_complex<_vtype>>& settings,                  \
        log_t<cuda_type<gko::remove_complex<_vtype>>>& logger,             \
        pre_t<cuda_type<_vtype>>& prec,                                    \
        const mat_t<const cuda_type<_vtype>>& mat,                         \
        const cuda_type<_vtype>* const __restrict__ b_values,              \
        cuda_type<_vtype>* const __restrict__ x_values,                    \
        cuda_type<_vtype>* const __restrict__ workspace_data,              \
        const int& block_size, const size_t& shared_size)

#define GKO_DECLARE_BATCH_BICGSTAB_LAUNCH_0_false(_vtype) \
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


template <typename CuValueType>
class kernel_caller {
public:
    using value_type = CuValueType;

    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<value_type>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const value_type>& b,
        const gko::batch::multi_vector::uniform_batch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type num_batch_items = mat.num_batch_items;
        constexpr int align_multiple = 8;
        const int padded_num_rows =
            ceildiv(mat.num_rows, align_multiple) * align_multiple;
        auto shem_guard =
            gko::kernels::cuda::detail::shared_memory_config_guard<
                value_type>();
        const int shmem_per_blk =
            get_max_dynamic_shared_memory<StopType, PrecType, LogType,
                                          BatchMatrixType, value_type>(exec_);
        const int block_size =
            get_num_threads_per_block<StopType, PrecType, LogType,
                                      BatchMatrixType, value_type>(
                exec_, mat.num_rows);
        GKO_ASSERT(block_size >= 2 * config::warp_size);

        const size_t prec_size = PrecType::dynamic_work_size(
            padded_num_rows, mat.get_single_item_num_nnz());
        const auto sconf =
            gko::kernels::batch_bicgstab::compute_shared_storage<PrecType,
                                                                 value_type>(
                shmem_per_blk, padded_num_rows, mat.get_single_item_num_nnz(),
                b.num_rhs);
        const size_t shared_size =
            sconf.n_shared * padded_num_rows * sizeof(value_type) +
            (sconf.prec_shared ? prec_size : 0);
        auto workspace = gko::array<value_type>(
            exec_,
            sconf.gmem_stride_bytes * num_batch_items / sizeof(value_type));
        GKO_ASSERT(sconf.gmem_stride_bytes % sizeof(value_type) == 0);

        value_type* const workspace_data = workspace.get_data();

        // Template parameters launch_apply_kernel<StopType, n_shared,
        // prec_shared>
        if (sconf.prec_shared) {
            launch_apply_kernel<ValueType, 9, true, StopType>(
                exec_, sconf, settings_, logger, prec, mat, b.values, x.values,
                workspace_data, block_size, shared_size);
        } else {
            switch (sconf.n_shared) {
            case 0:
                launch_apply_kernel<ValueType, 0, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 1:
                launch_apply_kernel<ValueType, 1, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 2:
                launch_apply_kernel<ValueType, 2, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 3:
                launch_apply_kernel<ValueType, 3, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 4:
                launch_apply_kernel<ValueType, 4, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 5:
                launch_apply_kernel<ValueType, 5, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 6:
                launch_apply_kernel<ValueType, 6, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 7:
                launch_apply_kernel<ValueType, 7, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 8:
                launch_apply_kernel<ValueType, 8, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            case 9:
                launch_apply_kernel<ValueType, 9, false, StopType>(
                    exec_, sconf, settings_, logger, prec, mat, b.values,
                    x.values, workspace_data, block_size, shared_size);
                break;
            default:
                GKO_NOT_IMPLEMENTED;
            }
        }
    }

private:
    std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<value_type>> settings_;
};


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const batch::BatchLinOp* const mat,
           const batch::BatchLinOp* const precon,
           const batch::MultiVector<ValueType>* const b,
           batch::MultiVector<ValueType>* const x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    using cu_value_type = cuda_type<ValueType>;
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<cu_value_type>(exec, settings), settings, mat, precon);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
