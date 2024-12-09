// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
#define GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>

#include "core/base/batch_instantiation.hpp"
#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


#if defined GKO_COMPILING_CUDA


#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/log/batch_logger.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/preconditioner/batch_preconditioners.hpp"
#include "common/cuda_hip/stop/batch_criteria.hpp"


namespace gko {
namespace batch {
namespace solver {


namespace device = gko::kernels::cuda;


template <typename ValueType>
using DeviceValueType = typename gko::kernels::cuda::cuda_type<ValueType>;


}  // namespace solver
}  // namespace batch
}  // namespace gko


#elif defined GKO_COMPILING_HIP


#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/log/batch_logger.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"
#include "common/cuda_hip/preconditioner/batch_preconditioners.hpp"
#include "common/cuda_hip/stop/batch_criteria.hpp"


namespace gko {
namespace batch {
namespace solver {


namespace device = gko::kernels::hip;


template <typename ValueType>
using DeviceValueType = gko::kernels::hip::hip_type<ValueType>;


}  // namespace solver
}  // namespace batch
}  // namespace gko


#elif defined GKO_COMPILING_DPCPP


#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/log/batch_logger.hpp"
#include "dpcpp/matrix/batch_struct.hpp"
#include "dpcpp/preconditioner/batch_preconditioners.hpp"
#include "dpcpp/stop/batch_criteria.hpp"


namespace gko {
namespace batch {
namespace solver {


namespace device = gko::kernels::dpcpp;


template <typename ValueType>
using DeviceValueType = gko::kernels::dpcpp::sycl_type<ValueType>;


}  // namespace solver
}  // namespace batch
}  // namespace gko


#else


#include "reference/base/batch_struct.hpp"
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_block_jacobi.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_scalar_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace host {


template <typename T>
inline std::decay_t<T> as_device_type(T val)
{
    return val;
}


}  // namespace host
}  // namespace kernels
}  // namespace gko


namespace gko {
namespace batch {
namespace solver {


namespace device = gko::kernels::host;


template <typename ValueType>
using DeviceValueType = ValueType;


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif


namespace gko {
namespace batch {
namespace solver {


template <typename DValueType>
class kernel_caller_interface {
public:
    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& mat,
                     const multi_vector::uniform_batch<DValueType>& b,
                     const multi_vector::uniform_batch<DValueType>& x) const
    {}
};


namespace log {
namespace detail {
/**
 *
 * Types of batch loggers available.
 */
enum class log_type { simple_convergence_completion };


}  // namespace detail
}  // namespace log


#define GKO_BATCH_INSTANTIATE_STOP(_next, ...)                               \
    GKO_INDIRECT(                                                            \
        _next(__VA_ARGS__,                                                   \
              ::gko::batch::solver::device::batch_stop::SimpleAbsResidual)); \
    GKO_INDIRECT(                                                            \
        _next(__VA_ARGS__,                                                   \
              ::gko::batch::solver::device::batch_stop::SimpleRelResidual))

#define GKO_BATCH_INSTANTIATE_DEVICE_PRECONDITIONER(_next, ...)               \
    GKO_INDIRECT(                                                             \
        _next(__VA_ARGS__,                                                    \
              ::gko::batch::solver::device::batch_preconditioner::Identity)); \
    GKO_INDIRECT(_next(                                                       \
        __VA_ARGS__,                                                          \
        ::gko::batch::solver::device::batch_preconditioner::ScalarJacobi));   \
    GKO_INDIRECT(_next(                                                       \
        __VA_ARGS__,                                                          \
        ::gko::batch::solver::device::batch_preconditioner::BlockJacobi))

#define GKO_BATCH_INSTANTIATE_LOGGER(_next, ...) \
    GKO_INDIRECT(                                \
        _next(__VA_ARGS__,                       \
              ::gko::batch::solver::device::batch_log::SimpleFinalLogger))

#define GKO_BATCH_INSTANTIATE_MATRIX_BATCH(_next, ...)                     \
    GKO_INDIRECT(_next(__VA_ARGS__, batch::matrix::ell::uniform_batch));   \
    GKO_INDIRECT(_next(__VA_ARGS__, batch::matrix::dense::uniform_batch)); \
    GKO_INDIRECT(_next(__VA_ARGS__, batch::matrix::csr::uniform_batch))

/**
 * Passes each valid configuration of batch solver template parameter to a
 * macro. The order of template parameters is: macro(..., <matrix>, <logger>,
 * <precond>, <stop>) Any additional macro parameter passed to
 * GKO_BATCH_INSTANTIATE will be prepended to the batch solver template
 * parameters.
 */
#define GKO_BATCH_INSTANTIATE(...)                                             \
    GKO_CALL(GKO_BATCH_INSTANTIATE_MATRIX_BATCH, GKO_BATCH_INSTANTIATE_LOGGER, \
             GKO_BATCH_INSTANTIATE_DEVICE_PRECONDITIONER,                      \
             GKO_BATCH_INSTANTIATE_STOP,                                       \
             GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_VARGS, __VA_ARGS__)


/**
 * Handles dispatching to the correct instantiation of a batched solver
 * depending on runtime parameters.
 *
 * @tparam ValueType  The user-facing value type.
 * @tparam KernelCaller  Class with an interface like kernel_caller_interface,
 *   that is responsible for finally calling the templated backend-specific
 *   kernel.
 * @tparam SettingsType  Structure type of options for the particular solver to
 * be used.
 */
template <typename ValueType, typename KernelCaller, typename SettingsType,
          typename BatchMatrixType, typename PrecType>
class batch_solver_dispatch {
public:
    using value_type = ValueType;
    using device_value_type = DeviceValueType<ValueType>;
    using real_type = remove_complex<value_type>;
    using device_real_type = DeviceValueType<real_type>;

    batch_solver_dispatch(
        const KernelCaller& kernel_caller, const SettingsType& settings,
        const BatchMatrixType* const matrix,
        const PrecType* const preconditioner,
        const log::detail::log_type logger_type =
            log::detail::log_type::simple_convergence_completion)
        : caller_{kernel_caller},
          settings_{settings},
          mat_{matrix},
          precond_{preconditioner},
          logger_type_{logger_type}
    {}

    template <typename PrecEntry, typename BatchMatrixEntry, typename LogType>
    void dispatch_on_stop(
        const LogType& logger, const BatchMatrixEntry& mat_item,
        PrecEntry precond,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item)
    {
        if (settings_.tol_type == stop::tolerance_type::absolute) {
            caller_.template call_kernel<
                BatchMatrixEntry, PrecEntry,
                device::batch_stop::SimpleAbsResidual<device_value_type>,
                LogType>(logger, mat_item, precond, b_item, x_item);
        } else if (settings_.tol_type == stop::tolerance_type::relative) {
            caller_.template call_kernel<
                BatchMatrixEntry, PrecEntry,
                device::batch_stop::SimpleRelResidual<device_value_type>,
                LogType>(logger, mat_item, precond, b_item, x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixEntry, typename LogType>
    void dispatch_on_preconditioner(
        const LogType& logger, const BatchMatrixEntry& mat_item,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item)
    {
        if constexpr (std::is_same_v<PrecType, matrix::Identity<value_type>>) {
            dispatch_on_stop(
                logger, mat_item,
                device::batch_preconditioner::Identity<device_value_type>(),
                b_item, x_item);
        } else if constexpr (std::is_same_v<
                                 PrecType,
                                 batch::preconditioner::Jacobi<value_type>>) {
            const auto max_block_size = precond_->get_max_block_size();
            if (max_block_size == 1) {
                dispatch_on_stop(logger, mat_item,
                                 device::batch_preconditioner::ScalarJacobi<
                                     device_value_type>(),
                                 b_item, x_item);
            } else {
                const auto num_blocks = precond_->get_num_blocks();
                const auto block_ptrs_arr =
                    precond_->get_const_block_pointers();
                const auto row_block_map_arr =
                    precond_->get_const_map_block_to_row();
                const auto blocks_arr =
                    reinterpret_cast<DeviceValueType<const ValueType*>>(
                        precond_->get_const_blocks());
                const auto blocks_cumul_storage =
                    precond_->get_const_blocks_cumulative_offsets();

                dispatch_on_stop(
                    logger, mat_item,
                    device::batch_preconditioner::BlockJacobi<
                        device_value_type>(max_block_size, num_blocks,
                                           blocks_cumul_storage, blocks_arr,
                                           block_ptrs_arr, row_block_map_arr),
                    b_item, x_item);
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixEntry>
    void dispatch_on_logger(
        const BatchMatrixEntry& amat,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item,
        batch::log::detail::log_data<real_type>& log_data)
    {
        if (logger_type_ ==
            log::detail::log_type::simple_convergence_completion) {
            device::batch_log::SimpleFinalLogger<device_real_type> logger(
                device::as_device_type(log_data.res_norms.get_data()),
                log_data.iter_counts.get_data());
            dispatch_on_preconditioner(logger, amat, b_item, x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    void dispatch_on_matrix(
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item,
        batch::log::detail::log_data<real_type>& log_data)
    {
        auto mat_item = device::get_batch_struct(mat_);
        dispatch_on_logger(mat_item, b_item, x_item, log_data);
    }

    /**
     * Solves a linear system from the given data and kernel caller.
     *
     * @note The correct backend-specific get_batch_struct function needs to be
     * available in the current scope.
     */
    void apply(const MultiVector<ValueType>* const b,
               MultiVector<ValueType>* const x,
               batch::log::detail::log_data<real_type>& log_data)
    {
        const auto x_item = device::get_batch_struct(x);
        const auto b_item = device::get_batch_struct(b);

        dispatch_on_matrix(b_item, x_item, log_data);
    }

private:
    const KernelCaller caller_;
    const SettingsType settings_;
    const BatchMatrixType* mat_;
    const PrecType* precond_;
    const log::detail::log_type logger_type_;
};


/**
 * Convenient function to create a dispatcher. Infers most template arguments.
 */
template <typename ValueType, typename KernelCaller, typename SettingsType,
          typename BatchMatrixType, typename PrecType>
batch_solver_dispatch<ValueType, KernelCaller, SettingsType, BatchMatrixType,
                      PrecType>
create_dispatcher(const KernelCaller& kernel_caller,
                  const SettingsType& settings,
                  const BatchMatrixType* const matrix,
                  const PrecType* const preconditioner,
                  const log::detail::log_type logger_type =
                      log::detail::log_type::simple_convergence_completion)
{
    return batch_solver_dispatch<ValueType, KernelCaller, SettingsType,
                                 BatchMatrixType, PrecType>(
        kernel_caller, settings, matrix, preconditioner, logger_type);
}


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
