// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
#define GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


#if defined GKO_COMPILING_CUDA


#include "cuda/base/batch_struct.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/log/batch_logger.cuh"
#include "cuda/matrix/batch_struct.hpp"
#include "cuda/preconditioner/batch_preconditioners.cuh"
#include "cuda/stop/batch_criteria.cuh"


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


#include "hip/base/batch_struct.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/log/batch_logger.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"
#include "hip/preconditioner/batch_preconditioners.hip.hpp"
#include "hip/stop/batch_criteria.hip.hpp"


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
using DeviceValueType = ValueType;


}  // namespace solver
}  // namespace batch
}  // namespace gko


#else


#include "reference/base/batch_struct.hpp"
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/stop/batch_criteria.hpp"


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
template <typename ValueType, typename KernelCaller, typename SettingsType>
class batch_solver_dispatch {
public:
    using value_type = ValueType;
    using device_value_type = DeviceValueType<ValueType>;
    using real_type = remove_complex<value_type>;

    batch_solver_dispatch(
        const KernelCaller& kernel_caller, const SettingsType& settings,
        const BatchLinOp* const matrix, const BatchLinOp* const preconditioner,
        const log::detail::log_type logger_type =
            log::detail::log_type::simple_convergence_completion)
        : caller_{kernel_caller},
          settings_{settings},
          mat_{matrix},
          precond_{preconditioner},
          logger_type_{logger_type}
    {}

    template <typename PrecType, typename BatchMatrixType, typename LogType>
    void dispatch_on_stop(
        const LogType& logger, const BatchMatrixType& mat_item,
        PrecType precond,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item)
    {
        if (settings_.tol_type == stop::tolerance_type::absolute) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::batch_stop::SimpleAbsResidual<device_value_type>,
                LogType>(logger, mat_item, precond, b_item, x_item);
        } else if (settings_.tol_type == stop::tolerance_type::relative) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::batch_stop::SimpleRelResidual<device_value_type>,
                LogType>(logger, mat_item, precond, b_item, x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType, typename LogType>
    void dispatch_on_preconditioner(
        const LogType& logger, const BatchMatrixType& mat_item,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item)
    {
        if (!precond_ ||
            dynamic_cast<const matrix::Identity<value_type>*>(precond_)) {
            dispatch_on_stop<
                device::batch_preconditioner::Identity<device_value_type>>(
                logger, mat_item,
                device::batch_preconditioner::Identity<device_value_type>(),
                b_item, x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType>
    void dispatch_on_logger(
        const BatchMatrixType& amat,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item,
        batch::log::detail::log_data<real_type>& log_data)
    {
        if (logger_type_ ==
            log::detail::log_type::simple_convergence_completion) {
            device::batch_log::SimpleFinalLogger<real_type> logger(
                log_data.res_norms.get_data(), log_data.iter_counts.get_data());
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
        if (auto batch_mat =
                dynamic_cast<const batch::matrix::Ell<ValueType, int32>*>(
                    mat_)) {
            auto mat_item = device::get_batch_struct(batch_mat);
            dispatch_on_logger(mat_item, b_item, x_item, log_data);
        } else if (auto batch_mat =
                       dynamic_cast<const batch::matrix::Dense<ValueType>*>(
                           mat_)) {
            auto mat_item = device::get_batch_struct(batch_mat);
            dispatch_on_logger(mat_item, b_item, x_item, log_data);
        } else if (auto batch_mat = dynamic_cast<
                       const batch::matrix::Csr<ValueType, int32>*>(mat_)) {
            auto mat_item = device::get_batch_struct(batch_mat);
            dispatch_on_logger(mat_item, b_item, x_item, log_data);
        } else {
            GKO_NOT_SUPPORTED(mat_);
        }
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
    const BatchLinOp* mat_;
    const BatchLinOp* precond_;
    const log::detail::log_type logger_type_;
};


/**
 * Convenient function to create a dispatcher. Infers most template arguments.
 */
template <typename ValueType, typename KernelCaller, typename SettingsType>
batch_solver_dispatch<ValueType, KernelCaller, SettingsType> create_dispatcher(
    const KernelCaller& kernel_caller, const SettingsType& settings,
    const BatchLinOp* const matrix, const BatchLinOp* const preconditioner,
    const log::detail::log_type logger_type =
        log::detail::log_type::simple_convergence_completion)
{
    return batch_solver_dispatch<ValueType, KernelCaller, SettingsType>(
        kernel_caller, settings, matrix, preconditioner, logger_type);
}


}  // namespace solver
}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
