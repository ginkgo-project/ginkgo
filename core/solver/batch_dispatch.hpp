/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
class DummyKernelCaller {
public:
    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& mat,
                     const multi_vector::uniform_batch<DValueType>& b,
                     const multi_vector::uniform_batch<DValueType>& x) const
    {}
};


/**
 * Handles dispatching to the correct instantiation of a batched solver
 * depending on runtime parameters.
 *
 * @tparam KernelCaller  Class with an interface like DummyKernelCaller,
 *   that is responsible for finally calling the templated backend-specific
 *   kernel.
 * @tparam SettingsType  Structure type of options for the particular solver to
 * be used.
 * @tparam ValueType  The user-facing value type.
 */
template <typename KernelCaller, typename SettingsType, typename ValueType>
class BatchSolverDispatch {
public:
    using value_type = ValueType;
    using device_value_type = DeviceValueType<ValueType>;
    using real_type = remove_complex<value_type>;

    BatchSolverDispatch(const KernelCaller& kernel_caller,
                        const SettingsType& settings,
                        const BatchLinOp* const matrix,
                        const BatchLinOp* const preconditioner,
                        const log::BatchLogType logger_type =
                            log::BatchLogType::simple_convergence_completion)
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
        if (settings_.tol_type == stop::ToleranceType::absolute) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::stop::SimpleAbsResidual<device_value_type>, LogType>(
                logger, mat_item, precond, b_item, x_item);
        } else if (settings_.tol_type == stop::ToleranceType::relative) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::stop::SimpleRelResidual<device_value_type>, LogType>(
                logger, mat_item, precond, b_item, x_item);
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
            dispatch_on_stop<device::Identity<device_value_type>>(
                logger, mat_item, device::Identity<device_value_type>(), b_item,
                x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType>
    void dispatch_on_logger(
        const BatchMatrixType& amat,
        const multi_vector::uniform_batch<const device_value_type>& b_item,
        const multi_vector::uniform_batch<device_value_type>& x_item,
        log::BatchLogData<real_type>& log_data)
    {
        if (logger_type_ == log::BatchLogType::simple_convergence_completion) {
            device::batch_log::SimpleFinalLogger<real_type> logger(
                log_data.res_norms.get_data(), log_data.iter_counts.get_data());
            dispatch_on_preconditioner(logger, amat, b_item, x_item);
        } else {
            GKO_NOT_IMPLEMENTED;
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
               log::BatchLogData<real_type>& log_data)
    {
        const auto x_item = device::get_batch_struct(x);
        const auto b_item = device::get_batch_struct(b);

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
        } else {
            GKO_NOT_SUPPORTED(mat_);
        }
    }

private:
    const KernelCaller caller_;
    const SettingsType settings_;
    const BatchLinOp* mat_;
    const BatchLinOp* precond_;
    const log::BatchLogType logger_type_;
};


/**
 * Convenient function to create a dispatcher. Infers most template arguments.
 */
template <typename ValueType, typename KernelCaller, typename SettingsType>
BatchSolverDispatch<KernelCaller, SettingsType, ValueType> create_dispatcher(
    const KernelCaller& kernel_caller, const SettingsType& settings,
    const BatchLinOp* const matrix, const BatchLinOp* const preconditioner,
    const log::BatchLogType logger_type =
        log::BatchLogType::simple_convergence_completion)
{
    return BatchSolverDispatch<KernelCaller, SettingsType, ValueType>(
        kernel_caller, settings, matrix, preconditioner, logger_type);
}


}  // namespace solver
}  // namespace batch
}  // namespace gko

#endif  // GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
