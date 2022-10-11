/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/batch_isai.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/log/batch_logging.hpp"


#if defined GKO_COMPILING_CUDA

#include "cuda/components/cooperative_groups.cuh"
#include "cuda/log/batch_loggers.cuh"
#include "cuda/matrix/batch_struct.hpp"
#include "cuda/preconditioner/batch_preconditioners.cuh"
#include "cuda/stop/batch_stop.cuh"

namespace gko {
namespace batch_solver {

namespace device = gko::kernels::cuda;

template <typename ValueType>
using DeviceValueType = typename gko::kernels::cuda::cuda_type<ValueType>;

}  // namespace batch_solver
}  // namespace gko

#elif defined GKO_COMPILING_HIP

#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/log/batch_loggers.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"
#include "hip/preconditioner/batch_preconditioners.hip.hpp"
#include "hip/stop/batch_stop.hip.hpp"

namespace gko {
namespace batch_solver {

namespace device = gko::kernels::hip;

template <typename ValueType>
using DeviceValueType = gko::kernels::hip::hip_type<ValueType>;

}  // namespace batch_solver
}  // namespace gko

#elif defined GKO_COMPILING_DPCPP

#error "Batch solvers are not yet supported on DPC++!"

namespace gko {

namespace device = gko::kernels::dpcpp;

namespace batch_solver {

template <typename ValueType>
using DeviceValueType = ValueType;

}
}  // namespace gko

#else

#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_ilu.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/preconditioner/batch_trsv.hpp"
#include "reference/stop/batch_criteria.hpp"

namespace gko {

namespace device = gko::kernels::host;

namespace batch_solver {

template <typename ValueType>
using DeviceValueType = ValueType;

}
}  // namespace gko

#endif

namespace gko {
namespace batch_solver {

template <typename DevValueType>
class DummyKernelCaller {
public:
    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& a,
        const gko::batch_dense::UniformBatch<DevValueType>& b,
        const gko::batch_dense::UniformBatch<DevValueType>& x) const
    {}
};


/**
 * Handles dispatching to the correct instantiation of a batched solver
 * depending on runtime parameters.
 *
 * @tparam KernelCaller  Class with an interface like DummyKernelCaller,
 *   that is reponsible for finally calling the templated backend-specific
 *   kernel.
 * @tparam OptsType  Structure type of options for the particular solver to be
 *   used.
 * @tparam ValueType  The user-facing value type.
 * @tparam DevValueType  The backend-specific value type corresponding to
 *   ValueType.
 */
template <typename KernelCaller, typename OptsType, typename ValueType>
class BatchSolverDispatch {
public:
    using value_type = ValueType;
    using device_value_type = DeviceValueType<ValueType>;

    BatchSolverDispatch(
        const KernelCaller& kernel_caller, const OptsType& opts,
        const BatchLinOp* const matrix, const BatchLinOp* const preconditioner,
        const gko::log::BatchLogType logger_type =
            gko::log::BatchLogType::simple_convergence_completion)
        : caller_{kernel_caller},
          opts_{opts},
          a_{matrix},
          precon_{preconditioner},
          logger_type_{logger_type}
    {}

    template <typename PrecType, typename BatchMatrixType, typename LogType>
    void dispatch_on_stop(
        const LogType& logger, const BatchMatrixType& amat, PrecType prec,
        const gko::batch_dense::UniformBatch<const device_value_type>& b_b,
        const gko::batch_dense::UniformBatch<device_value_type>& x_b)
    {
        if (opts_.tol_type == gko::stop::batch::ToleranceType::absolute) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::stop::SimpleAbsResidual<device_value_type>, LogType>(
                logger, amat, prec, b_b, x_b);
        } else if (opts_.tol_type ==
                   gko::stop::batch::ToleranceType::relative) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                device::stop::SimpleRelResidual<device_value_type>, LogType>(
                logger, amat, prec, b_b, x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType, typename LogType>
    void dispatch_on_preconditioner(
        const LogType& logger, const BatchMatrixType& amat,
        const gko::batch_dense::UniformBatch<const device_value_type>& b_b,
        const gko::batch_dense::UniformBatch<device_value_type>& x_b)
    {
        if (!precon_ ||
            dynamic_cast<const matrix::BatchIdentity<value_type>*>(precon_)) {
            dispatch_on_stop<device::BatchIdentity<device_value_type>>(
                logger, amat, device::BatchIdentity<device_value_type>(), b_b,
                x_b);
        } else if (auto precjac = dynamic_cast<
                       const preconditioner::BatchJacobi<value_type>*>(
                       precon_)) {
            dispatch_on_stop<device::BatchJacobi<device_value_type>>(
                logger, amat, device::BatchJacobi<device_value_type>(), b_b,
                x_b);
        } else if (auto prec = dynamic_cast<
                       const preconditioner::BatchIlu<value_type>*>(precon_)) {
            auto l_factor =
                device::get_batch_struct(prec->get_const_lower_factor());
            auto u_factor =
                device::get_batch_struct(prec->get_const_upper_factor());
            if (prec->get_parameters().trsv_type ==
                gko::preconditioner::batch_trsv_type::exact) {
                using trsv_type =
                    device::batch_exact_trsv_split<device_value_type>;
                // assuming split factors, or we need one more branch here
                using ilu_type =
                    device::batch_ilu_split<device_value_type, trsv_type>;
                dispatch_on_stop(logger, amat,
                                 ilu_type{l_factor, u_factor, trsv_type()}, b_b,
                                 x_b);
            } else {
                // TODO: Implement other batch TRSV types
                GKO_NOT_IMPLEMENTED;
            }
        } else if (auto prec = dynamic_cast<
                       const preconditioner::BatchIsai<value_type>*>(precon_)) {
            auto approx_inv =
                device::get_batch_struct(prec->get_const_approximate_inverse());
            // TODO: Define device preconditioners, add the includes to files
            //  like cuda/preconditioner/batch_preconditioners.cuh, and add a
            //  dispatch.
            GKO_NOT_IMPLEMENTED;
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType>
    void dispatch_on_logger(
        const BatchMatrixType& amat,
        const gko::batch_dense::UniformBatch<const device_value_type>& b_b,
        const gko::batch_dense::UniformBatch<device_value_type>& x_b,
        gko::log::BatchLogData<ValueType>& logdata)
    {
        if (logger_type_ == log::BatchLogType::simple_convergence_completion) {
            device::batch_log::SimpleFinalLogger<
                remove_complex<device_value_type>>
                logger(logdata.res_norms->get_values(),
                       logdata.iter_counts.get_data());
            dispatch_on_preconditioner(logger, amat, b_b, x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    /**
     * Solves a linear system from the given data and kernel caller.
     *
     * Note: The correct backend-specific get_batch_struct function needs to be
     * available in the current scope.
     */
    void apply(const matrix::BatchDense<ValueType>* const b,
               matrix::BatchDense<ValueType>* const x,
               log::BatchLogData<ValueType>& logdata)
    {
        const auto x_b = device::get_batch_struct(x);
        const auto b_b = device::get_batch_struct(b);

        if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a_)) {
            auto m_b = device::get_batch_struct(amat);
            dispatch_on_logger(m_b, b_b, x_b, logdata);
        } else if (auto amat =
                       dynamic_cast<const matrix::BatchEll<ValueType>*>(a_)) {
            auto m_b = device::get_batch_struct(amat);
            dispatch_on_logger(m_b, b_b, x_b, logdata);
        } else if (auto amat =
                       dynamic_cast<const matrix::BatchDense<ValueType>*>(a_)) {
            auto m_b = device::get_batch_struct(amat);
            dispatch_on_logger(m_b, b_b, x_b, logdata);
        } else {
            GKO_NOT_SUPPORTED(a_);
        }
    }

private:
    const KernelCaller caller_;
    const OptsType opts_;
    const BatchLinOp* a_;
    const BatchLinOp* precon_;
    const log::BatchLogType logger_type_;
};


/**
 * Conventient function to create a dispatcher. Infers most template arguments.
 */
template <typename ValueType, typename KernelCaller, typename OptsType>
BatchSolverDispatch<KernelCaller, OptsType, ValueType> create_dispatcher(
    const KernelCaller& kernel_caller, const OptsType& opts,
    const BatchLinOp* const a, const BatchLinOp* const preconditioner,
    const log::BatchLogType logger_type =
        log::BatchLogType::simple_convergence_completion)
{
    return BatchSolverDispatch<KernelCaller, OptsType, ValueType>(
        kernel_caller, opts, a, preconditioner, logger_type);
}


}  // namespace batch_solver
}  // namespace gko

#endif  // GKO_CORE_SOLVER_BATCH_DISPATCH_HPP_
