/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

template <typename DeviceValueType>
class DummyKernelCaller {
public:
    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& a,
        const gko::batch_dense::UniformBatch<DeviceValueType>& b,
        const gko::batch_dense::UniformBatch<DeviceValueType>& x) const
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
 * @tparam DeviceValueType  The backend-specific value type corresponding to
 *   ValueType.
 */
template <typename KernelCaller, typename OptsType, typename ValueType,
          typename DeviceValueType>
class BatchSolverDispatch {
public:
    BatchSolverDispatch(const KernelCaller& kernel_caller, const OptsType& opts,
                        const log::BatchLogType logger_type =
                            log::BatchLogType::simple_convergence_completion)
        : caller_{kernel_caller}, opts_{opts}, logger_type_{logger_type}
    {}

    template <typename PrecType, typename BatchMatrixType, typename LogType>
    void dispatch_on_stop(
        const LogType& logger, const BatchMatrixType& amat,
        const gko::batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const gko::batch_dense::UniformBatch<DeviceValueType>& x_b)
    {
        if (opts_.tol_type == gko::stop::batch::ToleranceType::absolute) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                stop::SimpleAbsResidual<DeviceValueType>, LogType>(logger, amat,
                                                                   b_b, x_b);
        } else if (opts_.tol_type ==
                   gko::stop::batch::ToleranceType::relative) {
            caller_.template call_kernel<
                BatchMatrixType, PrecType,
                stop::SimpleRelResidual<DeviceValueType>, LogType>(logger, amat,
                                                                   b_b, x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename MatrixType, typename LogType>
    void dispatch_on_preconditioner(
        const LogType& logger, const MatrixType* const amat,
        const gko::batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const gko::batch_dense::UniformBatch<DeviceValueType>& x_b)
    {
        auto m_b = get_batch_struct(amat);
        if (opts_.preconditioner == gko::preconditioner::batch::type::none) {
            dispatch_on_stop<BatchIdentity<DeviceValueType>>(logger, m_b, b_b,
                                                             x_b);
        } else if (opts_.preconditioner ==
                   gko::preconditioner::batch::type::jacobi) {
            dispatch_on_stop<BatchJacobi<DeviceValueType>>(logger, m_b, b_b,
                                                           x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename BatchMatrixType>
    void dispatch_on_logger(
        const BatchMatrixType* const amat,
        const gko::batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const gko::batch_dense::UniformBatch<DeviceValueType>& x_b,
        log::BatchLogData<ValueType>& logdata)
    {
        if (logger_type_ == log::BatchLogType::simple_convergence_completion) {
            batch_log::SimpleFinalLogger<remove_complex<DeviceValueType>>
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
    void apply(const BatchLinOp* const a,
               const matrix::BatchDense<ValueType>* const b,
               matrix::BatchDense<ValueType>* const x,
               log::BatchLogData<ValueType>& logdata)
    {
        const auto x_b = get_batch_struct(x);
        const auto b_b = get_batch_struct(b);

        if (auto amat = dynamic_cast<const matrix::BatchCsr<ValueType>*>(a)) {
            dispatch_on_logger(amat, b_b, x_b, logdata);
        } else if (auto amat =
                       dynamic_cast<const matrix::BatchEll<ValueType>*>(a)) {
            dispatch_on_logger(amat, b_b, x_b, logdata);
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }

private:
    const KernelCaller caller_;
    const OptsType opts_;
    const log::BatchLogType logger_type_;
};


/**
 * Conventient function to create a dispatcher. Infers most template arguments.
 */
template <typename ValueType, typename DeviceValueType, typename KernelCaller,
          typename OptsType>
BatchSolverDispatch<KernelCaller, OptsType, ValueType, DeviceValueType>
create_dispatcher(const KernelCaller& kernel_caller, const OptsType& opts,
                  const log::BatchLogType logger_type =
                      log::BatchLogType::simple_convergence_completion)
{
    return BatchSolverDispatch<KernelCaller, OptsType, ValueType,
                               DeviceValueType>(kernel_caller, opts,
                                                logger_type);
}
