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
        const gko::batch_dense::UniformBatch<DeviceValueType>& x) const {};
};


/**
 * Handles dispatching to the correct instantiation of a batched solver
 * depending on runtime parameters.
 *
 * Requires the following to be defined before including this class:
 * - A KernelCaller class with an interface like DummyKernelCaller.
 *
 * @tparam ExecType  Executor to run the kernel on
 * @tparam OptsType  Structure type of options for the particular solver to be
 *   used.
 * @tparam ValueType  The user-facing value type.
 * @tparam DeviceValueType  The backend-specific value type corresponding to
 *   ValueType.
 */
template <typename ExecType, typename OptsType, typename ValueType,
          typename DeviceValueType>
class BatchSolverDispatch {
public:
    BatchSolverDispatch(const KernelCaller<DeviceValueType>& kernel_caller,
                        std::shared_ptr<const ExecType> exec,
                        const OptsType& opts,
                        const log::BatchLogType logger_type =
                            log::BatchLogType::simple_convergence_completion)
        : caller_(kernel_caller),
          exec_(exec),
          opts_(opts),
          logger_type_{logger_type}
    {}

    template <typename PrecType, typename BatchMatrixType, typename LogType>
    void dispatch_on_stop(
        const LogType& logger, const BatchMatrixType& amat,
        const batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const batch_dense::UniformBatch<DeviceValueType>& x_b)
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
        const batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const batch_dense::UniformBatch<DeviceValueType>& x_b)
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
        const batch_dense::UniformBatch<const DeviceValueType>& b_b,
        const batch_dense::UniformBatch<DeviceValueType>& x_b,
        log::BatchLogData<ValueType>& logdata)
    {
        if (logger_type_ == log::BatchLogType::simple_convergence_completion) {
            batch_log::SimpleFinalLogger<remove_complex<DeviceValueType>>
                logger(logdata.res_norms->get_values(),
                       logdata.iter_counts.get_data());
            dispatch_on_preconditioner(logger, amat, b_b, x_b);
        } else if (logger_type_ == log::BatchLogType::convergence_completion) {
            batch_log::FinalLogger<remove_complex<DeviceValueType>> logger(
                b_b.num_rhs, opts_.max_its, logdata.res_norms->get_values(),
                logdata.iter_counts.get_data());
            dispatch_on_preconditioner(logger, amat, b_b, x_b);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

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
    const KernelCaller<DeviceValueType>& caller_;
    std::shared_ptr<const ExecType> exec_;
    const OptsType& opts_;
    const log::BatchLogType logger_type_;
};
