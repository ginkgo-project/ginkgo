// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/batch_bicgstab.hpp"

#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_external.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>

#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"


namespace gko {
namespace batch {
namespace solver {
namespace bicgstab {


GKO_REGISTER_OPERATION(apply, batch_bicgstab::apply);


}  // namespace bicgstab


template <typename ValueType>
Bicgstab<ValueType>::Bicgstab(std::shared_ptr<const Executor> exec)
    : EnableBatchSolver<Bicgstab, ValueType>(std::move(exec))
{}


template <typename ValueType>
Bicgstab<ValueType>::Bicgstab(const Factory* factory,
                              std::shared_ptr<const BatchLinOp> system_matrix)
    : EnableBatchSolver<Bicgstab, ValueType>(factory->get_executor(),
                                             std::move(system_matrix),
                                             factory->get_parameters()),
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
void Bicgstab<ValueType>::solver_apply(
    const MultiVector<ValueType>* b, MultiVector<ValueType>* x,
    log::detail::log_data<remove_complex<ValueType>>* log_data) const
{
    const kernels::batch_bicgstab::settings<remove_complex<ValueType>> settings{
        this->max_iterations_, static_cast<real_type>(this->residual_tol_),
        parameters_.tolerance_type};
    auto exec = this->get_executor();

    run<matrix::Dense<ValueType>, matrix::Csr<ValueType>,
        matrix::Ell<ValueType>, matrix::External<ValueType>>(
        this->system_matrix_.get(), [&](auto matrix) {
            run<matrix::Identity<ValueType>, preconditioner::Jacobi<ValueType>>(
                this->preconditioner_.get(), [&](auto preconditioner) {
                    exec->run(bicgstab::make_apply(
                        settings, matrix, preconditioner, b, x, *log_data));
                });
        });
}


#define GKO_DECLARE_BATCH_BICGSTAB(_type) class Bicgstab<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB);


}  // namespace solver
}  // namespace batch
}  // namespace gko
