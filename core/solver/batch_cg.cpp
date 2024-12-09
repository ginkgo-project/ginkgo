// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/batch_cg.hpp"

#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>

#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/solver/batch_cg_kernels.hpp"

namespace gko {
namespace batch {
namespace solver {
namespace cg {


GKO_REGISTER_OPERATION(apply, batch_cg::apply);


}  // namespace cg


template <typename ValueType>
Cg<ValueType>::Cg(std::shared_ptr<const Executor> exec)
    : EnableBatchSolver<Cg, ValueType>(std::move(exec))
{}


template <typename ValueType>
Cg<ValueType>::Cg(const Factory* factory,
                  std::shared_ptr<const BatchLinOp> system_matrix)
    : EnableBatchSolver<Cg, ValueType>(factory->get_executor(),
                                       std::move(system_matrix),
                                       factory->get_parameters()),
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
void Cg<ValueType>::solver_apply(
    const MultiVector<ValueType>* b, MultiVector<ValueType>* x,
    log::detail::log_data<remove_complex<ValueType>>* log_data) const
{
    const kernels::batch_cg::settings<remove_complex<ValueType>> settings{
        this->max_iterations_, static_cast<real_type>(this->residual_tol_),
        parameters_.tolerance_type};
    auto exec = this->get_executor();

    run<batch::matrix::Dense<ValueType>, batch::matrix::Csr<ValueType>,
        batch::matrix::Ell<ValueType>>(
        this->system_matrix_.get(), [&](auto matrix) {
            run<batch::matrix::Identity<ValueType>,
                batch::preconditioner::Jacobi<ValueType>>(
                this->preconditioner_.get(), [&](auto preconditioner) {
                    exec->run(cg::make_apply(settings, matrix, preconditioner,
                                             b, x, *log_data));
                });
        });
}


#define GKO_DECLARE_BATCH_CG(_type) class Cg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG);


}  // namespace solver
}  // namespace batch
}  // namespace gko
