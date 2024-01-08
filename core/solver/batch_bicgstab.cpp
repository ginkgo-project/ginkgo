// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"


namespace gko {
namespace batch {
namespace solver {
namespace bicgstab {


GKO_REGISTER_OPERATION(apply, batch_bicgstab::apply);


}  // namespace bicgstab


template <typename ValueType>
void Bicgstab<ValueType>::solver_apply(
    const MultiVector<ValueType>* b, MultiVector<ValueType>* x,
    log::detail::log_data<remove_complex<ValueType>>* log_data) const
{
    using MVec = MultiVector<ValueType>;
    const kernels::batch_bicgstab::settings<remove_complex<ValueType>> settings{
        this->max_iterations_, static_cast<real_type>(this->residual_tol_),
        parameters_.tolerance_type};
    auto exec = this->get_executor();
    exec->run(bicgstab::make_apply(settings, this->system_matrix_.get(),
                                   this->preconditioner_.get(), b, x,
                                   *log_data));
}


#define GKO_DECLARE_BATCH_BICGSTAB(_type) class Bicgstab<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB);


}  // namespace solver
}  // namespace batch
}  // namespace gko
