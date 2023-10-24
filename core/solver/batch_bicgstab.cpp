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

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>


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
    log::BatchLogData<remove_complex<ValueType>>* log_data) const
{
    using MVec = MultiVector<ValueType>;
    const kernels::batch_bicgstab::BicgstabSettings<remove_complex<ValueType>>
        settings{this->max_iterations_,
                 static_cast<real_type>(this->residual_tol_),
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
