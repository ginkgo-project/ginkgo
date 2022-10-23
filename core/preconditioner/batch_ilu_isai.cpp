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

#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>
#include <ginkgo/core/preconditioner/batch_isai.hpp>

#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_ilu_isai_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace batch_ilu_isai {
namespace {}  // namespace
}  // namespace batch_ilu_isai


template <typename ValueType, typename IndexType>
void BatchIluIsai<ValueType, IndexType>::generate_precond(
    const BatchLinOp* const system_matrix)
{
    // generate entire batch of factorizations
    if (!system_matrix->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }

    auto exec = this->get_executor();

    auto batch_ilu_factory =
        gko::preconditioner::BatchIlu<ValueType, IndexType>::build()
            .with_skip_sorting(this->parameters_.skip_sorting)
            .with_ilu_type(this->parameters_.ilu_type)
            .with_parilu_num_sweeps(this->parameters_.parilu_num_sweeps)
            .on(exec);

    // When this smart pointer is assigned to another shared ptr, is the deletor
    // also copied??
    std::shared_ptr<const BatchLinOp> sys_smart_ptr(
        system_matrix, [](const BatchLinOp* plain_ptr) {});

    auto batch_ilu_precond = batch_ilu_factory->generate(sys_smart_ptr);

    std::pair<std::shared_ptr<const matrix_type>,
              std::shared_ptr<const matrix_type>>
        l_and_u_factors =
            batch_ilu_precond->generate_split_factors_from_factored_matrix();

    std::shared_ptr<const matrix_type> l_factor = l_and_u_factors.first;
    std::shared_ptr<const matrix_type> u_factor = l_and_u_factors.second;

    auto lower_factor_isai_precond =
        gko::preconditioner::BatchIsai<ValueType, IndexType>::build()
            .with_skip_sorting(true)
            .with_isai_input_matrix_type(
                batch_isai_input_matrix_type::lower_tri)
            .with_sparsity_power(
                this->parameters_.lower_factor_isai_sparsity_power)
            .on(exec)
            ->generate(l_factor);

    auto upper_factor_isai_precond =
        gko::preconditioner::BatchIsai<ValueType, IndexType>::build()
            .with_skip_sorting(true)
            .with_isai_input_matrix_type(
                batch_isai_input_matrix_type::upper_tri)
            .with_sparsity_power(
                this->parameters_.upper_factor_isai_sparsity_power)
            .on(exec)
            ->generate(u_factor);

    this->lower_factor_isai_ =
        lower_factor_isai_precond->get_const_approximate_inverse();
    this->upper_factor_isai_ =
        upper_factor_isai_precond->get_const_approximate_inverse();

    if (this->parameters_.apply_type ==
        batch_ilu_isai_apply::inv_factors_spgemm) {
        GKO_NOT_IMPLEMENTED;

        // z = precond * r
        // L * U * z = r
        // lai_L * L * U * z = lai_L * r
        // U * z = lai_L * r
        // lai_U * U * z = lai_U * lai_L * r
        // z = lai_U * lai_L * r
        // z = mult_inv * r

        // Therefore, mult_inv = lai_U * lai_L
        this->mult_inv_ = gko::share(matrix_type::create(exec));
        // mult_inv_ : memory allocation? to store solution (u_inv * l_inv)
        this->upper_factor_isai_->apply(this->lower_factor_isai_.get(),
                                        this->mult_inv_.get());
    }
}


#define GKO_DECLARE_BATCH_ILU_ISAI(ValueType) \
    class BatchIluIsai<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ILU_ISAI);


}  // namespace preconditioner
}  // namespace gko
