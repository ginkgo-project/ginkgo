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

#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/batch_isai.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_ilu_isai_kernels.hpp"

namespace gko {
namespace preconditioner {


template <typename ValueType, typename IndexType>
void BatchIluIsai<ValueType, IndexType>::generate_precond()
{
    // generate entire batch of preconditioners
    if (!this->system_matrix_->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }

    auto exec = this->get_executor();

    auto batch_ilu_factory =
        gko::preconditioner::BatchIlu<ValueType, IndexType>::build()
            .with_skip_sorting(this->parameters_.skip_sorting)
            .with_ilu_type(this->parameters_.ilu_type)
            .with_parilu_num_sweeps(this->parameters_.parilu_num_sweeps)
            .on(exec);

    auto batch_ilu_precond = batch_ilu_factory->generate(this->system_matrix_);

    auto l_and_u_factors =
        batch_ilu_precond->generate_split_factors_from_factored_matrix();

    auto l_factor = l_and_u_factors.first;
    auto u_factor = l_and_u_factors.second;
    this->lower_factor_ = l_factor;
    this->upper_factor_ = u_factor;

    // Note: The generated l_factor and u_factor are always sorted irrespective
    // of whether the input matrix is sorted or unsorted.
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

    // Note: mult_inv and iteration matrices etc. are created/computed outside
    // the solver kernel (in precond. external generate)
    // since they require SpGemm (memory allocation to store the result is not
    // straightforward)

    if (this->parameters_.apply_type ==
        batch_ilu_isai_apply::spmv_isai_with_spgemm) {
        // z = precond * r
        // L * U * z = r
        // lai_L * L * U * z = lai_L * r
        // U * z = lai_L * r
        // lai_U * U * z = lai_U * lai_L * r
        // z = lai_U * lai_L * r
        // z = mult_inv * r
        // Therefore, mult_inv = lai_U * lai_L
        this->mult_inv_ = gko::share(
            matrix_type::create(exec, this->system_matrix_->get_size()));
        this->upper_factor_isai_->apply(this->lower_factor_isai_.get(),
                                        this->mult_inv_.get());
    } else if (this->parameters_.apply_type ==
               batch_ilu_isai_apply::relaxation_steps_isai_with_spgemm) {
        // compute iteration matrices and store in the member variables of
        // BatchIluIsai object  ( has shared_ptrs for iter. matrices )
        // z = precond * r
        // L * U * z = r
        // L * y = r  and then U * z = y
        // y_updated = lai_L * r + (I - lai_L * L) * y_old    (iterate)
        // Once y is obtained, z_updated = lai_U * y + (I - lai_U * U) * z_old
        // (iterate) Therefore, iter_mat_lower_solve = I - lai_L * L   and
        // iter_mat_upper_solve = I - lai_U * U

        using vec_type = gko::matrix::BatchDense<ValueType>;
        auto one = gko::batch_initialize<vec_type>(
            this->system_matrix_->get_num_batch_entries(), {1.0}, exec);
        auto neg_one = gko::batch_initialize<vec_type>(
            this->system_matrix_->get_num_batch_entries(), {-1.0}, exec);

        this->iter_mat_lower_solve_ = gko::share(
            matrix_type::create(exec, this->system_matrix_->get_size()));
        this->lower_factor_isai_->apply(this->lower_factor_.get(),
                                        this->iter_mat_lower_solve_.get());
        this->iter_mat_lower_solve_->add_scaled_identity(
            one.get(),
            neg_one.get());  //  M <- a I + b M, thus a = one, b = neg_one

        this->iter_mat_upper_solve_ = gko::share(
            matrix_type::create(exec, this->system_matrix_->get_size()));
        this->upper_factor_isai_->apply(this->upper_factor_.get(),
                                        this->iter_mat_upper_solve_.get());
        this->iter_mat_upper_solve_->add_scaled_identity(one.get(),
                                                         neg_one.get());

        // Another way:
        // iter_mat = identity  (But the BatchIdentity class does not yet have
        // the kernel: convert_to BatchCsr) lai_L->apply(neg_one, L, one,
        // iter_mat)
    }
}


#define GKO_DECLARE_BATCH_ILU_ISAI(ValueType) \
    class BatchIluIsai<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ILU_ISAI);


}  // namespace preconditioner
}  // namespace gko
