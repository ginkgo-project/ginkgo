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

#include <ginkgo/core/solver/bicg.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicg_kernels.hpp"


namespace gko {
namespace solver {
namespace bicg {
namespace {


GKO_REGISTER_OPERATION(initialize, bicg::initialize);
GKO_REGISTER_OPERATION(step_1, bicg::step_1);
GKO_REGISTER_OPERATION(step_2, bicg::step_2);


}  // anonymous namespace
}  // namespace bicg


template <typename ValueType>
std::unique_ptr<LinOp> Bicg<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Bicg<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


/**
 * @internal
 * (Conjugate-)Transposes the matrix by converting it into a CSR matrix of type
 * CsrType, followed by (conjugate-)transposing.
 *
 * @param mtx  Matrix to (conjugate-)transpose
 * @tparam CsrType  Matrix format in which the matrix mtx is converted into
 *                  before (conjugate-)transposing it
 */
template <typename CsrType>
std::unique_ptr<LinOp> conj_transpose_with_csr(const LinOp* mtx)
{
    auto csr_matrix_unique_ptr = copy_and_convert_to<CsrType>(
        mtx->get_executor(), const_cast<LinOp*>(mtx));

    csr_matrix_unique_ptr->set_strategy(
        std::make_shared<typename CsrType::classical>());

    return csr_matrix_unique_ptr->conj_transpose();
}


template <typename ValueType>
void Bicg<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Bicg<ValueType>::apply_dense_impl(const matrix::Dense<ValueType>* dense_b,
                                       matrix::Dense<ValueType>* dense_x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto r = this->create_workspace_with_config_of(0, dense_b);
    auto r2 = this->create_workspace_with_config_of(1, dense_b);
    auto z = this->create_workspace_with_config_of(2, dense_b);
    auto z2 = this->create_workspace_with_config_of(3, dense_b);
    auto p = this->create_workspace_with_config_of(4, dense_b);
    auto p2 = this->create_workspace_with_config_of(5, dense_b);
    auto q = this->create_workspace_with_config_of(6, dense_b);
    auto q2 = this->create_workspace_with_config_of(7, dense_b);

    auto alpha = this->template create_workspace_scalar<ValueType>(
        8, dense_b->get_size()[1]);
    auto beta = this->template create_workspace_scalar<ValueType>(
        9, dense_b->get_size()[1]);
    auto prev_rho = this->template create_workspace_scalar<ValueType>(
        10, dense_b->get_size()[1]);
    auto rho = this->template create_workspace_scalar<ValueType>(
        11, dense_b->get_size()[1]);

    auto one_op = this->template create_workspace_scalar<ValueType>(12, 1);
    auto neg_one_op = this->template create_workspace_scalar<ValueType>(13, 1);
    one_op->fill(one<ValueType>());
    neg_one_op->fill(-one<ValueType>());

    bool one_changed{};
    auto& stop_status = this->template create_workspace_array<stopping_status>(
        0, dense_b->get_size()[1]);
    auto& reduction_tmp = this->template create_workspace_array<char>(1, 0);

    // TODO: replace this with automatic merged kernel generator
    exec->run(bicg::make_initialize(dense_b, r, z, p, q, prev_rho, rho, r2, z2,
                                    p2, q2, &stop_status));
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    // r = r2 = dense_b
    // z2 = p2 = q2 = 0

    std::unique_ptr<LinOp> conj_trans_A;
    auto conj_transposable_system_matrix =
        dynamic_cast<const Transposable*>(this->get_system_matrix().get());

    if (conj_transposable_system_matrix) {
        conj_trans_A = conj_transposable_system_matrix->conj_transpose();
    } else {
        // TODO Extend when adding more IndexTypes
        // Try to figure out the IndexType that can be used for the CSR matrix
        using Csr32 = matrix::Csr<ValueType, int32>;
        using Csr64 = matrix::Csr<ValueType, int64>;
        auto supports_int64 = dynamic_cast<const ConvertibleTo<Csr64>*>(
            this->get_system_matrix().get());
        if (supports_int64) {
            conj_trans_A =
                conj_transpose_with_csr<Csr64>(this->get_system_matrix().get());
        } else {
            conj_trans_A =
                conj_transpose_with_csr<Csr32>(this->get_system_matrix().get());
        }
    }

    auto conj_trans_preconditioner =
        as<const Transposable>(this->get_preconditioner())->conj_transpose();

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    // r = r - Ax =  -1.0 * A*dense_x + 1.0*r
    r2->copy_from(r);
    // r2 = r
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);

    int iter = -1;

    /* Memory movement summary:
     * 28n * values + matrix/preconditioner storage + conj storage
     * 2x SpMV:                4n * values + storage + conj storage
     * 2x Preconditioner:      4n * values + storage + conj storage
     * 2x dot                  4n
     * 1x step 1 (axpys)       6n
     * 1x step 2 (axpys)       9n
     * 1x norm2 residual        n
     */
    while (true) {
        this->get_preconditioner()->apply(r, z);
        conj_trans_preconditioner->apply(r2, z2);
        z->compute_conj_dot(r2, rho, reduction_tmp);

        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r, dense_x, nullptr, rho);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho
        // p = z + tmp * p
        // p2 = z2 + tmp * p2
        exec->run(bicg::make_step_1(p, z, p2, z2, rho, prev_rho, &stop_status));
        this->get_system_matrix()->apply(p, q);
        conj_trans_A->apply(p2, q2);
        p2->compute_conj_dot(q, beta, reduction_tmp);
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        // r2 = r2 - tmp * q2
        exec->run(bicg::make_step_2(dense_x, r, r2, p, q, q2, beta, rho,
                                    &stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                 const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_BICG(_type) class Bicg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG);


}  // namespace solver
}  // namespace gko
