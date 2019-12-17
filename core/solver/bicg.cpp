/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicg_kernels.hpp"


namespace gko {
namespace solver {


namespace bicg {


GKO_REGISTER_OPERATION(initialize, bicg::initialize);
GKO_REGISTER_OPERATION(step_1, bicg::step_1);
GKO_REGISTER_OPERATION(step_2, bicg::step_2);


}  // namespace bicg


template <typename ValueType, typename IndexType>
void Bicg<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;  // Damit man swap benutzen kann, obwohl swap nur in einem
                      // anderen namespace definiert ist
    using Vector =
        matrix::Dense<ValueType>;  // Vector bezeichnet nun das selbe wie
                                   // matrix::De..., also eine Abkürzung

    using CsrMatrix = matrix::Csr<ValueType, IndexType>;

    constexpr uint8 RelativeStoppingId{1};  // wird schon zur compilezeit
                                            // berchnet, (aber was macht das
                                            // {1}?)

    auto exec = this->get_executor();  // jeder solver ist ein LinOp und jedes
                                       // LinOp hat seinen eigenen exec...dieser
                                       // wird zurück gegeben

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);  // 1-vektor?
    auto neg_one_op =
        initialize<Vector>({-one<ValueType>()}, exec);  // -1-vektor?

    auto dense_b =
        as<const Vector>(b);  // casted den linop b pointer auf ein dense vector
    auto dense_x = as<Vector>(x);  // x vektor
    auto r = Vector::create_with_config_of(dense_b);
    auto r2 = Vector::create_with_config_of(dense_b);  //
    auto z = Vector::create_with_config_of(dense_b);   //
    auto z2 = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);  //
    auto p2 = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);
    auto q2 = Vector::create_with_config_of(dense_b);  //

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});  //
    auto beta = Vector::create_with_config_of(alpha.get());                //
    auto prev_rho = Vector::create_with_config_of(alpha.get());            //
    auto rho = Vector::create_with_config_of(alpha.get());                 //

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    // initialisiert die Variablen, die man für das cg verfahren benötigt. Ist
    // im jeweiligen Kernel definiert, aber durch initialize und nicht
    // make_initialize?
    //.get übergibt den smartpointer, damit der solver die pointer hält

    exec->run(bicg::make_initialize(
        dense_b, r.get(), z.get(), p.get(), q.get(), prev_rho.get(), rho.get(),
        r2.get(), z2.get(), p2.get(), q2.get(), &stop_status));


    // auto trans_A = gko::LinOp(exec,system_matrix_->get_size() ));//Linop ->
    // Matrix casting? auto trans_P =
    // gko::LinOp(exec,get_preconditioner()->get_size());

    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    // r2 = dense_b
    // z2 = p2 = q2 = 0

    decltype(copy_and_convert_to<CsrMatrix>(
        exec, const_cast<LinOp *>(
                  system_matrix_.get()))) csr_system_matrix_unique_ptr{};


    auto csr_system_matrix_ =
        dynamic_cast<const CsrMatrix *>(system_matrix_.get());
    //  if the cast is not possible, use copy_and_convert to
    if (csr_system_matrix_ == nullptr ||
        csr_system_matrix_->get_executor() != exec) {
        csr_system_matrix_unique_ptr = copy_and_convert_to<CsrMatrix>(
            exec, const_cast<LinOp *>(system_matrix_.get()));

        // matrix is not of CSR type, so we need to convert it
        csr_system_matrix_unique_ptr->set_strategy(
            std::make_shared<typename CsrMatrix::classical>());

        csr_system_matrix_ = csr_system_matrix_unique_ptr.get();
        // csr_system_matrix_unique_ptr = CsrMatrix::create(exec);

        // as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        //    ->convert_to(csr_system_matrix_unique_ptr.get());
        // csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }
    auto trans_A = csr_system_matrix_->transpose();


    // auto trans_A = system_matrix_->transpose();
    // auto trans_P = static_cast<Vector>(get_preconditioner())->transpose();

    system_matrix_->apply(
        neg_one_op.get(), dense_x, one_op.get(),
        r.get());  // system_matrix_ ist im header definiert. Da die Matrix ein
                   // Linop ist, hat sie ein apply
                   // stop kriterium wird durch fabrik generiert, genaueres?
                   // r = r - Ax =  -1.0 * A*dense_x + 1.0*r
    // TODO set r2 = r so there has to be only one apply
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          r2.get());  // r2 = r
    // auto trans = Vector::create(exec,
    // gko::transpose(get_preconditioner->get_size()));//ist vector hier das
    // richtige? auto trans = system_matrix_.Matrix<ValueType>::transpose();
    // trans->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    int iter = -1;  // zählt in der wievielten iteration man sich gerade aufhält
    while (true) {
        // TODO preconditioner transponieren
        get_preconditioner()->apply(
            r.get(), z.get());  // preconditioner berechnen?(berechnet P*r und
                                // speichert in z)
        // TODO transponieren
        get_preconditioner()->apply(
            r2.get(), z2.get());  // preconditioner berechnen?(berechnet P*r2
                                  // und speichert in z2)
        z2->compute_dot(r2.get(), rho.get());  // rho = r * z

        ++iter;  // sind jetzt in nächster iteration
        this->template log<log::Logger::iteration_complete>(this, iter,
                                                            r.get(),  //?
                                                            dense_x);
        // checkt ob man abbrechen sollte
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status,
                       &one_changed)) {  //?
            break;
        }
        // run nimmt eine funktion und führt sie dann demetsprechend auf dem
        // kernel aus?
        exec->run(bicg::make_step_1(p.get(), z.get(), p2.get(), z2.get(),
                                    rho.get(), prev_rho.get(), &stop_status));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        // p2 = z2 + tmp * p2
        system_matrix_->apply(p.get(), q.get());  // q = A*p

        trans_A->apply(p2.get(), q2.get());  // q2 = A*p2

        p2->compute_dot(q.get(), beta.get());  // beta = p2 * q (skalarprodukt)

        exec->run(bicg::make_step_2(dense_x, r.get(), r2.get(), p.get(),
                                    q.get(), q2.get(), beta.get(), rho.get(),
                                    &stop_status));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        // r2 = r2 - tmp * q2
        swap(prev_rho, rho);
    }
}


template <typename ValueType, typename IndexType>
void Bicg<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                            const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_BICG(_type) class Bicg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG);


}  // namespace solver
}  // namespace gko
