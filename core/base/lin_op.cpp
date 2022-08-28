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

#include <ginkgo/core/base/lin_op.hpp>


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {


LinOp* LinOp::apply(const LinOp* b, LinOp* x)
{
    this->template log<log::Logger::linop_apply_started>(this, b, x);
    this->validate_application_parameters(b, x);
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b, x);
    return this;
}


const LinOp* LinOp::apply(const LinOp* b, LinOp* x) const
{
    this->template log<log::Logger::linop_apply_started>(this, b, x);
    this->validate_application_parameters(b, x);
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_apply_completed>(this, b, x);
    return this;
}


LinOp* LinOp::apply(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x)
{
    this->template log<log::Logger::linop_advanced_apply_started>(this, alpha,
                                                                  b, beta, x);
    this->validate_application_parameters(alpha, b, beta, x);
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_advanced_apply_completed>(this, alpha,
                                                                    b, beta, x);
    return this;
}


const LinOp* LinOp::apply(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                          LinOp* x) const
{
    this->template log<log::Logger::linop_advanced_apply_started>(this, alpha,
                                                                  b, beta, x);
    this->validate_application_parameters(alpha, b, beta, x);
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    this->template log<log::Logger::linop_advanced_apply_completed>(this, alpha,
                                                                    b, beta, x);
    return this;
}


const dim<2>& LinOp::get_size() const noexcept { return size_; }


bool LinOp::apply_uses_initial_guess() const { return false; }


LinOp& LinOp::operator=(LinOp&& other)
{
    if (this != &other) {
        EnableAbstractPolymorphicObject<LinOp>::operator=(std::move(other));
        this->set_size(other.get_size());
        other.set_size({});
    }
    return *this;
}


LinOp::LinOp(LinOp&& other)
    : EnableAbstractPolymorphicObject<LinOp>(std::move(other)),
      size_{std::exchange(other.size_, dim<2>{})}
{}


LinOp::LinOp(std::shared_ptr<const Executor> exec, const dim<2>& size)
    : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}
{}


void LinOp::set_size(const dim<2>& value) noexcept { size_ = value; }


void LinOp::validate_application_parameters(const LinOp* b,
                                            const LinOp* x) const
{
    GKO_ASSERT_CONFORMANT(this, b);
    GKO_ASSERT_EQUAL_ROWS(this, x);
    GKO_ASSERT_EQUAL_COLS(b, x);
}


void LinOp::validate_application_parameters(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta,
                                            const LinOp* x) const
{
    this->validate_application_parameters(b, x);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
}


std::unique_ptr<LinOp> LinOpFactory::generate(
    std::shared_ptr<const LinOp> input) const
{
    this->template log<log::Logger::linop_factory_generate_started>(
        this, input.get());
    const auto exec = this->get_executor();
    std::unique_ptr<LinOp> generated;
    if (input->get_executor() == exec) {
        generated = this->AbstractFactory::generate(input);
    } else {
        generated = this->AbstractFactory::generate(gko::clone(exec, input));
    }
    this->template log<log::Logger::linop_factory_generate_completed>(
        this, input.get(), generated.get());
    return generated;
}


void ScaledIdentityAddable::add_scaled_identity(const LinOp* const a,
                                                const LinOp* const b)
{
    GKO_ASSERT_IS_SCALAR(a);
    GKO_ASSERT_IS_SCALAR(b);
    auto ae = make_temporary_clone(as<LinOp>(this)->get_executor(), a);
    auto be = make_temporary_clone(as<LinOp>(this)->get_executor(), b);
    add_scaled_identity_impl(ae.get(), be.get());
}


}  // namespace gko
