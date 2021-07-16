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

#include <ginkgo/core/matrix/fft.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/fft_kernels.hpp"


namespace gko {
namespace matrix {
namespace fft {


GKO_REGISTER_OPERATION(fft, fft::fft);
GKO_REGISTER_OPERATION(fft2, fft::fft2);
GKO_REGISTER_OPERATION(fft3, fft::fft3);


}  // namespace fft


std::unique_ptr<LinOp> Fft::transpose() const
{
    return Fft::create(this->get_executor(), this->get_size()[0], inverse_);
}


std::unique_ptr<LinOp> Fft::conj_transpose() const
{
    return Fft::create(this->get_executor(), this->get_size()[0], !inverse_);
}


bool Fft::is_inverse() const { return inverse_; }


void Fft::apply_impl(const LinOp *b, LinOp *x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>> *>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft(float_b, dense_x, inverse_, buffer_));
    } else {
        auto dense_b = as<Dense<value_type>>(b);
        auto dense_x = as<Dense<value_type>>(x);
        get_executor()->run(fft::make_fft(dense_b, dense_x, inverse_, buffer_));
    }
}


void Fft::apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                     LinOp *x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>> *>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        float_x->scale(beta);
        float_x->add_scaled(alpha, lend(clone_x));
    } else {
        auto dense_x = as<Dense<value_type>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(clone_x));
    }
}


std::unique_ptr<LinOp> Fft2::transpose() const
{
    return Fft2::create(this->get_executor(), size1_, size2_, inverse_);
}


std::unique_ptr<LinOp> Fft2::conj_transpose() const
{
    return Fft2::create(this->get_executor(), size1_, size2_, !inverse_);
}


bool Fft2::is_inverse() const { return inverse_; }


void Fft2::apply_impl(const LinOp *b, LinOp *x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>> *>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft2(float_b, dense_x, size1_, size2_,
                                           inverse_, buffer_));
    } else {
        auto dense_b = as<Dense<value_type>>(b);
        auto dense_x = as<Dense<value_type>>(x);
        get_executor()->run(fft::make_fft2(dense_b, dense_x, size1_, size2_,
                                           inverse_, buffer_));
    }
}


void Fft2::apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                      LinOp *x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>> *>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        float_x->scale(beta);
        float_x->add_scaled(alpha, lend(clone_x));
    } else {
        auto dense_x = as<Dense<value_type>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(clone_x));
    }
}


std::unique_ptr<LinOp> Fft3::transpose() const
{
    return Fft3::create(this->get_executor(), size1_, size2_, size3_, inverse_);
}


std::unique_ptr<LinOp> Fft3::conj_transpose() const
{
    return Fft3::create(this->get_executor(), size1_, size2_, size3_,
                        !inverse_);
}


bool Fft3::is_inverse() const { return inverse_; }


void Fft3::apply_impl(const LinOp *b, LinOp *x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>> *>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft3(float_b, dense_x, size1_, size2_,
                                           size3_, inverse_, buffer_));
    } else {
        auto dense_b = as<Dense<value_type>>(b);
        auto dense_x = as<Dense<value_type>>(x);
        get_executor()->run(fft::make_fft3(dense_b, dense_x, size1_, size2_,
                                           size3_, inverse_, buffer_));
    }
}


void Fft3::apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                      LinOp *x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>> *>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        float_x->scale(beta);
        float_x->add_scaled(alpha, lend(clone_x));
    } else {
        auto dense_x = as<Dense<value_type>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, lend(clone_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(clone_x));
    }
}


}  // namespace matrix
}  // namespace gko
