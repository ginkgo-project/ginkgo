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

#include <ginkgo/core/base/lin_op.hpp>


#include <complex>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
T get_value_from_linop(const gko::LinOp *val);


template <typename T>
class DummyValueTypedLinOp
    : public gko::EnableValueTypedLinOp<DummyValueTypedLinOp<T>, T>,
      public gko::EnableCreateMethod<DummyValueTypedLinOp<T>> {
    using Self = DummyValueTypedLinOp<T>;

public:
    explicit DummyValueTypedLinOp(std::shared_ptr<const gko::Executor> exec,
                                  gko::dim<2> size = gko::dim<2>{})
        : gko::EnableValueTypedLinOp<Self, T>(exec, size), value_()
    {}
    DummyValueTypedLinOp(std::shared_ptr<const gko::Executor> exec,
                         gko::dim<2> size, T value)
        : gko::EnableValueTypedLinOp<Self, T>(exec, size), value_(value)
    {}

    T get_value() const { return value_; }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        T alpha_v = get_value_from_linop<T>(alpha);
        T beta_v = get_value_from_linop<T>(beta);

        gko::as<Self>(x)->value_ =
            alpha_v * gko::as<Self>(this)->value_ * gko::as<Self>(b)->value_ +
            beta_v * gko::as<Self>(x)->value_;
    }

    T value_;
};


template <typename T>
T get_value_from_linop(const gko::LinOp *val)
{
    if (auto *dense = dynamic_cast<const gko::matrix::Dense<T> *>(val)) {
        return dense->at(0, 0);
    } else {
        return (dynamic_cast<const DummyValueTypedLinOp<T> *>(val))
            ->get_value();
    }
}


template <typename T>
class EnableValueTypedLinOp : public ::testing::Test {
protected:
    using dummy_type = DummyValueTypedLinOp<T>;
    using value_type = T;

    gko::dim<2> dim{1, 1};
    std::shared_ptr<const gko::ReferenceExecutor> ref{
        gko::ReferenceExecutor::create()};
    std::unique_ptr<dummy_type> op{dummy_type::create(ref, dim, T{1.0})};
    T alpha_v{2.0};
    std::unique_ptr<dummy_type> alpha{dummy_type::create(ref, dim, alpha_v)};
    T beta_v{3.0};
    std::unique_ptr<dummy_type> beta{dummy_type::create(ref, dim, beta_v)};
    std::unique_ptr<dummy_type> b{dummy_type::create(ref, dim, T{4.0})};
    std::unique_ptr<dummy_type> x{dummy_type::create(ref, dim, T{5.0})};
};


TYPED_TEST_SUITE(EnableValueTypedLinOp, gko::test::ValueTypes);


TYPED_TEST(EnableValueTypedLinOp, CanCallExtendedApplyImplLinopLinop)
{
    using value_type = typename TestFixture::value_type;
    this->op->apply(lend(this->alpha), lend(this->b), lend(this->beta),
                    lend(this->x));

    ASSERT_EQ(this->x->get_value(), value_type{23.0});
}


TYPED_TEST(EnableValueTypedLinOp, CanCallExtendedApplyImplValueValue)
{
    auto reference_result = gko::clone(this->x);

    this->op->apply(this->alpha_v, lend(this->b), this->beta_v, lend(this->x));
    this->op->apply(lend(this->alpha), lend(this->b), lend(this->beta),
                    lend(reference_result));

    ASSERT_EQ(reference_result->get_value(), this->x->get_value());
}


TYPED_TEST(EnableValueTypedLinOp, CanCallExtendedApplyImplValueLinop)
{
    auto reference_result = gko::clone(this->x);

    this->op->apply(this->alpha_v, lend(this->b), lend(this->beta),
                    lend(this->x));
    this->op->apply(lend(this->alpha), lend(this->b), lend(this->beta),
                    lend(reference_result));

    ASSERT_EQ(reference_result->get_value(), this->x->get_value());
}


TYPED_TEST(EnableValueTypedLinOp, CanCallExtendedApplyImplLinopValue)
{
    auto reference_result = gko::clone(this->x);

    this->op->apply(lend(this->alpha), lend(this->b), this->beta_v,
                    lend(this->x));
    this->op->apply(lend(this->alpha), lend(this->b), lend(this->beta),
                    lend(reference_result));

    ASSERT_EQ(reference_result->get_value(), this->x->get_value());
}


}  // namespace
