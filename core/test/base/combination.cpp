/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/base/combination.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>


namespace {


struct DummyOperator : public gko::EnableLinOp<DummyOperator> {
    DummyOperator(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyOperator>(exec, gko::dim<2>{1, 1})
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override {}

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {}
};


template <typename T>
class Combination : public ::testing::Test {
protected:
    Combination()
        : exec{gko::ReferenceExecutor::create()},
          operators{std::make_shared<DummyOperator>(exec),
                    std::make_shared<DummyOperator>(exec)},
          coefficients{std::make_shared<DummyOperator>(exec),
                       std::make_shared<DummyOperator>(exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
};


TYPED_TEST_CASE(Combination, gko::test::ValueTypes);


TYPED_TEST(Combination, CanBeEmpty)
{
    auto cmb = gko::Combination<TypeParam>::create(this->exec);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
    ASSERT_EQ(cmb->get_operators().size(), 0);
}


TYPED_TEST(Combination, CanCreateFromIterators)
{
    auto cmb = gko::Combination<TypeParam>::create(
        begin(this->coefficients), end(this->coefficients),
        begin(this->operators), end(this->operators));

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(1, 1));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);
    ASSERT_EQ(cmb->get_coefficients()[0], this->coefficients[0]);
    ASSERT_EQ(cmb->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmb->get_coefficients()[1], this->coefficients[1]);
    ASSERT_EQ(cmb->get_operators()[1], this->operators[1]);
}


TYPED_TEST(Combination, CanCreateFromList)
{
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(1, 1));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);
    ASSERT_EQ(cmb->get_coefficients()[0], this->coefficients[0]);
    ASSERT_EQ(cmb->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmb->get_coefficients()[1], this->coefficients[1]);
    ASSERT_EQ(cmb->get_operators()[1], this->operators[1]);
}


}  // namespace
