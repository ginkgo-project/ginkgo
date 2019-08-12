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

#include <ginkgo/core/base/reflection.hpp>


#include <vector>


#include <gtest/gtest.h>


namespace {


struct DummyOperator : public gko::EnableLinOp<DummyOperator> {
    DummyOperator(std::shared_ptr<const gko::Executor> exec,
                  gko::dim<2> size = {})
        : gko::EnableLinOp<DummyOperator>(exec, size)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override {}

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {}
};


struct TransposableDummyOperator
    : public gko::EnableLinOp<TransposableDummyOperator>,
      public gko::Transposable {
    TransposableDummyOperator(std::shared_ptr<const gko::Executor> exec,
                              gko::dim<2> size = {})
        : gko::EnableLinOp<TransposableDummyOperator>(exec, size)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override {}

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {}

    std::unique_ptr<LinOp> transpose() const override
    {
        auto result = std::unique_ptr<TransposableDummyOperator>(
            new TransposableDummyOperator(this->get_executor(),
                                          gko::transpose(this->get_size())));
        return std::move(result);
    }

    std::unique_ptr<LinOp> conj_transpose() const override
    {
        auto result = this->transpose();
        return std::move(result);
    }
};


class Reflection : public ::testing::Test {
protected:
    Reflection()
        : exec{gko::ReferenceExecutor::create()},
          U{std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1})},
          V{std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 2})},
          transU{std::make_shared<TransposableDummyOperator>(
              exec, gko::dim<2>{3, 1})},
          coef{std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 1})}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::LinOp> U;
    std::shared_ptr<gko::LinOp> V;
    std::shared_ptr<gko::LinOp> transU;
    std::shared_ptr<gko::LinOp> coef;
};


TEST_F(Reflection, CanBeEmpty)
{
    auto cmp = gko::Reflection<>::create(exec);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(0, 0));
}


TEST_F(Reflection, CanCreateFromTwoOperators)
{
    auto cmp = gko::Reflection<>::create(coef, U, V);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 2));
    ASSERT_EQ(cmp->get_u_operator(), U);
    ASSERT_EQ(cmp->get_v_operator(), V);
}


TEST_F(Reflection, CanNotCreateFromOneNonTransableOperators)
{
    ASSERT_THROW(gko::Reflection<>::create(coef, U), gko::NotSupported);
}


TEST_F(Reflection, CanCreateFromOneTransableOperators)
{
    auto cmp = gko::Reflection<>::create(coef, transU);
    ASSERT_EQ(cmp->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(cmp->get_u_operator(), transU);
    ASSERT_EQ(cmp->get_v_operator()->get_size(), gko::dim<2>(1, 3));
}


}  // namespace
