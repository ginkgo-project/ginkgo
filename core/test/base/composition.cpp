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

#include <ginkgo/core/base/composition.hpp>


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


class Composition : public ::testing::Test {
protected:
    Composition()
        : exec{gko::ReferenceExecutor::create()},
          operators{std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1}),
                    std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 3})}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};


TEST_F(Composition, CanBeEmpty)
{
    auto cmp = gko::Composition<>::create(exec);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmp->get_operators().size(), 0);
}


TEST_F(Composition, CanCreateFromIterators)
{
    auto cmp = gko::Composition<>::create(begin(operators), end(operators));

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(cmp->get_operators().size(), 2);
    ASSERT_EQ(cmp->get_operators()[0], operators[0]);
    ASSERT_EQ(cmp->get_operators()[1], operators[1]);
}


TEST_F(Composition, CanCreateFromList)
{
    auto cmp = gko::Composition<>::create(operators[0], operators[1]);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(cmp->get_operators().size(), 2);
    ASSERT_EQ(cmp->get_operators()[0], operators[0]);
    ASSERT_EQ(cmp->get_operators()[1], operators[1]);
}


}  // namespace
