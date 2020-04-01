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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <memory>

#include <iostream>  // TODO: delete me

#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"


namespace {


struct DummyOperator : public gko::EnableLinOp<DummyOperator>,
                       gko::EnableCreateMethod<DummyOperator> {
    DummyOperator(std::shared_ptr<const gko::Executor> exec,
                  gko::dim<2> size = {})
        : gko::EnableLinOp<DummyOperator>(exec, size)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override {}

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {}
};


template <typename ValueIndexType>
class IsaiFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Isai = gko::preconditioner::Isai<value_type, index_type>;
    using Comp = gko::Composition<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    IsaiFactory()
        : exec(gko::ReferenceExecutor::create()),
          isai_factory(Isai::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Isai::Factory> isai_factory;
};

TYPED_TEST_CASE(IsaiFactory, gko::test::ValueIndexTypes);


TYPED_TEST(IsaiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->isai_factory->get_executor(), this->exec);
}


TYPED_TEST(IsaiFactory, ThrowsWrongInput)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    auto mtx = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto comp = Comp::create(std::move(mtx));

    ASSERT_THROW(this->isai_factory->generate(gko::share(comp)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsWrongExclusive)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    using Isai = typename TestFixture::Isai;
    auto mtx1 = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto mtx2 = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto comp = Comp::create(std::move(mtx1), std::move(mtx2));

    auto factory = Isai::build()
                       .with_exclusive_factor_l(true)
                       .with_exclusive_factor_u(true)
                       .on(this->exec);

    ASSERT_THROW(factory->generate(gko::share(comp)), gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsWrongComposition)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = Csr::create(this->exec, gko::dim<2>{1, 1}, 1);

    ASSERT_THROW(this->isai_factory->generate(gko::share(mtx)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimension)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    auto mtx1 = Csr::create(this->exec, gko::dim<2>{1, 2}, 1);
    auto mtx2 = Csr::create(this->exec, gko::dim<2>{2, 1}, 1);
    auto comp = Comp::create(std::move(mtx1), std::move(mtx2));

    ASSERT_THROW(this->isai_factory->generate(gko::share(comp)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimension2)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    auto mtx1 = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto mtx2 = Csr::create(this->exec, gko::dim<2>{2, 3}, 1);
    auto comp = Comp::create(std::move(mtx1), std::move(mtx2));

    ASSERT_THROW(this->isai_factory->generate(gko::share(comp)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsr)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    auto mtx1 = DummyOperator::create(this->exec, gko::dim<2>{2, 2});
    auto mtx2 = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto comp = Comp::create(std::move(mtx1), std::move(mtx2));

    ASSERT_THROW(this->isai_factory->generate(gko::share(comp)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsr2)
{
    using Csr = typename TestFixture::Csr;
    using Comp = typename TestFixture::Comp;
    using Isai = typename TestFixture::Isai;
    auto mtx1 = Csr::create(this->exec, gko::dim<2>{2, 2}, 1);
    auto mtx2 = DummyOperator::create(this->exec, gko::dim<2>{2, 2});
    auto comp = Comp::create(std::move(mtx1), std::move(mtx2));
    auto factory = Isai::build().with_exclusive_factor_u(true).on(this->exec);

    ASSERT_THROW(factory->generate(gko::share(comp)), gko::NotSupported);
}


}  // namespace
