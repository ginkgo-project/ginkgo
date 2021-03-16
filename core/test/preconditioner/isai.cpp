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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


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
    using excess_solver_type = gko::solver::Bicgstab<value_type>;
    using GeneralIsai =
        gko::preconditioner::GeneralIsai<value_type, index_type>;
    using SpdIsai = gko::preconditioner::SpdIsai<value_type, index_type>;
    using LowerIsai = gko::preconditioner::LowerIsai<value_type, index_type>;
    using UpperIsai = gko::preconditioner::UpperIsai<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    IsaiFactory()
        : exec(gko::ReferenceExecutor::create()),
          excess_solver_factory(excess_solver_type::build().on(exec)),
          general_isai_factory(GeneralIsai::build().on(exec)),
          spd_isai_factory(SpdIsai::build().on(exec)),
          lower_isai_factory(LowerIsai::build().on(exec)),
          upper_isai_factory(UpperIsai::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename excess_solver_type::Factory> excess_solver_factory;
    std::unique_ptr<typename GeneralIsai::Factory> general_isai_factory;
    std::unique_ptr<typename SpdIsai::Factory> spd_isai_factory;
    std::unique_ptr<typename LowerIsai::Factory> lower_isai_factory;
    std::unique_ptr<typename UpperIsai::Factory> upper_isai_factory;
};

TYPED_TEST_SUITE(IsaiFactory, gko::test::ValueIndexTypes);


TYPED_TEST(IsaiFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->general_isai_factory->get_executor(), this->exec);
    ASSERT_EQ(this->spd_isai_factory->get_executor(), this->exec);
    ASSERT_EQ(this->lower_isai_factory->get_executor(), this->exec);
    ASSERT_EQ(this->upper_isai_factory->get_executor(), this->exec);
}


TYPED_TEST(IsaiFactory, SetsSkipSortingCorrectly)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using SpdIsai = typename TestFixture::SpdIsai;
    using LowerIsai = typename TestFixture::LowerIsai;
    using UpperIsai = typename TestFixture::UpperIsai;

    auto a_isai_factory =
        GeneralIsai::build().with_skip_sorting(true).on(this->exec);
    auto spd_isai_factory =
        SpdIsai::build().with_skip_sorting(true).on(this->exec);
    auto l_isai_factory =
        LowerIsai::build().with_skip_sorting(true).on(this->exec);
    auto u_isai_factory =
        UpperIsai::build().with_skip_sorting(true).on(this->exec);

    ASSERT_EQ(a_isai_factory->get_parameters().skip_sorting, true);
    ASSERT_EQ(spd_isai_factory->get_parameters().skip_sorting, true);
    ASSERT_EQ(l_isai_factory->get_parameters().skip_sorting, true);
    ASSERT_EQ(u_isai_factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(IsaiFactory, SetsDefaultSkipSortingCorrectly)
{
    ASSERT_EQ(this->general_isai_factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(this->spd_isai_factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(this->lower_isai_factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(this->upper_isai_factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(IsaiFactory, SetsSparsityPowerCorrectly)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using SpdIsai = typename TestFixture::SpdIsai;
    using LowerIsai = typename TestFixture::LowerIsai;
    using UpperIsai = typename TestFixture::UpperIsai;

    auto a_isai_factory =
        GeneralIsai::build().with_sparsity_power(2).on(this->exec);
    auto spd_isai_factory =
        SpdIsai::build().with_sparsity_power(2).on(this->exec);
    auto l_isai_factory =
        LowerIsai::build().with_sparsity_power(2).on(this->exec);
    auto u_isai_factory =
        UpperIsai::build().with_sparsity_power(2).on(this->exec);

    ASSERT_EQ(a_isai_factory->get_parameters().sparsity_power, 2);
    ASSERT_EQ(spd_isai_factory->get_parameters().sparsity_power, 2);
    ASSERT_EQ(l_isai_factory->get_parameters().sparsity_power, 2);
    ASSERT_EQ(u_isai_factory->get_parameters().sparsity_power, 2);
}


TYPED_TEST(IsaiFactory, SetsDefaultSparsityPowerCorrectly)
{
    ASSERT_EQ(this->general_isai_factory->get_parameters().sparsity_power, 1);
    ASSERT_EQ(this->spd_isai_factory->get_parameters().sparsity_power, 1);
    ASSERT_EQ(this->lower_isai_factory->get_parameters().sparsity_power, 1);
    ASSERT_EQ(this->upper_isai_factory->get_parameters().sparsity_power, 1);
}


TYPED_TEST(IsaiFactory, SetsExcessLimitCorrectly)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using SpdIsai = typename TestFixture::SpdIsai;
    using LowerIsai = typename TestFixture::LowerIsai;
    using UpperIsai = typename TestFixture::UpperIsai;

    auto a_isai_factory =
        GeneralIsai::build().with_excess_limit(1024u).on(this->exec);
    auto spd_isai_factory =
        SpdIsai::build().with_excess_limit(1024u).on(this->exec);
    auto l_isai_factory =
        LowerIsai::build().with_excess_limit(1024u).on(this->exec);
    auto u_isai_factory =
        UpperIsai::build().with_excess_limit(1024u).on(this->exec);

    ASSERT_EQ(a_isai_factory->get_parameters().excess_limit, 1024u);
    ASSERT_EQ(spd_isai_factory->get_parameters().excess_limit, 1024u);
    ASSERT_EQ(l_isai_factory->get_parameters().excess_limit, 1024u);
    ASSERT_EQ(u_isai_factory->get_parameters().excess_limit, 1024u);
}


TYPED_TEST(IsaiFactory, SetsDefaultExcessLimitCorrectly)
{
    ASSERT_EQ(this->general_isai_factory->get_parameters().excess_limit, 0u);
    ASSERT_EQ(this->spd_isai_factory->get_parameters().excess_limit, 0u);
    ASSERT_EQ(this->lower_isai_factory->get_parameters().excess_limit, 0u);
    ASSERT_EQ(this->upper_isai_factory->get_parameters().excess_limit, 0u);
}


TYPED_TEST(IsaiFactory, CanSetExcessSolverFactoryA)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    auto general_isai_factory =
        GeneralIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);

    ASSERT_EQ(general_isai_factory->get_parameters().excess_solver_factory,
              this->excess_solver_factory);
}


TYPED_TEST(IsaiFactory, CanSetExcessSolverFactorySpd)
{
    using SpdIsai = typename TestFixture::SpdIsai;
    auto spd_isai_factory =
        SpdIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);

    ASSERT_EQ(spd_isai_factory->get_parameters().excess_solver_factory,
              this->excess_solver_factory);
}


TYPED_TEST(IsaiFactory, CanSetExcessSolverFactoryL)
{
    using LowerIsai = typename TestFixture::LowerIsai;
    auto lower_isai_factory =
        LowerIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);

    ASSERT_EQ(lower_isai_factory->get_parameters().excess_solver_factory,
              this->excess_solver_factory);
}


TYPED_TEST(IsaiFactory, CanSetExcessSolverFactoryU)
{
    using UpperIsai = typename TestFixture::UpperIsai;
    auto upper_isai_factory =
        UpperIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);

    ASSERT_EQ(upper_isai_factory->get_parameters().excess_solver_factory,
              this->excess_solver_factory);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionA)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = Csr::create(this->exec, gko::dim<2>{1, 2}, 1);

    ASSERT_THROW(this->general_isai_factory->generate(gko::share(mtx)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionSpd)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = Csr::create(this->exec, gko::dim<2>{1, 2}, 1);

    ASSERT_THROW(this->spd_isai_factory->generate(gko::share(mtx)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionL)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = Csr::create(this->exec, gko::dim<2>{1, 2}, 1);

    ASSERT_THROW(this->lower_isai_factory->generate(gko::share(mtx)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionU)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = Csr::create(this->exec, gko::dim<2>{1, 2}, 1);

    ASSERT_THROW(this->upper_isai_factory->generate(gko::share(mtx)),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrA)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = DummyOperator::create(this->exec, gko::dim<2>{2, 2});

    ASSERT_THROW(this->general_isai_factory->generate(gko::share(mtx)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrSpd)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = DummyOperator::create(this->exec, gko::dim<2>{2, 2});

    ASSERT_THROW(this->spd_isai_factory->generate(gko::share(mtx)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrL)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = DummyOperator::create(this->exec, gko::dim<2>{2, 2});

    ASSERT_THROW(this->lower_isai_factory->generate(gko::share(mtx)),
                 gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrU)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = DummyOperator::create(this->exec, gko::dim<2>{2, 2});

    ASSERT_THROW(this->upper_isai_factory->generate(gko::share(mtx)),
                 gko::NotSupported);
}


}  // namespace
