// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
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

TYPED_TEST_SUITE(IsaiFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


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


TYPED_TEST(IsaiFactory, SetsExcessSolverReductionCorrectly)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using SpdIsai = typename TestFixture::SpdIsai;
    using LowerIsai = typename TestFixture::LowerIsai;
    using UpperIsai = typename TestFixture::UpperIsai;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto a_isai_factory = GeneralIsai::build()
                              .with_excess_solver_reduction(real_type{1e-3})
                              .on(this->exec);
    auto spd_isai_factory = SpdIsai::build()
                                .with_excess_solver_reduction(real_type{1e-3})
                                .on(this->exec);
    auto l_isai_factory = LowerIsai::build()
                              .with_excess_solver_reduction(real_type{1e-3})
                              .on(this->exec);
    auto u_isai_factory = UpperIsai::build()
                              .with_excess_solver_reduction(real_type{1e-3})
                              .on(this->exec);

    ASSERT_NEAR(a_isai_factory->get_parameters().excess_solver_reduction, 1e-3,
                r<value_type>::value);
    ASSERT_NEAR(spd_isai_factory->get_parameters().excess_solver_reduction,
                1e-3, r<value_type>::value);
    ASSERT_NEAR(l_isai_factory->get_parameters().excess_solver_reduction, 1e-3,
                r<value_type>::value);
    ASSERT_NEAR(u_isai_factory->get_parameters().excess_solver_reduction, 1e-3,
                r<value_type>::value);
}


TYPED_TEST(IsaiFactory, SetsDefaultExcessSolverReductionCorrectly)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_NEAR(
        this->general_isai_factory->get_parameters().excess_solver_reduction,
        1e-6, r<value_type>::value);
    ASSERT_NEAR(
        this->spd_isai_factory->get_parameters().excess_solver_reduction, 1e-6,
        r<value_type>::value);
    ASSERT_NEAR(
        this->lower_isai_factory->get_parameters().excess_solver_reduction,
        1e-6, r<value_type>::value);
    ASSERT_NEAR(
        this->upper_isai_factory->get_parameters().excess_solver_reduction,
        1e-6, r<value_type>::value);
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
    auto mtx = gko::share(Csr::create(this->exec, gko::dim<2>{1, 2}, 1));

    ASSERT_THROW(this->general_isai_factory->generate(mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionSpd)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(Csr::create(this->exec, gko::dim<2>{1, 2}, 1));

    ASSERT_THROW(this->spd_isai_factory->generate(mtx), gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionL)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(Csr::create(this->exec, gko::dim<2>{1, 2}, 1));

    ASSERT_THROW(this->lower_isai_factory->generate(mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsWrongDimensionU)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(Csr::create(this->exec, gko::dim<2>{1, 2}, 1));

    ASSERT_THROW(this->upper_isai_factory->generate(mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrA)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(DummyOperator::create(this->exec, gko::dim<2>{2, 2}));

    ASSERT_THROW(this->general_isai_factory->generate(mtx), gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrSpd)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(DummyOperator::create(this->exec, gko::dim<2>{2, 2}));

    ASSERT_THROW(this->spd_isai_factory->generate(mtx), gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrL)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(DummyOperator::create(this->exec, gko::dim<2>{2, 2}));

    ASSERT_THROW(this->lower_isai_factory->generate(mtx), gko::NotSupported);
}


TYPED_TEST(IsaiFactory, ThrowsNoConversionCsrU)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = gko::share(DummyOperator::create(this->exec, gko::dim<2>{2, 2}));

    ASSERT_THROW(this->upper_isai_factory->generate(mtx), gko::NotSupported);
}


}  // namespace
