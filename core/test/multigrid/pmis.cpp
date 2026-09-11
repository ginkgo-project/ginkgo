// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class PmisFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;
    using real_type = gko::remove_complex<value_type>;
    PmisFactory()
        : exec(gko::ReferenceExecutor::create()),
          pmis_factory(MgLevel::build()
                           .with_strength_threshold(real_type{0.125})
                           .with_skip_sorting(true)
                           .on(exec))

    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> pmis_factory;
};

TYPED_TEST_SUITE(PmisFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(PmisFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->pmis_factory->get_executor(), this->exec);
}


TYPED_TEST(PmisFactory, DefaultSetting)
{
    using real_type = typename TestFixture::real_type;
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().strength_threshold, real_type{0.25});
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(PmisFactory, SetStrengthThreshold)
{
    using real_type = typename TestFixture::real_type;
    ASSERT_EQ(this->pmis_factory->get_parameters().strength_threshold,
              real_type{0.125});
}

TYPED_TEST(PmisFactory, SetSkipSorting)
{
    ASSERT_EQ(this->pmis_factory->get_parameters().skip_sorting, true);
}
