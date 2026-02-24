// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/multigrid/rs.hpp>

#include "core/test/utils.hpp"


namespace {


// template <typename ValueIndexType>
// class RsFactory : public ::testing::Test {
// protected:
//     using value_type =
//         typename std::tuple_element<0, decltype(ValueIndexType())>::type;
//     using index_type =
//         typename std::tuple_element<1, decltype(ValueIndexType())>::type;
//     using Mtx = gko::matrix::Csr<value_type, index_type>;
//     using Vec = gko::matrix::Dense<value_type>;
//     using MgLevel = gko::multigrid::Rs<value_type, index_type>;
//     RsFactory()
//         : exec(gko::ReferenceExecutor::create()),
//           rs_factory(MgLevel::build()
//                          .with_coarse_rows(gko::array<index_type>(exec, {2,
//                          3})) .with_skip_sorting(true) .on(exec))
//     {}

//     std::shared_ptr<const gko::Executor> exec;
//     std::unique_ptr<typename MgLevel::Factory> rs_factory;
// };

// TYPED_TEST_SUITE(RsFactory, gko::test::ValueIndexTypes,
//                  PairTypenameNameGenerator);


// TYPED_TEST(RsFactory, FactoryKnowsItsExecutor)
// {
//     ASSERT_EQ(this->rs_factory->get_executor(), this->exec);
// }


// TYPED_TEST(RsFactory, DefaultSetting)
// {
//     using MgLevel = typename TestFixture::MgLevel;
//     auto factory = MgLevel::build().on(this->exec);

//     ASSERT_EQ(factory->get_parameters().coarse_rows.get_const_data(),
//     nullptr); ASSERT_EQ(factory->get_parameters().skip_sorting, false);
// }


// TYPED_TEST(RsFactory, SetCoarseRows)
// {
//     using T = typename TestFixture::index_type;
//     GKO_ASSERT_ARRAY_EQ(this->rs_factory->get_parameters().coarse_rows,
//                         gko::array<T>(this->exec, {2, 3}));
// }


// TYPED_TEST(RsFactory, SetSkipSorting)
// {
//     ASSERT_EQ(this->rs_factory->get_parameters().skip_sorting, true);
// }


}  // namespace
