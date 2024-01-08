// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


class Rcm : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    using new_reorder_type = gko::experimental::reorder::Rcm<i_type>;

    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          rcm_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<reorder_type::Factory> rcm_factory;
};


TEST_F(Rcm, RcmFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->rcm_factory->get_executor(), this->exec);
}


TEST_F(Rcm, NewInterfaceDefaults)
{
    auto param = new_reorder_type::build();

    ASSERT_EQ(param.skip_symmetrize, false);
    ASSERT_EQ(param.strategy,
              gko::reorder::starting_strategy::pseudo_peripheral);
}


TEST_F(Rcm, NewInterfaceSetParameters)
{
    auto param =
        new_reorder_type::build().with_skip_symmetrize(true).with_strategy(
            gko::reorder::starting_strategy::minimum_degree);

    ASSERT_EQ(param.skip_symmetrize, true);
    ASSERT_EQ(param.strategy, gko::reorder::starting_strategy::minimum_degree);
}


}  // namespace
