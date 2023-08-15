// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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

}  // namespace
