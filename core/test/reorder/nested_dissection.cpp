// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/nested_dissection.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {

class NestedDissection : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using reorder_type =
        gko::experimental::reorder::NestedDissection<value_type, index_type>;

    NestedDissection()
        : exec(gko::ReferenceExecutor::create()),
          nd_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<reorder_type> nd_factory;
};

TEST_F(NestedDissection, KnowsItsExecutor)
{
    ASSERT_EQ(this->nd_factory->get_executor(), this->exec);
}

}  // namespace
