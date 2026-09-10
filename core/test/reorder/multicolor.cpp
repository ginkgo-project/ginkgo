// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils.hpp"


template <typename IndexType>
class Multicolor : public ::testing::Test {
protected:
    using v_type = float;
    using i_type = IndexType;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;

    Multicolor()
        : exec(gko::ReferenceExecutor::create()),
          mc_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename reorder_type::Factory> mc_factory;
};

TYPED_TEST_SUITE(Multicolor, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(Multicolor, MulticolorFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->mc_factory->get_executor(), this->exec);
}
