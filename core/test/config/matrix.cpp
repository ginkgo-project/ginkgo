// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "cmake-build-debug/_deps/googletest-src/googletest/include/gtest/gtest-typed-test.h"
#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


template <typename ValueType>
class Identity : public ::testing::Test {
protected:
    using value_type = ValueType;

    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::matrix::Identity<value_type>> ans =
        gko::matrix::Identity<value_type>::create(exec, 4u);
};

TYPED_TEST_SUITE(Identity, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Identity, CanParse)
{
    using value_type = typename TestFixture::value_type;
    auto config = pnode({{"type", pnode("matrix::Identity")}});

    auto res = parse(config, {}, make_type_descriptor<value_type>())
                   .on(this->exec)
                   ->generate(this->ans);

    ASSERT_TRUE(dynamic_cast<gko::matrix::Identity<value_type>*>(res.get()));
    GKO_ASSERT_EQUAL_DIMENSIONS(res, this->ans);
}
