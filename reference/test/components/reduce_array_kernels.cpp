// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/reduce_array_kernels.hpp"


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class ReduceArray : public ::testing::Test {
protected:
    using value_type = T;
    ReduceArray()
        : ref{gko::ReferenceExecutor::create()},
          out{ref, I<T>{3}},
          vals{ref, I<T>{1, 4, 6}}
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::array<value_type> out;
    gko::array<value_type> vals;
};

TYPED_TEST_SUITE(ReduceArray, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(ReduceArray, KernelWorks)
{
    using T = typename TestFixture::value_type;

    gko::kernels::reference::components::reduce_add_array(this->ref, this->vals,
                                                          this->out);

    ASSERT_EQ(this->out.get_data()[0], T{14});
}


TYPED_TEST(ReduceArray, CoreWorks)
{
    using T = typename TestFixture::value_type;

    gko::reduce_add(this->vals, this->out);

    ASSERT_EQ(this->out.get_data()[0], T{14});
}


TYPED_TEST(ReduceArray, CoreWorks2)
{
    using T = typename TestFixture::value_type;

    auto result = gko::reduce_add(this->vals, T{1});

    ASSERT_EQ(result, T{12});
}


}  // namespace
