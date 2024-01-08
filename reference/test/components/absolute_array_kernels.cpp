// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/absolute_array_kernels.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class AbsoluteArray : public ::testing::Test {
protected:
    using value_type = T;
    using abs_type = gko::remove_complex<T>;
    AbsoluteArray()
        : ref(gko::ReferenceExecutor::create()),
          total_size(6344),
          inplace_expected(ref, total_size),
          outplace_expected(ref, total_size),
          vals(ref, total_size)
    {
        std::fill_n(inplace_expected.get_data(), total_size, T(6453));
        std::fill_n(vals.get_data(), total_size, T(-6453));
        std::fill_n(outplace_expected.get_data(), total_size, abs_type(6453));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::size_type total_size;
    gko::array<value_type> inplace_expected;
    gko::array<abs_type> outplace_expected;
    gko::array<value_type> vals;
};

TYPED_TEST_SUITE(AbsoluteArray, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(AbsoluteArray, InplaceEqualsExpected)
{
    using T = typename TestFixture::value_type;

    gko::kernels::reference::components::inplace_absolute_array(
        this->ref, this->vals.get_data(), this->total_size);

    GKO_ASSERT_ARRAY_EQ(this->vals, this->inplace_expected);
}


TYPED_TEST(AbsoluteArray, OutplaceEqualsExpected)
{
    using T = typename TestFixture::value_type;
    using AbsT = typename TestFixture::abs_type;
    gko::array<AbsT> abs_vals(this->ref, this->total_size);

    gko::kernels::reference::components::outplace_absolute_array(
        this->ref, this->vals.get_const_data(), this->total_size,
        abs_vals.get_data());

    GKO_ASSERT_ARRAY_EQ(abs_vals, this->outplace_expected);
}


}  // namespace
