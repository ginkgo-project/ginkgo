/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/mtx_io.hpp>


#include <cstring>
#include <sstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/native_type.hpp>


#include "core/test/utils.hpp"


template <typename ValueType>
struct raw_array {
    ValueType* data;
    gko::size_type size;
};


template <typename ValueType>
struct raw_dense {
    ValueType* data;
    gko::dim<2> size;
    gko::size_type stride;
};


struct checks_compatibility {
    static int num_calls;

    template <typename T>
    static void check_compatibility(T&& obj)
    {
        num_calls++;
    }
};
int checks_compatibility::num_calls = 0;


struct array_mapper : checks_compatibility {
    template <typename ValueType>
    using type = raw_array<ValueType>;

    template <typename ValueType>
    static type<ValueType> map(ValueType* data, gko::size_type size)
    {
        return type<ValueType>{data, size};
    }
};


struct dense_mapper : checks_compatibility {
    template <typename ValueType>
    using type = raw_dense<ValueType>;

    template <typename ValueType>
    static type<ValueType> map(ValueType* data, gko::dim<2> size,
                               gko::size_type stride)
    {
        return type<ValueType>{data, size, stride};
    }
};

using native = gko::native<array_mapper, dense_mapper>;

template <typename ValueType>
class ArrayMapper : public ::testing::Test {
protected:
    using value_type = ValueType;

    std::shared_ptr<gko::ReferenceExecutor> ref =
        gko::ReferenceExecutor::create();
    gko::array<value_type> array = {ref, I<value_type>{1, 2, 3, 4}};
};

TYPED_TEST_SUITE(ArrayMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ArrayMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_array = native::map(this->array);

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<array_type, raw_array<value_type>>);
    ASSERT_EQ(mapped_array.data, this->array.get_data());
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}


TYPED_TEST(ArrayMapper, CanMapRValue)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_array = native::map(this->array.as_view());

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<array_type, raw_array<value_type>>);
    ASSERT_EQ(mapped_array.data, this->array.get_data());
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}


TYPED_TEST(ArrayMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_array =
        native::map(const_cast<const gko::array<value_type>&>(this->array));

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<array_type, raw_array<const value_type>>);
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}


template <typename ValueType>
class DenseMapper : public ::testing::Test {
protected:
    using value_type = ValueType;
    using mtx_type = gko::matrix::Dense<value_type>;

    std::shared_ptr<gko::ReferenceExecutor> ref =
        gko::ReferenceExecutor::create();
    std::unique_ptr<mtx_type> mtx =
        gko::initialize<mtx_type>({1, 2, 3, 4}, ref);
};

TYPED_TEST_SUITE(DenseMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_mtx = native::map(this->mtx);

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(std::is_same_v<array_type, raw_dense<value_type>>);
    ASSERT_EQ(mapped_mtx.data, this->mtx->get_values());
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}


TYPED_TEST(DenseMapper, CanMapRValue)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_mtx = native::map(gko::make_dense_view(this->mtx));

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(std::is_same_v<array_type, raw_dense<value_type>>);
    ASSERT_EQ(mapped_mtx.data, this->mtx->get_values());
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}


TYPED_TEST(DenseMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    auto num_call_checks = checks_compatibility::num_calls;

    auto mapped_array = native::map(gko::make_const_dense_view(this->mtx));

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<array_type, raw_dense<const value_type>>);
    ASSERT_EQ(checks_compatibility::num_calls, num_call_checks + 1);
}
