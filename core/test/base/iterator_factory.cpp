/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/base/iterator_factory.hpp"


#include <algorithm>
#include <vector>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>


namespace {


template <typename ValueIndexType>
class IteratorFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    IteratorFactory()
        : reversed_index{100, 50, 10, 9, 8, 7, 5, 5, 4, 3, 2, 1, 0, -1, -2},
          ordered_index{-2, -1, 0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 50, 100},
          reversed_value{15., 14., 13., 12., 11., 10., 9., 7.,
                         7.,  6.,  5.,  4.,  3.,  2.,  -1.},
          ordered_value{-1., 2.,  3.,  4.,  5.,  6.,  7., 7.,
                        9.,  10., 11., 12., 13., 14., 15.}
    {}

    template <typename T1, typename T2>
    void check_vector_equal(const std::vector<T1> &v1,
                            const std::vector<T2> &v2)
    {
        ASSERT_TRUE(std::equal(v1.begin(), v1.end(), v2.begin()));
    }

    // Require that Iterator has a `value_type` specified
    template <typename Iterator, typename = typename Iterator::value_type>
    bool is_sorted_iterator(Iterator begin, Iterator end)
    {
        using value_type = typename Iterator::value_type;
        for (; begin + 1 < end; ++begin) {
            auto curr_ref = *begin;
            auto curr_val = static_cast<value_type>(curr_ref);
            auto next_ref = *(begin + 1);
            auto next_val = static_cast<value_type>(next_ref);

            // Test all combinations of the `<` operator
            if (*(begin + 1) < *begin || next_ref < curr_ref ||
                next_ref < curr_val || next_val < curr_ref ||
                next_val < curr_val) {
                return false;
            }
        }
        return true;
    }

    const std::vector<index_type> reversed_index;
    const std::vector<index_type> ordered_index;
    const std::vector<value_type> reversed_value;
    const std::vector<value_type> ordered_value;
};


TYPED_TEST_CASE(IteratorFactory, gko::test::ValueIndexTypes);


TYPED_TEST(IteratorFactory, EmptyIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        nullptr, nullptr, 0);

    ASSERT_TRUE(test_iter.begin() == test_iter.end());
    ASSERT_NO_THROW(std::sort(test_iter.begin(), test_iter.end()));
}


TYPED_TEST(IteratorFactory, SortingReversedWithIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    std::sort(test_iter.begin(), test_iter.end());

    this->check_vector_equal(vec1, this->ordered_index);
    this->check_vector_equal(vec2, this->reversed_value);
}


TYPED_TEST(IteratorFactory, SortingAlreadySortedWithIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->ordered_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    std::sort(test_iter.begin(), test_iter.end());

    this->check_vector_equal(vec1, this->ordered_index);
    this->check_vector_equal(vec2, this->ordered_value);
}


TYPED_TEST(IteratorFactory, IteratorReferenceOperatorSmaller)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    bool is_sorted =
        this->is_sorted_iterator(test_iter.begin(), test_iter.end());

    ASSERT_FALSE(is_sorted);
}


TYPED_TEST(IteratorFactory, IteratorReferenceOperatorSmaller2)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->ordered_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    bool is_sorted =
        this->is_sorted_iterator(test_iter.begin(), test_iter.end());

    ASSERT_TRUE(is_sorted);
}


TYPED_TEST(IteratorFactory, IncreasingIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    auto begin = test_iter.begin();
    auto plus_2 = begin + 2;
    auto plus_minus_2 = plus_2 - 2;
    auto increment_pre_2 = begin;
    ++increment_pre_2;
    ++increment_pre_2;
    auto increment_post_2 = begin;
    increment_post_2++;
    increment_post_2++;
    auto increment_pre_test = begin;
    auto increment_post_test = begin;

    ASSERT_TRUE(begin == plus_minus_2);
    ASSERT_TRUE(plus_2 == increment_pre_2);
    ASSERT_TRUE(increment_pre_2 == increment_post_2);
    ASSERT_TRUE(begin == increment_post_test++);
    ASSERT_TRUE(begin + 1 == ++increment_pre_test);
    ASSERT_TRUE((*plus_2).dominant() == vec1[2]);
    ASSERT_TRUE((*plus_2).secondary() == vec2[2]);
}


TYPED_TEST(IteratorFactory, DecreasingIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    auto iter = test_iter.begin() + 5;
    auto minus_2 = iter - 2;
    auto minus_plus_2 = minus_2 + 2;
    auto decrement_pre_2 = iter;
    --decrement_pre_2;
    --decrement_pre_2;
    auto decrement_post_2 = iter;
    decrement_post_2--;
    decrement_post_2--;
    auto decrement_pre_test = iter;
    auto decrement_post_test = iter;

    ASSERT_TRUE(iter == minus_plus_2);
    ASSERT_TRUE(minus_2 == decrement_pre_2);
    ASSERT_TRUE(decrement_pre_2 == decrement_post_2);
    ASSERT_TRUE(iter == decrement_post_test--);
    ASSERT_TRUE(iter - 1 == --decrement_pre_test);
    ASSERT_TRUE((*minus_2).dominant() == vec1[3]);
    ASSERT_TRUE((*minus_2).secondary() == vec2[3]);
}


TYPED_TEST(IteratorFactory, CorrectDereferencing)
{
    using index_type_it = typename TestFixture::index_type;
    using value_type_it = typename TestFixture::value_type;
    std::vector<index_type_it> vec1{this->reversed_index};
    std::vector<value_type_it> vec2{this->ordered_value};
    constexpr int element_to_test = 3;

    auto test_iter = gko::detail::IteratorFactory<index_type_it, value_type_it>(
        vec1.data(), vec2.data(), vec1.size());
    auto begin = test_iter.begin();
    using value_type = typename decltype(begin)::value_type;
    auto to_test_ref = *(begin + element_to_test);
    value_type to_test_pair = to_test_ref;  // Testing implicit conversion

    ASSERT_TRUE(to_test_pair.dominant == vec1[element_to_test]);
    ASSERT_TRUE(to_test_pair.dominant == to_test_ref.dominant());
    ASSERT_TRUE(to_test_pair.secondary == vec2[element_to_test]);
    ASSERT_TRUE(to_test_pair.secondary == to_test_ref.secondary());
}


TYPED_TEST(IteratorFactory, CorrectSwapping)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    auto first_el_reference = *test_iter.begin();
    auto second_el_reference = *(test_iter.begin() + 1);
    swap(first_el_reference, second_el_reference);

    ASSERT_TRUE(vec1[0] == this->reversed_index[1]);
    ASSERT_TRUE(vec1[1] == this->reversed_index[0]);
    ASSERT_TRUE(vec2[0] == this->ordered_value[1]);
    ASSERT_TRUE(vec2[1] == this->ordered_value[0]);
    // Make sure the other values were not touched.
    for (size_t i = 2; i < vec1.size(); ++i) {
        ASSERT_TRUE(vec1[i] == this->reversed_index[i]);
        ASSERT_TRUE(vec2[i] == this->ordered_value[i]);
    }
}


TYPED_TEST(IteratorFactory, CorrectHandWrittenSwapping)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::IteratorFactory<index_type, value_type>(
        vec1.data(), vec2.data(), vec1.size());
    auto first_el_reference = *test_iter.begin();
    auto second_el_reference = *(test_iter.begin() + 1);
    auto temp = static_cast<typename decltype(test_iter.begin())::value_type>(
        first_el_reference);
    first_el_reference = second_el_reference;
    second_el_reference = temp;

    ASSERT_TRUE(vec1[0] == this->reversed_index[1]);
    ASSERT_TRUE(vec1[1] == this->reversed_index[0]);
    ASSERT_TRUE(vec2[0] == this->ordered_value[1]);
    ASSERT_TRUE(vec2[1] == this->ordered_value[0]);
    // Make sure the other values were not touched.
    for (size_t i = 2; i < vec1.size(); ++i) {
        ASSERT_TRUE(vec1[i] == this->reversed_index[i]);
        ASSERT_TRUE(vec2[i] == this->ordered_value[i]);
    }
}


}  // namespace
