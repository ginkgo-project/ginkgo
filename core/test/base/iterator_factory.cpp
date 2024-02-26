// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/iterator_factory.hpp"


#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace std {


// add a comparison for std::complex to allow comparisons without custom
// comparator
// HERE BE DRAGONS! This is technically UB, since we are adding things to
// namespace std, but since it is only inside a test, it should be fine :)
template <typename ValueType>
bool operator<(const std::complex<ValueType>& a,
               const std::complex<ValueType>& b)
{
    return a.real() < b.real();
}


}  // namespace std


namespace {


template <typename ValueIndexType>
class ZipIterator : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    ZipIterator()
        : reversed_index{100, 50, 10, 9, 8, 7, 5, 5, 4, 3, 2, 1, 0, -1, -2},
          ordered_index{-2, -1, 0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 50, 100},
          reversed_value{15., 14., 13., 12., 11., 10., 9., 7.,
                         7.,  6.,  5.,  4.,  3.,  2.,  -1.},
          ordered_value{-1., 2.,  3.,  4.,  5.,  6.,  7., 7.,
                        9.,  10., 11., 12., 13., 14., 15.}
    {}

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

TYPED_TEST_SUITE(ZipIterator, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ZipIterator, EmptyIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;

    auto test_iter = gko::detail::make_zip_iterator<index_type*, value_type*>(
        nullptr, nullptr);

    ASSERT_NO_THROW(std::sort(test_iter, test_iter));
}


TYPED_TEST(ZipIterator, SortingReversedWithIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    std::sort(test_iter, test_iter + vec1.size());

    ASSERT_EQ(vec1, this->ordered_index);
    ASSERT_EQ(vec2, this->reversed_value);
}


TYPED_TEST(ZipIterator, SortingAlreadySortedWithIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->ordered_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    std::sort(test_iter, test_iter + vec1.size());

    ASSERT_EQ(vec1, this->ordered_index);
    ASSERT_EQ(vec2, this->ordered_value);
}


TYPED_TEST(ZipIterator, IteratorReferenceOperatorSmaller)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    bool is_sorted =
        this->is_sorted_iterator(test_iter, test_iter + vec1.size());

    ASSERT_FALSE(is_sorted);
}


TYPED_TEST(ZipIterator, IteratorReferenceOperatorSmaller2)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->ordered_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    bool is_sorted =
        this->is_sorted_iterator(test_iter, test_iter + vec1.size());

    ASSERT_TRUE(is_sorted);
}


TYPED_TEST(ZipIterator, IncreasingIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    auto begin = test_iter;
    auto plus_2 = begin + 2;
    auto plus_2_rev = 2 + begin;
    auto plus_minus_2 = plus_2 - 2;
    auto increment_pre_2 = begin;
    ++increment_pre_2;
    ++increment_pre_2;
    auto increment_post_2 = begin;
    increment_post_2++;
    increment_post_2++;
    auto increment_pre_test = begin;
    auto increment_post_test = begin;

    // check results for equality
    ASSERT_TRUE(begin == plus_minus_2);
    ASSERT_TRUE(plus_2 == increment_pre_2);
    ASSERT_TRUE(plus_2_rev == increment_pre_2);
    ASSERT_TRUE(increment_pre_2 == increment_post_2);
    ASSERT_TRUE(begin == increment_post_test++);
    ASSERT_TRUE(begin + 1 == ++increment_pre_test);
    ASSERT_TRUE(std::get<0>(*plus_2) == vec1[2]);
    ASSERT_TRUE(std::get<1>(*plus_2) == vec2[2]);
    // check other comparison operators and difference
    std::vector<gko::detail::zip_iterator<index_type*, value_type*>> its{
        begin,
        plus_2,
        plus_2_rev,
        plus_minus_2,
        increment_pre_2,
        increment_post_2,
        increment_pre_test,
        increment_post_test,
        begin + 5,
        begin + 9};
    std::sort(its.begin(), its.end());
    std::vector<int> dists;
    std::vector<int> ref_dists{0, 1, 0, 1, 0, 0, 0, 3, 4};
    for (int i = 0; i < its.size() - 1; i++) {
        SCOPED_TRACE(i);
        dists.push_back(its[i + 1] - its[i]);
        auto equal = dists.back() > 0;
        ASSERT_EQ(its[i + 1] > its[i], equal);
        ASSERT_EQ(its[i] < its[i + 1], equal);
        ASSERT_EQ(its[i] != its[i + 1], equal);
        ASSERT_EQ(its[i] == its[i + 1], !equal);
        ASSERT_EQ(its[i] >= its[i + 1], !equal);
        ASSERT_EQ(its[i + 1] <= its[i], !equal);
        ASSERT_TRUE(its[i + 1] >= its[i]);
        ASSERT_TRUE(its[i] <= its[i + 1]);
    }
    ASSERT_EQ(dists, ref_dists);
}


#ifndef NDEBUG


bool check_assertion_exit_code(int exit_code)
{
#ifdef _MSC_VER
    // MSVC picks up the exit code incorrectly,
    // so we can only check that it exits
    return true;
#else
    return exit_code != 0;
#endif
}


TYPED_TEST(ZipIterator, IncompatibleIteratorDeathTest)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto it1 = gko::detail::make_zip_iterator(this->ordered_index.data(),
                                              this->ordered_value.data());
    auto it2 = gko::detail::make_zip_iterator(this->ordered_index.data() + 1,
                                              this->ordered_value.data());

    // a set of operations that return inconsistent results for the two
    // different iterators
    EXPECT_EXIT((void)(it2 - it1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it2 == it1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it2 != it1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it1 < it2), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it2 <= it1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it2 > it1), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(it1 >= it2), check_assertion_exit_code, "");
}


#endif


TYPED_TEST(ZipIterator, DecreasingIterator)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    auto iter = test_iter + 5;
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
    ASSERT_TRUE(std::get<0>(*minus_2) == vec1[3]);
    ASSERT_TRUE(std::get<1>(*minus_2) == vec2[3]);
}


TYPED_TEST(ZipIterator, CorrectDereferencing)
{
    using index_type_it = typename TestFixture::index_type;
    using value_type_it = typename TestFixture::value_type;
    std::vector<index_type_it> vec1{this->reversed_index};
    std::vector<value_type_it> vec2{this->ordered_value};
    constexpr int element_to_test = 3;

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    auto begin = test_iter;
    using value_type = typename decltype(begin)::value_type;
    auto to_test_ref = *(begin + element_to_test);
    value_type to_test_pair = to_test_ref;  // Testing implicit conversion

    ASSERT_TRUE(std::get<0>(to_test_pair) == vec1[element_to_test]);
    ASSERT_TRUE(std::get<0>(to_test_pair) == std::get<0>(to_test_ref));
    ASSERT_TRUE(std::get<1>(to_test_pair) == vec2[element_to_test]);
    ASSERT_TRUE(std::get<1>(to_test_pair) == std::get<1>(to_test_ref));
}


TYPED_TEST(ZipIterator, CorrectSwapping)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    auto first_el_reference = *test_iter;
    auto second_el_reference = *(test_iter + 1);
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


TYPED_TEST(ZipIterator, CorrectHandWrittenSwapping)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    std::vector<index_type> vec1{this->reversed_index};
    std::vector<value_type> vec2{this->ordered_value};

    auto test_iter = gko::detail::make_zip_iterator(vec1.data(), vec2.data());
    auto first_el_reference = *test_iter;
    auto second_el_reference = *(test_iter + 1);
    auto temp = static_cast<typename decltype(test_iter)::value_type>(
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


template <typename ValueType>
class PermuteIterator : public ::testing::Test {
protected:
    using value_type = ValueType;
};

TYPED_TEST_SUITE(PermuteIterator, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(PermuteIterator, EmptyIterator)
{
    auto test_iter = gko::detail::make_permute_iterator<TypeParam*>(
        nullptr, [](int i) { return i; });

    ASSERT_NO_THROW(std::sort(test_iter, test_iter));
}


TYPED_TEST(PermuteIterator, SortingWithIdentityPermutation)
{
    std::vector<TypeParam> vec{6, 2, 5, 2, 4};
    std::vector<TypeParam> sorted{2, 2, 4, 5, 6};
    auto test_iter = gko::detail::make_permute_iterator(
        vec.begin(), [](int i) { return i; });

    std::sort(test_iter, test_iter + vec.size());

    ASSERT_EQ(vec, sorted);
}


TYPED_TEST(PermuteIterator, SortingWithReversePermutation)
{
    std::vector<TypeParam> vec{6, 2, 5, 2, 4};
    std::vector<TypeParam> sorted{6, 5, 4, 2, 2};
    auto test_iter = gko::detail::make_permute_iterator(
        vec.begin(),
        [size = vec.size()](int i) { return static_cast<int>(size) - 1 - i; });

    std::sort(test_iter, test_iter + vec.size());

    ASSERT_EQ(vec, sorted);
}


TYPED_TEST(PermuteIterator, SortingWithStridedPermutation)
{
    std::vector<TypeParam> vec{6, 8, 2, 9, 5, 1, 2, 7, 4, 0};
    std::vector<TypeParam> sorted{2, 8, 2, 9, 4, 1, 5, 7, 6, 0};
    auto test_iter = gko::detail::make_permute_iterator(
        vec.begin(), [](int i) { return 2 * i; });

    std::sort(test_iter, test_iter + vec.size() / 2);

    ASSERT_EQ(vec, sorted);
}


TYPED_TEST(PermuteIterator, IncreasingIterator)
{
    std::vector<TypeParam> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto perm = [size = vec.size()](int i) {
        return static_cast<int>(size) - 1 - i;
    };

    auto test_iter = gko::detail::make_permute_iterator(vec.begin(), perm);
    auto begin = test_iter;
    auto plus_2 = begin + 2;
    auto plus_2_rev = 2 + begin;
    auto plus_minus_2 = plus_2 - 2;
    auto increment_pre_2 = begin;
    ++increment_pre_2;
    ++increment_pre_2;
    auto increment_post_2 = begin;
    increment_post_2++;
    increment_post_2++;
    auto increment_pre_test = begin;
    auto increment_post_test = begin;

    // check results for equality
    ASSERT_TRUE(begin == plus_minus_2);
    ASSERT_TRUE(plus_2 == increment_pre_2);
    ASSERT_TRUE(plus_2_rev == increment_pre_2);
    ASSERT_TRUE(increment_pre_2 == increment_post_2);
    ASSERT_TRUE(begin == increment_post_test++);
    ASSERT_TRUE(begin + 1 == ++increment_pre_test);
    ASSERT_TRUE(*plus_2 == vec[perm(2)]);
    // check other comparison operators and difference
    std::vector<gko::detail::permute_iterator<
        typename std::vector<TypeParam>::iterator, decltype(perm)>>
        its{begin,
            plus_2,
            plus_2_rev,
            plus_minus_2,
            increment_pre_2,
            increment_post_2,
            increment_pre_test,
            increment_post_test,
            begin + 5,
            begin + 9};
    std::sort(its.begin(), its.end());
    std::vector<int> dists;
    std::vector<int> ref_dists{0, 1, 0, 1, 0, 0, 0, 3, 4};
    for (int i = 0; i < its.size() - 1; i++) {
        SCOPED_TRACE(i);
        dists.push_back(its[i + 1] - its[i]);
        auto equal = dists.back() > 0;
        ASSERT_EQ(its[i + 1] > its[i], equal);
        ASSERT_EQ(its[i] < its[i + 1], equal);
        ASSERT_EQ(its[i] != its[i + 1], equal);
        ASSERT_EQ(its[i] == its[i + 1], !equal);
        ASSERT_EQ(its[i] >= its[i + 1], !equal);
        ASSERT_EQ(its[i + 1] <= its[i], !equal);
        ASSERT_TRUE(its[i + 1] >= its[i]);
        ASSERT_TRUE(its[i] <= its[i + 1]);
    }
    ASSERT_EQ(dists, ref_dists);
}


TYPED_TEST(PermuteIterator, DecreasingIterator)
{
    std::vector<TypeParam> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto perm = [size = vec.size()](int i) {
        return static_cast<int>(size) - 1 - i;
    };

    auto test_iter = gko::detail::make_permute_iterator(vec.begin(), perm);

    auto iter = test_iter + 5;
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
    ASSERT_TRUE(*minus_2 == vec[perm(3)]);
}


}  // namespace
