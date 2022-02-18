/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/synthesizer/containers.hpp>


#include <gtest/gtest.h>

namespace {


struct IntegerSequenceExtensions : public ::testing::Test {
    using range1 = gko::syn::as_list<gko::syn::range<1, 6>>;
    using range1_exp = std::integer_sequence<int, 1, 2, 3, 4, 5>;
    using range2 = gko::syn::as_list<gko::syn::range<0, 7, 2>>;
    using range2_exp = std::integer_sequence<int, 0, 2, 4, 6>;

    using test = gko::syn::value_list<int, 0, 7614, 453, 16, 9, 16, 0, 0, -4>;
    using sorted_asc_dups =
        std::integer_sequence<int, -4, 0, 0, 0, 9, 16, 16, 453, 7614>;
    using sorted_asc_nodups =
        std::integer_sequence<int, -4, 0, 9, 16, 453, 7614>;
    using sorted_desc_dups =
        std::integer_sequence<int, 7614, 453, 16, 16, 9, 0, 0, 0, -4>;
    using sorted_desc_nodups =
        std::integer_sequence<int, 7614, 453, 16, 9, 0, -4>;

    using empty = std::integer_sequence<int>;

protected:
    IntegerSequenceExtensions() {}
};

using ::testing::StaticAssertTypeEq;


TEST_F(IntegerSequenceExtensions, CanCreateRanges)
{
    StaticAssertTypeEq<range1, range1_exp>();
    StaticAssertTypeEq<range2, range2_exp>();
}


TEST_F(IntegerSequenceExtensions, CanConcatenate)
{
    using test_case =
        gko::syn::concatenate<range1, range2, std::integer_sequence<int, 7>>;
    using expected = std::integer_sequence<int, 1, 2, 3, 4, 5, 0, 2, 4, 6, 7>;

    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanGetArrayAndValue)
{
    auto expected_array = std::array<int, 9>{0, 7614, 453, 16, 9, 16, 0, 0, -4};
    using front = gko::syn::front<test>;

    ASSERT_EQ(gko::syn::as_array(test{}), expected_array);
    ASSERT_EQ(gko::syn::as_array(front{})[0], 0);
    ASSERT_EQ(gko::syn::as_value(front{}), 0);
}


TEST_F(IntegerSequenceExtensions, CanSortAscendingNoDuplicates)
{
    using sorted = gko::syn::sort<true, test>;

    StaticAssertTypeEq<sorted, sorted_asc_nodups>();
}


TEST_F(IntegerSequenceExtensions, CanSortDescendingNoDuplicates)
{
    using sorted = gko::syn::sort<false, test>;

    StaticAssertTypeEq<sorted, sorted_desc_nodups>();
}


TEST_F(IntegerSequenceExtensions, CanSortAscendingWithDuplicates)
{
    using sorted = gko::syn::sort_keep<true, test>;

    StaticAssertTypeEq<sorted, sorted_asc_dups>();
}


TEST_F(IntegerSequenceExtensions, CanSortDescendingWithDuplicates)
{
    using sorted = gko::syn::sort_keep<false, test>;

    StaticAssertTypeEq<sorted, sorted_desc_dups>();
}


TEST_F(IntegerSequenceExtensions, CanAccessMax)
{
    using test_case = gko::syn::max<test>;
    using expected = std::integer_sequence<int, 7614>;
    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanAccessMin)
{
    using test_case = gko::syn::min<test>;
    using expected = std::integer_sequence<int, -4>;
    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanAccessMedian)
{
    using test_case = gko::syn::median<test>;
    using expected = std::integer_sequence<int, 16>;

    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanAccessFront)
{
    using test_case = gko::syn::front<test>;
    using expected = std::integer_sequence<int, 0>;

    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanAccessBack)
{
    using test_case = gko::syn::back<test>;
    using expected = std::integer_sequence<int, -4>;

    StaticAssertTypeEq<test_case, expected>();
}


TEST_F(IntegerSequenceExtensions, CanUseAtIndex)
{
    using idx1 = gko::syn::at_index<1, test>;
    using exp_idx1 = std::integer_sequence<int, 7614>;
    using idx2 = gko::syn::at_index<2, test>;
    using exp_idx2 = std::integer_sequence<int, 453>;
    using idx4 = gko::syn::at_index<4, test>;
    using exp_idx4 = std::integer_sequence<int, 9>;
    using idx6 = gko::syn::at_index<6, test>;
    using exp_idx6 = std::integer_sequence<int, 0>;

    StaticAssertTypeEq<idx1, exp_idx1>();
    StaticAssertTypeEq<idx2, exp_idx2>();
    StaticAssertTypeEq<idx4, exp_idx4>();
    StaticAssertTypeEq<idx6, exp_idx6>();
}


TEST_F(IntegerSequenceExtensions, EmptyTests)
{
    using test1 = gko::syn::at_index<72, empty>;
    using test2 = gko::syn::back<empty>;
    using test3 = gko::syn::front<empty>;
    using test4 = gko::syn::median<empty>;
    using test5 = gko::syn::min<empty>;
    using test6 = gko::syn::sort<true, empty>;
    using test7 = gko::syn::sort<false, empty>;
    using test8 = gko::syn::sort_keep<true, empty>;
    using test9 = gko::syn::sort_keep<false, empty>;

    StaticAssertTypeEq<test1, empty>();
    StaticAssertTypeEq<test2, empty>();
    StaticAssertTypeEq<test3, empty>();
    StaticAssertTypeEq<test4, empty>();
    StaticAssertTypeEq<test5, empty>();
    StaticAssertTypeEq<test6, empty>();
    StaticAssertTypeEq<test7, empty>();
    StaticAssertTypeEq<test8, empty>();
    StaticAssertTypeEq<test9, empty>();
}

struct int_encoder {
    using can_encode = std::true_type;

    static constexpr int encode() { return 1; }

    template <typename... Rest>
    static constexpr int encode(int v1, Rest&&... rest)
    {
        return v1 * encode(std::forward<Rest>(rest)...);
    }
};


TEST_F(IntegerSequenceExtensions, CanMergeEmptyList)
{
    StaticAssertTypeEq<gko::syn::merge<int_encoder, empty, empty>, empty>();
}


TEST_F(IntegerSequenceExtensions, CanMergeOneList)
{
    StaticAssertTypeEq<gko::syn::merge<int_encoder, test, empty>, test>();
}


TEST_F(IntegerSequenceExtensions, CanMergeTwoLists)
{
    using list1 = std::integer_sequence<int, 2, 3>;
    using list2 = std::integer_sequence<int, 4, 8>;
    using expected = std::integer_sequence<int, 8, 16, 12, 24>;

    using res = gko::syn::merge<int_encoder, list1, list2>;

    StaticAssertTypeEq<res, expected>();
}


TEST_F(IntegerSequenceExtensions, CanMergeThreeLists)
{
    using list1 = std::integer_sequence<int, 2, 3>;
    using list2 = std::integer_sequence<int, 4, 8>;
    using list3 = std::integer_sequence<int, 2, 3>;
    using expected1 = std::integer_sequence<int, 8, 16, 12, 24>;
    using expected2 =
        std::integer_sequence<int, 16, 32, 24, 48, 24, 48, 36, 72>;

    using res1 = gko::syn::merge<int_encoder, list1, list2>;
    using res2 = gko::syn::merge<int_encoder, list3, res1>;

    StaticAssertTypeEq<res1, expected1>();
    StaticAssertTypeEq<res2, expected2>();
}


TEST_F(IntegerSequenceExtensions, CanMergeThreeListsOtherOrder)
{
    using list1 = std::integer_sequence<int, 2, 3>;
    using list2 = std::integer_sequence<int, 4, 8>;
    using list3 = std::integer_sequence<int, 2, 3>;
    using expected1 = std::integer_sequence<int, 8, 16, 12, 24>;
    using expected2 =
        std::integer_sequence<int, 16, 24, 32, 48, 24, 36, 48, 72>;

    using res1 = gko::syn::merge<int_encoder, list1, list2>;
    using res2 = gko::syn::merge<int_encoder, res1, list3>;

    StaticAssertTypeEq<res1, expected1>();
    StaticAssertTypeEq<res2, expected2>();
}

}  // namespace
