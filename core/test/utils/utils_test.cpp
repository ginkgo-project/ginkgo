// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


using i_type = std::integral_constant<int, 42>;
using t_type = std::tuple<int>;

using testing_types1 = testing::Types<double>;
using testing_types2 = testing::Types<t_type, int>;
using testing_types3 = testing::Types<i_type, short, float>;
using testing_empty = testing::Types<>;

using tuple_types1 = std::tuple<double>;
using tuple_types2 = std::tuple<t_type, int>;
using tuple_types3 = std::tuple<i_type, short, float>;
using tuple_empty = std::tuple<>;

template <typename... Args>
struct type_list {};


TEST(TypeListHelper, ChangeOuterWrapperPredefined)
{
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<std::tuple, testing_types1>,
        tuple_types1>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<std::tuple, testing_types2>,
        tuple_types2>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<std::tuple, testing_types3>,
        tuple_types3>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<testing::Types, tuple_types1>,
        testing_types1>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<testing::Types, tuple_types2>,
        testing_types2>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<testing::Types, tuple_types3>,
        testing_types3>();
}


TEST(TypeListHelper, ChangeOuterWrapperCustomType)
{
    using type_list1 = type_list<i_type, t_type, double>;
    using testing_list1 = testing::Types<i_type, t_type, double>;

    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<testing::Types, type_list1>,
        testing_list1>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<type_list, testing_list1>,
        type_list1>();
}


TEST(TypeListHelper, ChangeOuterWrapperEmpty)
{
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<testing::Types, tuple_empty>,
        testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::change_outer_wrapper_t<std::tuple, testing_empty>,
        tuple_empty>();
}


TEST(TypeListHelper, AddInternalWrapperTuple)
{
    using expected_iw1 = testing::Types<std::tuple<i_type>, std::tuple<short>,
                                        std::tuple<float>>;

    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::tuple, testing_types3>,
        expected_iw1>();
    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::tuple, tuple_types3>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_iw1>>();
}


TEST(TypeListHelper, AddInternalWrapperComplex)
{
    using expected_iw2 = testing::Types<std::complex<double>>;

    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::complex, testing_types1>,
        expected_iw2>();
    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::complex, tuple_types1>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_iw2>>();
}


TEST(TypeListHelper, AddInternalWrapperEmpty)
{
    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::tuple, testing_empty>,
        testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::add_inner_wrapper_t<std::tuple, tuple_empty>, tuple_empty>();
}


TEST(TypeListHelper, MergeTypeListLarge)
{
    using expected_m1 = testing::Types<i_type, short, float, t_type, int>;

    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<testing_types3, testing_types2>,
        expected_m1>();
    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<tuple_types3, tuple_types2>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_m1>>();
}


TEST(TypeListHelper, MergeTypeListEmpty)
{
    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<testing_types1, testing::Types<>>,
        testing_types1>();
    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<tuple_types1, std::tuple<>>,
        tuple_types1>();
    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<testing::Types<>, testing_types1>,
        testing_types1>();
    testing::StaticAssertTypeEq<
        gko::test::merge_type_list_t<std::tuple<>, tuple_types1>,
        tuple_types1>();
}


TEST(TypeListHelper, CartesianTypeProductLarge)
{
    using expected_c1 =
        testing::Types<std::tuple<t_type, i_type>, std::tuple<t_type, short>,
                       std::tuple<t_type, float>, std::tuple<int, i_type>,
                       std::tuple<int, short>, std::tuple<int, float>>;

    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<testing_types2, testing_types3>,
        expected_c1>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<tuple_types2, tuple_types3>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_c1>>();
}


TEST(TypeListHelper, CartesianTypeProductSmall)
{
    using expected_c2 =
        testing::Types<std::tuple<double, t_type>, std::tuple<double, int>>;

    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<testing_types1, testing_types2>,
        expected_c2>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<tuple_types1, tuple_types2>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_c2>>();
}


TEST(TypeListHelper, CartesianTypeProductEmpty)
{
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<testing_empty, testing_types2>,
        testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<testing_types1, testing_empty>,
        testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<testing_empty, testing_empty>,
        testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<tuple_empty, tuple_types2>,
        tuple_empty>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<tuple_types1, tuple_empty>,
        tuple_empty>();
    testing::StaticAssertTypeEq<
        gko::test::cartesian_type_product_t<tuple_empty, tuple_empty>,
        tuple_empty>();
}


TEST(TypeListHelper, AddToCartesianTypeProductLarge)
{
    using list1 =
        testing::Types<std::tuple<double, int>, std::tuple<double, short>,
                       std::tuple<float, int>, std::tuple<float, short>>;
    using list2 = testing::Types<long, char>;
    using tlist1 = gko::test::change_outer_wrapper_t<std::tuple, list1>;
    using tlist2 = std::tuple<long, char>;
    using expected_a1 = testing::Types<
        std::tuple<double, int, long>, std::tuple<double, int, char>,
        std::tuple<double, short, long>, std::tuple<double, short, char>,
        std::tuple<float, int, long>, std::tuple<float, int, char>,
        std::tuple<float, short, long>, std::tuple<float, short, char>>;

    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<list1, list2>,
        expected_a1>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<tlist1, tlist2>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_a1>>();
}


TEST(TypeListHelper, AddToCartesianTypeProductSmall)
{
    using list3 = testing::Types<std::tuple<long>>;
    using list4 = testing::Types<double>;
    using tlist3 = std::tuple<std::tuple<long>>;
    using tlist4 = std::tuple<double>;
    using expected_a2 = testing::Types<std::tuple<long, double>>;

    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<list3, list4>,
        expected_a2>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<tlist3, tlist4>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_a2>>();
}


TEST(TypeListHelper, AddToCartesianTypeProductEmpty)
{
    using list3 = testing::Types<std::tuple<long>>;
    using tlist3 = std::tuple<std::tuple<long>>;

    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<list3, testing_empty>,
        testing_empty>();
    testing::StaticAssertTypeEq<gko::test::add_to_cartesian_type_product_t<
                                    testing_empty, testing_types1>,
                                testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<tlist3, tuple_empty>,
        tuple_empty>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_t<tuple_empty, tuple_types1>,
        tuple_empty>();
}


TEST(TypeListHelper, AddToCartesianTypeProductLeftLarge)
{
    using list1 = testing::Types<long, char>;
    using list2 =
        testing::Types<std::tuple<double, int>, std::tuple<double, short>,
                       std::tuple<float, int>, std::tuple<float, short>>;
    using tlist1 = std::tuple<long, char>;
    using tlist2 = gko::test::change_outer_wrapper_t<std::tuple, list2>;
    using expected_a1 = testing::Types<
        std::tuple<long, double, int>, std::tuple<char, double, int>,
        std::tuple<long, double, short>, std::tuple<char, double, short>,
        std::tuple<long, float, int>, std::tuple<char, float, int>,
        std::tuple<long, float, short>, std::tuple<char, float, short>>;

    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<list1, list2>,
        expected_a1>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<tlist1, tlist2>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_a1>>();
}


TEST(TypeListHelper, AddToCartesianTypeProductLeftSmall)
{
    using list3 = testing::Types<double>;
    using list4 = testing::Types<std::tuple<long>>;
    using tlist3 = std::tuple<double>;
    using tlist4 = std::tuple<std::tuple<long>>;
    using expected_a2 = testing::Types<std::tuple<double, long>>;

    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<list3, list4>,
        expected_a2>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<tlist3, tlist4>,
        gko::test::change_outer_wrapper_t<std::tuple, expected_a2>>();
}


TEST(TypeListHelper, AddToCartesianTypeProductLeftEmpty)
{
    using list3 = testing::Types<std::tuple<long>>;
    using tlist3 = std::tuple<std::tuple<long>>;

    testing::StaticAssertTypeEq<gko::test::add_to_cartesian_type_product_left_t<
                                    testing_types1, testing_empty>,
                                testing_empty>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<testing_empty, list3>,
        testing_empty>();
    testing::StaticAssertTypeEq<gko::test::add_to_cartesian_type_product_left_t<
                                    tuple_types1, tuple_empty>,
                                tuple_empty>();
    testing::StaticAssertTypeEq<
        gko::test::add_to_cartesian_type_product_left_t<tuple_empty, tlist3>,
        tuple_empty>();
}


}  // namespace
