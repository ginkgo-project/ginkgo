/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/executor.hpp>


#include <thread>
#include <type_traits>


#if defined(__unix__) || defined(__APPLE__)
#include <utmpx.h>
#endif


#include <gtest/gtest.h>
#include "rapidjson/document.h"

#include <resource_manager/base/element_types.hpp>


namespace {

using namespace gko::extension::resource_manager;

TEST(GetString, GetStringFromTemplate)
{
    ASSERT_EQ(get_string<int>(), "int");
    ASSERT_EQ(get_string<gko::int64>(), "int64");
    ASSERT_EQ(get_string<double>(), "double");
    ASSERT_EQ(get_string<float>(), "float");
}


TEST(GetString, GetStringFromTypeList)
{
    ASSERT_EQ(get_string(int{}), "int");
    ASSERT_EQ(get_string(double{}), "double");
    ASSERT_EQ(get_string(type_list<double, int>{}), "double+int");
    ASSERT_EQ(get_string(type_list<int, double>{}), "int+double");
}


template <typename K>
struct DummyType {
    using ktype = K;
    struct Factory {
        using rev_kype = K;
    };
};

template <typename K, typename T>
struct DummyType2 {
    using ktype = K;
    using ttype = T;
    struct Factory {
        using rev_ktype = T;
        using rev_ttype = K;
    };
};


TEST(GetType, ApplyOneTemplate)
{
    using type = typename get_the_type<DummyType, double>::type;
    using type2 = typename get_the_type<DummyType, type_list<double>>::type;
    using factory = typename get_the_factory_type<DummyType, double>::type;
    using factory2 =
        typename get_the_factory_type<DummyType, type_list<double>>::type;
    using ref_type = DummyType<double>;
    using ref_factory = typename ref_type::Factory;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<factory, ref_factory>::value));
    ASSERT_TRUE((std::is_same<type2, ref_type>::value));
    ASSERT_TRUE((std::is_same<factory2, ref_factory>::value));
}


TEST(GetType, ApplyTwoTemplate)
{
    using type =
        typename get_the_type<DummyType2, type_list<double, int>>::type;
    using factory =
        typename get_the_factory_type<DummyType2, type_list<double, int>>::type;
    using ref_type = DummyType2<double, int>;
    using ref_factory = typename ref_type::Factory;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<factory, ref_factory>::value));
}


TEST(Concatenate, TwoType)
{
    using type = typename concatenate<double, int>::type;
    using both_side = typename concatenate<tt_list<double>, tt_list<int>>::type;
    using left_side = typename concatenate<tt_list<double>, int>::type;
    using right_side = typename concatenate<double, tt_list<int>>::type;
    using ref_type = tt_list<double, int>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
}

TEST(Concatenate, OneSideMultipleTypes)
{
    using left_side = typename concatenate<tt_list<double, float>, int>::type;
    using right_side = typename concatenate<double, tt_list<float, int>>::type;
    using both_sidel =
        typename concatenate<tt_list<double, float>, tt_list<int>>::type;
    using both_sider =
        typename concatenate<tt_list<double>, tt_list<float, int>>::type;
    using ref_type = tt_list<double, float, int>;

    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sidel, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sider, ref_type>::value));
}


TEST(Concatenate, BothSideMultipleTypes)
{
    using type = typename concatenate<tt_list<double, float>,
                                      tt_list<int, gko::int64>>::type;
    using ref_type = tt_list<double, float, int, gko::int64>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
}


TEST(Concat, TwoType)
{
    using type = typename concat<double, int>::type;
    using both_side = typename concat<type_list<double>, type_list<int>>::type;
    using left_side = typename concat<type_list<double>, int>::type;
    using right_side = typename concat<double, type_list<int>>::type;
    using ref_type = type_list<double, int>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
}

TEST(Concat, OneSideMultipleTypes)
{
    using left_side = typename concat<type_list<double, float>, int>::type;
    using right_side = typename concat<double, type_list<float, int>>::type;
    using both_sidel =
        typename concat<type_list<double, float>, type_list<int>>::type;
    using both_sider =
        typename concat<type_list<double>, type_list<float, int>>::type;
    using ref_type = type_list<double, float, int>;

    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sidel, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sider, ref_type>::value));
}


TEST(Concat, BothSideMultipleTypes)
{
    using type = typename concat<type_list<double, float>,
                                 type_list<int, gko::int64>>::type;
    using ref_type = type_list<double, float, int, gko::int64>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
}


TEST(Span, TwoType)
{
    using type = typename span<double, int>::type;
    using both_side = typename span<tt_list<double>, tt_list<int>>::type;
    using left_side = typename span<tt_list<double>, int>::type;
    using right_side = typename span<double, tt_list<int>>::type;
    using ref_type = tt_list<type_list<double, int>>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
}

TEST(Span, LeftSideMultipleTypes)
{
    using left_side = typename span<tt_list<double, float, int>, int>::type;
    using both_sidel =
        typename span<tt_list<double, float, int>, tt_list<int>>::type;
    using ref_type = tt_list<type_list<double, int>, type_list<float, int>,
                             type_list<int, int>>;

    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sidel, ref_type>::value));
}

TEST(Span, RightSideMultipleTypes)
{
    using right_side = typename span<double, tt_list<double, float, int>>::type;
    using both_sider =
        typename span<tt_list<double>, tt_list<double, float, int>>::type;
    using ref_type = tt_list<type_list<double, double>,
                             type_list<double, float>, type_list<double, int>>;

    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sider, ref_type>::value));
}


TEST(Span, BothSideMultipleTypes)
{
    using type = typename span<tt_list<double, float, int>,
                               tt_list<int, gko::int64, double>>::type;
    using ref_type =
        tt_list<type_list<double, int>, type_list<double, gko::int64>,
                type_list<double, double>, type_list<float, int>,
                type_list<float, gko::int64>, type_list<float, double>,
                type_list<int, int>, type_list<int, gko::int64>,
                type_list<int, double>>;


    ASSERT_TRUE((std::is_same<type, ref_type>::value));
}


TEST(SpanList, TwoType)
{
    using type = typename span_list<double, int>::type;
    using both_side = typename span_list<tt_list<double>, tt_list<int>>::type;
    using left_side = typename span_list<tt_list<double>, int>::type;
    using right_side = typename span_list<double, tt_list<int>>::type;
    using ref_type = tt_list<type_list<double, int>>;

    ASSERT_TRUE((std::is_same<type, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
}

TEST(SpanList, LeftSideMultipleTypes)
{
    using left_side =
        typename span_list<tt_list<double, float, int>, int>::type;
    using both_sidel =
        typename span_list<tt_list<double, float, int>, tt_list<int>>::type;
    using ref_type = tt_list<type_list<double, int>, type_list<float, int>,
                             type_list<int, int>>;

    ASSERT_TRUE((std::is_same<left_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sidel, ref_type>::value));
}

TEST(SpanList, RightSideMultipleTypes)
{
    using right_side =
        typename span_list<double, tt_list<double, float, int>>::type;
    using both_sider =
        typename span_list<tt_list<double>, tt_list<double, float, int>>::type;
    using ref_type = tt_list<type_list<double, double>,
                             type_list<double, float>, type_list<double, int>>;

    ASSERT_TRUE((std::is_same<right_side, ref_type>::value));
    ASSERT_TRUE((std::is_same<both_sider, ref_type>::value));
}


TEST(SpanList, BothSideMultipleTypes)
{
    using type = typename span_list<tt_list<double, float, int>,
                                    tt_list<int, gko::int64, double>>::type;
    using ref_type =
        tt_list<type_list<double, int>, type_list<double, gko::int64>,
                type_list<double, double>, type_list<float, int>,
                type_list<float, gko::int64>, type_list<float, double>,
                type_list<int, int>, type_list<int, gko::int64>,
                type_list<int, double>>;


    ASSERT_TRUE((std::is_same<type, ref_type>::value));
}

TEST(SpanList, ThreeType)
{
    using type =
        typename span_list<tt_list<double, float>, tt_list<int, gko::int64>,
                           tt_list<float, double>>::type;
    using ref_type =
        tt_list<type_list<double, int, float>, type_list<double, int, double>,
                type_list<double, gko::int64, float>,
                type_list<double, gko::int64, double>,
                type_list<float, int, float>, type_list<float, int, double>,
                type_list<float, gko::int64, float>,
                type_list<float, gko::int64, double>>;


    ASSERT_TRUE((std::is_same<type, ref_type>::value));
}


}  // namespace
