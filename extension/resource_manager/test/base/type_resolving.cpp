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

#include <type_traits>


#include <gtest/gtest.h>


#include "resource_manager/base/type_resolving.hpp"


namespace {


using namespace gko::extension::resource_manager;


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


}  // namespace
