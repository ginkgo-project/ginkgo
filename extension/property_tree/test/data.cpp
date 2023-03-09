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

#include <memory>


#include <gtest/gtest.h>


#include <property_tree/data.hpp>


using namespace gko::extension;


TEST(Data, DataTypeIsCorrect)
{
    using namespace std::literals::string_literals;
    // Empty
    ASSERT_EQ(data_s().get_tag(), data_s::tag_type::empty_t);
    // String
    ASSERT_EQ(data_s("1").get_tag(), data_s::tag_type::str_t);
    ASSERT_EQ(data_s("1"s).get_tag(), data_s::tag_type::str_t);
    // Floating point
    ASSERT_EQ(data_s(1.23f).get_tag(), data_s::tag_type::double_t);
    ASSERT_EQ(data_s(1.23).get_tag(), data_s::tag_type::double_t);
    // Bool
    ASSERT_EQ(data_s(true).get_tag(), data_s::tag_type::bool_t);
    // Integer
    ASSERT_EQ(data_s(1).get_tag(), data_s::tag_type::int_t);
    ASSERT_EQ(data_s(1L).get_tag(), data_s::tag_type::int_t);
    ASSERT_EQ(data_s(1LL).get_tag(), data_s::tag_type::int_t);
    ASSERT_EQ(data_s(1U).get_tag(), data_s::tag_type::int_t);
    ASSERT_EQ(data_s(1UL).get_tag(), data_s::tag_type::int_t);
    ASSERT_EQ(data_s(1ULL).get_tag(), data_s::tag_type::int_t);
}


TEST(Data, DataContentIsCorrect)
{
    // String
    ASSERT_EQ(data_s("1").template get<std::string>(), "1");
    // Floating point
    ASSERT_EQ(data_s(1.23).template get<double>(), 1.23);
    // Bool
    ASSERT_EQ(data_s(true).template get<bool>(), true);
    // Integer
    ASSERT_EQ(data_s(1).template get<long long int>(), 1);
}
