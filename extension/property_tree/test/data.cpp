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
    ASSERT_TRUE(holds_alternative<monostate>(data_s()));
    // String
    ASSERT_TRUE(holds_alternative<std::string>(data_s("1")));
    ASSERT_TRUE(holds_alternative<std::string>(data_s("1"s)));
    // Floating point
    ASSERT_TRUE(holds_alternative<double>(data_s(1.23f)));
    ASSERT_TRUE(holds_alternative<double>(data_s(1.23)));
    // Bool
    ASSERT_TRUE(holds_alternative<bool>(data_s(true)));
    // Integer
    ASSERT_TRUE(holds_alternative<long long int>(data_s(1)));
    ASSERT_TRUE(holds_alternative<long long int>(data_s(1L)));
    ASSERT_TRUE(holds_alternative<long long int>(data_s(1LL)));
    ASSERT_TRUE(holds_alternative<long long int>(data_s(1U)));
    ASSERT_TRUE(holds_alternative<long long int>(data_s(1UL)));
    // ASSERT_THROW(data_s(1ULL));
}


TEST(Data, DataContentIsCorrect)
{
    // String
    ASSERT_EQ(get<std::string>(data_s("1")), "1");
    // Floating point
    ASSERT_EQ(get<double>(data_s(1.23)), 1.23);
    // Bool
    ASSERT_EQ(get<bool>(data_s(true)), true);
    // Integer
    ASSERT_EQ(get<long long int>(data_s(1)), 1);
}
