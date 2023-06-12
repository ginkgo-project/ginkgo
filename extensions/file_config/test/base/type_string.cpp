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

#include <complex>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/ginkgo.hpp>


#include "file_config/base/type_string.hpp"


namespace {


using namespace gko::extensions::file_config;


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
    ASSERT_EQ(get_string(type_list<double, int>{}), "double,int");
    ASSERT_EQ(get_string(type_list<int, double>{}), "int,double");
}


TEST(GetString, GetStringFromComplex)
{
    ASSERT_EQ(get_string<std::complex<float>>(), "complex<float>");
    ASSERT_EQ(get_string<std::complex<double>>(), "complex<double>");
}


TEST(GetString, GetStringFromBase)
{
    ASSERT_EQ(get_string<gko::solver::LowerTrs<>>(), "LowerTrs<double,int>");
    ASSERT_EQ(get_string<gko::solver::UpperTrs<>>(), "UpperTrs<double,int>");
}


}  // namespace
