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

#include <gtest/gtest.h>


#include "file_config/base/template_helper.hpp"


namespace {


using namespace gko::extensions::file_config;


TEST(TemplateHelper, RemoveSpace)
{
    ASSERT_EQ(remove_space("class < type1, type2>   "), "class<type1,type2>");
    ASSERT_EQ(remove_space("std : : class< gko ::type1<A, B>, type2>   "),
              "std::class<gko::type1<A,B>,type2>");
}


TEST(TemplateHelper, GetBaseClass)
{
    ASSERT_EQ(get_base_class("class < type1, type2>   "), "class");
    ASSERT_EQ(get_base_class("std : : class< gko ::type1<A, B>, type2>   "),
              "std::class");
}


TEST(TemplateHelper, GetBaseTemplate)
{
    ASSERT_EQ(get_base_template("class < type1, type2>   "), "type1,type2");
    ASSERT_EQ(get_base_template("std : : class< gko ::type1<A, B>, type2>   "),
              "gko::type1<A,B>,type2");
}


TEST(TemplateHelper, CombineTemplate)
{
    // use base template first
    ASSERT_EQ(combine_template("double,int", "float,int32"), "double,int");
    // if the base contains empty or missing the rest, it will use the type
    // template.
    ASSERT_EQ(combine_template("base1,,base3", "type1,type2,type3,type4"),
              "base1,type2,base3,type4");
    // split the template correctly
    // if it is wrong, the first will be base<A | <B | C>> | | base3 and type1 |
    // type<I | II> | type3
    ASSERT_EQ(
        combine_template("base<A,<B,C>>,,base3", "type1,type<I,II>,type3"),
        "base<A,<B,C>>,type<I,II>,base3");
}


TEST(TemplateHelper, CombineTemplateReportError)
{
#ifndef NDEBUG
    GTEST_SKIP() << "check the full stderr output only when nodebug mode";
#endif
    // if both contain empty template, it will report them to stderr
    testing::internal::CaptureStderr();

    ASSERT_EQ(combine_template("A,,C<>", "A,,T<>,"), "A,,C<>,");
    ASSERT_EQ(testing::internal::GetCapturedStderr(),
              "The 1-th (0-based) template parameter is empty\nThe 3-th "
              "(0-based) template parameter is empty\n");
}


TEST(TemplateHelperDeath, CombineTemplateReportError)
{
#ifdef NDEBUG
    GTEST_SKIP() << "It only introduces failure when in debug mode";
#endif
    ASSERT_DEBUG_DEATH({ combine_template("A,,C<>", "A,,T<>,"); },
                       "The 1-th \\(0-based\\) template parameter is empty");
}


}  // namespace
