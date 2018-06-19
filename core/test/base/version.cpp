/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/version.hpp>


#include <gtest/gtest.h>


#include <sstream>


namespace {


TEST(Version, ConstructsVersionInformation)
{
    auto tag = "test";
    gko::version v{1, 2, 3, tag};

    ASSERT_EQ(v.major, 1);
    ASSERT_EQ(v.minor, 2);
    ASSERT_EQ(v.patch, 3);
    ASSERT_EQ(v.tag, tag);
}


TEST(Version, ComparesUsingLessThan)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_TRUE(v1 < v2);
}


TEST(Version, ComparesUsingLessOrEqual)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_TRUE(v1 <= v2);
}


TEST(Version, ComparesUsingEqual)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_FALSE(v1 == v2);
}


TEST(Version, ComparesUsingNotEqual)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_TRUE(v1 != v2);
}


TEST(Version, ComparesUsingGreaterThan)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_FALSE(v1 > v2);
}


TEST(Version, ComparesUsingGreaterOrEqual)
{
    gko::version v1{1, 2, 3, ""};
    gko::version v2{1, 3, 2, ""};

    ASSERT_FALSE(v1 >= v2);
}


TEST(Version, PrintsVersionToStream)
{
    std::stringstream ss;

    ss << gko::version{1, 2, 3, "test"};

    ASSERT_EQ(ss.str(), "1.2.3 (test)");
}


TEST(VersionInfo, ReturnsVersionInformation)
{
    auto info = gko::version_info::get();

    ASSERT_EQ(info.header_version, info.core_version);
}


TEST(VersionInfo, WritesInfoToStream)
{
    std::stringstream ss;

    ss << gko::version_info::get();

    ASSERT_TRUE(ss.str().find("Ginkgo") != std::string::npos);
    ASSERT_TRUE(ss.str().find("reference") != std::string::npos);
    ASSERT_TRUE(ss.str().find("OpenMP") != std::string::npos);
    ASSERT_TRUE(ss.str().find("GPU") != std::string::npos);
}


}  // namespace
