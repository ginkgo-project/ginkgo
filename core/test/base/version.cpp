// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/version.hpp>


#include <sstream>


#include <gtest/gtest.h>


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
    ASSERT_TRUE(ss.str().find("CUDA") != std::string::npos);
}


}  // namespace
