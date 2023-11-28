// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/data.hpp>


#include <memory>


#include <gtest/gtest.h>


using namespace gko::config;


TEST(Data, DataTypeIsCorrect)
{
    using namespace std::literals::string_literals;
    // Empty
    ASSERT_TRUE(holds_alternative<monostate>(data()));
    // String
    ASSERT_TRUE(holds_alternative<std::string>(data("1")));
    ASSERT_TRUE(holds_alternative<std::string>(data("1"s)));
    // Floating point
    ASSERT_TRUE(holds_alternative<double>(data(1.23f)));
    ASSERT_TRUE(holds_alternative<double>(data(1.23)));
    // Bool
    ASSERT_TRUE(holds_alternative<bool>(data(true)));
    // Integer
    ASSERT_TRUE(holds_alternative<long long int>(data(1)));
    ASSERT_TRUE(holds_alternative<long long int>(data(1L)));
    ASSERT_TRUE(holds_alternative<long long int>(data(1LL)));
    ASSERT_TRUE(holds_alternative<long long int>(data(1U)));
    ASSERT_TRUE(holds_alternative<long long int>(data(1UL)));
}


TEST(Data, DataContentIsCorrect)
{
    // String
    ASSERT_EQ(get<std::string>(data("1")), "1");
    // Floating point
    ASSERT_EQ(get<double>(data(1.23)), 1.23);
    // Bool
    ASSERT_EQ(get<bool>(data(true)), true);
    // Integer
    ASSERT_EQ(get<long long int>(data(1)), 1);
}
