#include <memory>


#include <gtest/gtest.h>

#include "property_tree/data.hpp"


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