#include <core/base/utils.hpp>


#include <gtest/gtest.h>


namespace {


class Base {
public:
    virtual ~Base() = default;
};


class Derived : public Base {
};


class NonRelated {
    virtual ~NonRelated() = default;
};


TEST(As, ConvertsPolymorphicType)
{
    Derived d;
    Base *b = &d;
    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertIfNotRelated)
{
    Derived d;
    Base *b = &d;
    ASSERT_THROW(gko::as<NonRelated>(b), gko::NotSupported);
}


TEST(As, ConvertsConstantPolymorphicType)
{
    Derived d;
    const Base *b = &d;
    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertConstantIfNotRelated)
{
    Derived d;
    const Base *b = &d;
    ASSERT_THROW(gko::as<NonRelated>(b), gko::NotSupported);
}


}  // namespace
