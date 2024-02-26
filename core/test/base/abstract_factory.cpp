// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/abstract_factory.hpp>


#include <gtest/gtest.h>


namespace {


struct IntFactory;
struct MyInt;


struct parameters_type
    : gko::enable_parameters_type<parameters_type, IntFactory> {
    int coefficient{5};
};


using base = gko::AbstractFactory<MyInt, int>;


struct IntFactory
    : gko::EnableDefaultFactory<IntFactory, MyInt, parameters_type, base> {
    friend class gko::enable_parameters_type<parameters_type, IntFactory>;
    friend class gko::EnablePolymorphicObject<IntFactory, base>;
    using gko::EnableDefaultFactory<IntFactory, MyInt, parameters_type,
                                    base>::EnableDefaultFactory;
};

struct MyInt : gko::log::EnableLogging<MyInt> {
    MyInt(const IntFactory* factory, int orig_value)
        : value{orig_value * factory->get_parameters().coefficient}
    {}
    int value;
};


TEST(EnableDefaultFactory, StoresParameters)
{
    auto fact = IntFactory::create().on(gko::ReferenceExecutor::create());

    ASSERT_EQ(fact->get_parameters().coefficient, 5);
}


TEST(EnableDefaultFactory, GeneratesProduct)
{
    auto fact = IntFactory::create().on(gko::ReferenceExecutor::create());

    auto prod = fact->generate(3);

    ASSERT_EQ(prod->value, 15);
}


}  // namespace
