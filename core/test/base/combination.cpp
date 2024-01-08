// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/combination.hpp>


#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


struct DummyOperator : public gko::EnableLinOp<DummyOperator> {
    DummyOperator(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyOperator>(exec, gko::dim<2>{1, 1})
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}
};


template <typename T>
class Combination : public ::testing::Test {
protected:
    Combination()
        : exec{gko::ReferenceExecutor::create()},
          operators{std::make_shared<DummyOperator>(exec),
                    std::make_shared<DummyOperator>(exec)},
          coefficients{std::make_shared<DummyOperator>(exec),
                       std::make_shared<DummyOperator>(exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
};

TYPED_TEST_SUITE(Combination, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Combination, CanBeEmpty)
{
    auto cmb = gko::Combination<TypeParam>::create(this->exec);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
    ASSERT_EQ(cmb->get_operators().size(), 0);
}


TYPED_TEST(Combination, CanCreateFromIterators)
{
    auto cmb = gko::Combination<TypeParam>::create(
        begin(this->coefficients), end(this->coefficients),
        begin(this->operators), end(this->operators));

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(1, 1));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);
    ASSERT_EQ(cmb->get_coefficients()[0], this->coefficients[0]);
    ASSERT_EQ(cmb->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmb->get_coefficients()[1], this->coefficients[1]);
    ASSERT_EQ(cmb->get_operators()[1], this->operators[1]);
}


TYPED_TEST(Combination, CanCreateFromList)
{
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);

    ASSERT_EQ(cmb->get_size(), gko::dim<2>(1, 1));
    ASSERT_EQ(cmb->get_coefficients().size(), 2);
    ASSERT_EQ(cmb->get_operators().size(), 2);
    ASSERT_EQ(cmb->get_coefficients()[0], this->coefficients[0]);
    ASSERT_EQ(cmb->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmb->get_coefficients()[1], this->coefficients[1]);
    ASSERT_EQ(cmb->get_operators()[1], this->operators[1]);
}


}  // namespace
