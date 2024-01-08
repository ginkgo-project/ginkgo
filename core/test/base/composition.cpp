// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/composition.hpp>


#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


struct DummyOperator : public gko::EnableLinOp<DummyOperator> {
    DummyOperator(std::shared_ptr<const gko::Executor> exec,
                  gko::dim<2> size = {})
        : gko::EnableLinOp<DummyOperator>(exec, size)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}
};


template <typename T>
class Composition : public ::testing::Test {
protected:
    Composition()
        : exec{gko::ReferenceExecutor::create()},
          operators{std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1}),
                    std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 3})}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};

TYPED_TEST_SUITE(Composition, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Composition, CanBeEmpty)
{
    auto cmp = gko::Composition<TypeParam>::create(this->exec);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(cmp->get_operators().size(), 0);
}


TYPED_TEST(Composition, CanCreateFromIterators)
{
    auto cmp = gko::Composition<TypeParam>::create(begin(this->operators),
                                                   end(this->operators));

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(cmp->get_operators().size(), 2);
    ASSERT_EQ(cmp->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmp->get_operators()[1], this->operators[1]);
}


TYPED_TEST(Composition, CanCreateFromList)
{
    auto cmp = gko::Composition<TypeParam>::create(this->operators[0],
                                                   this->operators[1]);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(cmp->get_operators().size(), 2);
    ASSERT_EQ(cmp->get_operators()[0], this->operators[0]);
    ASSERT_EQ(cmp->get_operators()[1], this->operators[1]);
}


}  // namespace
