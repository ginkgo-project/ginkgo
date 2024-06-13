// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/perturbation.hpp>


#include <memory>


#include <gtest/gtest.h>


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


struct TransposableDummyOperator
    : public gko::EnableLinOp<TransposableDummyOperator>,
      public gko::Transposable {
    TransposableDummyOperator(std::shared_ptr<const gko::Executor> exec,
                              gko::dim<2> size = {})
        : gko::EnableLinOp<TransposableDummyOperator>(exec, size)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override {}

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

    std::unique_ptr<LinOp> transpose() const override
    {
        auto result = std::unique_ptr<TransposableDummyOperator>(
            new TransposableDummyOperator(this->get_executor(),
                                          gko::transpose(this->get_size())));
        return std::move(result);
    }

    std::unique_ptr<LinOp> conj_transpose() const override
    {
        auto result = this->transpose();
        return std::move(result);
    }
};


class Perturbation : public ::testing::Test {
protected:
    Perturbation()
        : exec{gko::ReferenceExecutor::create()},
          basis{std::make_shared<DummyOperator>(exec, gko::dim<2>{2, 1})},
          projector{std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 2})},
          trans_basis{std::make_shared<TransposableDummyOperator>(
              exec, gko::dim<2>{3, 1})},
          scalar{std::make_shared<DummyOperator>(exec, gko::dim<2>{1, 1})}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::LinOp> basis;
    std::shared_ptr<gko::LinOp> projector;
    std::shared_ptr<gko::LinOp> trans_basis;
    std::shared_ptr<gko::LinOp> scalar;
};


TEST_F(Perturbation, CanBeEmpty)
{
    auto cmp = gko::Perturbation<>::create(exec);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(0, 0));
}


TEST_F(Perturbation, CanCreateFromTwoOperators)
{
    auto cmp = gko::Perturbation<>::create(scalar, basis, projector);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(2, 2));
    ASSERT_EQ(cmp->get_basis(), basis);
    ASSERT_EQ(cmp->get_projector(), projector);
    ASSERT_EQ(cmp->get_scalar(), scalar);
}


TEST_F(Perturbation, CannotCreateFromOneNonTransposableOperator)
{
    ASSERT_THROW(gko::Perturbation<>::create(scalar, basis), gko::NotSupported);
}


TEST_F(Perturbation, CanCreateFromOneTranposableOperator)
{
    auto cmp = gko::Perturbation<>::create(scalar, trans_basis);

    ASSERT_EQ(cmp->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(cmp->get_basis(), trans_basis);
    ASSERT_EQ(cmp->get_projector()->get_size(), gko::dim<2>(1, 3));
    ASSERT_EQ(cmp->get_scalar(), scalar);
}


}  // namespace
