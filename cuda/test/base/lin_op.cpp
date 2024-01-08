// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>


#include "cuda/test/utils.hpp"


namespace {


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

    void access() const { last_access = this->get_executor(); }

    mutable std::shared_ptr<const gko::Executor> last_access;
    mutable std::shared_ptr<const gko::Executor> last_b_access;
    mutable std::shared_ptr<const gko::Executor> last_x_access;
    mutable std::shared_ptr<const gko::Executor> last_alpha_access;
    mutable std::shared_ptr<const gko::Executor> last_beta_access;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        this->access();
        static_cast<const DummyLinOp*>(b)->access();
        static_cast<const DummyLinOp*>(x)->access();
        last_b_access = b->get_executor();
        last_x_access = x->get_executor();
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        this->access();
        static_cast<const DummyLinOp*>(alpha)->access();
        static_cast<const DummyLinOp*>(b)->access();
        static_cast<const DummyLinOp*>(beta)->access();
        static_cast<const DummyLinOp*>(x)->access();
        last_alpha_access = alpha->get_executor();
        last_b_access = b->get_executor();
        last_beta_access = beta->get_executor();
        last_x_access = x->get_executor();
    }
};


class EnableLinOp : public CudaTestFixture {
protected:
    EnableLinOp()
        : op{DummyLinOp::create(exec, gko::dim<2>{3, 5})},
          alpha{DummyLinOp::create(ref, gko::dim<2>{1})},
          beta{DummyLinOp::create(ref, gko::dim<2>{1})},
          b{DummyLinOp::create(ref, gko::dim<2>{5, 4})},
          x{DummyLinOp::create(ref, gko::dim<2>{3, 4})}
    {}

    std::unique_ptr<DummyLinOp> op;
    std::unique_ptr<DummyLinOp> alpha;
    std::unique_ptr<DummyLinOp> beta;
    std::unique_ptr<DummyLinOp> b;
    std::unique_ptr<DummyLinOp> x;
};


TEST_F(EnableLinOp, ApplyCopiesDataToCorrectExecutor)
{
    op->apply(b, x);

    ASSERT_EQ(op->last_b_access, exec);
    ASSERT_EQ(op->last_x_access, exec);
}


TEST_F(EnableLinOp, ApplyCopiesBackOnlyX)
{
    op->apply(b, x);

    ASSERT_EQ(b->last_access, nullptr);
    ASSERT_EQ(x->last_access, exec);
}


TEST_F(EnableLinOp, ExtendedApplyCopiesDataToCorrectExecutor)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(op->last_alpha_access, exec);
    ASSERT_EQ(op->last_b_access, exec);
    ASSERT_EQ(op->last_beta_access, exec);
    ASSERT_EQ(op->last_x_access, exec);
}


TEST_F(EnableLinOp, ExtendedApplyCopiesBackOnlyX)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(alpha->last_access, nullptr);
    ASSERT_EQ(b->last_access, nullptr);
    ASSERT_EQ(beta->last_access, nullptr);
    ASSERT_EQ(x->last_access, exec);
}


class FactoryParameter : public ::testing::Test {
protected:
    FactoryParameter() {}

public:
    // FACTORY_PARAMETER macro needs self, which is usually available in
    // enable_parameters_type. To reduce complexity, we add self here.
    GKO_ENABLE_SELF(FactoryParameter);

    std::vector<int> GKO_FACTORY_PARAMETER_VECTOR(vector_parameter, 10, 11);
    int GKO_FACTORY_PARAMETER_SCALAR(scalar_parameter, -4);
};


TEST_F(FactoryParameter, WorksOnCudaDefault)
{
    std::vector<int> expected{10, 11};

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(scalar_parameter, -4);
}


TEST_F(FactoryParameter, WorksOnCuda0)
{
    std::vector<int> expected{};

    auto result = &this->with_vector_parameter();

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(result, this);
}


TEST_F(FactoryParameter, WorksOnCuda1)
{
    std::vector<int> expected{2};

    this->with_vector_parameter(2).with_scalar_parameter(3);

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(scalar_parameter, 3);
}


TEST_F(FactoryParameter, WorksOnCuda2)
{
    std::vector<int> expected{8, 3};

    this->with_vector_parameter(8, 3);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnCuda3)
{
    std::vector<int> expected{1, 7, 2};

    this->with_vector_parameter(1, 7, 2);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnCuda4)
{
    std::vector<int> expected{4, 5, 4, 2};

    this->with_vector_parameter(4, 5, 4, 2);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnCuda5)
{
    std::vector<int> expected{9, 3, 4, 2, 7};

    this->with_vector_parameter(9, 3, 4, 2, 7);

    ASSERT_EQ(vector_parameter, expected);
}


}  // namespace
