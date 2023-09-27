/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/batch_lin_op.hpp>


#include <complex>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>


namespace {


class DummyBatchLinOp : public gko::batch::EnableBatchLinOp<DummyBatchLinOp>,
                        public gko::EnableCreateMethod<DummyBatchLinOp> {
public:
    DummyBatchLinOp(std::shared_ptr<const gko::Executor> exec,
                    gko::batch_dim<2> size = gko::batch_dim<2>{})
        : gko::batch::EnableBatchLinOp<DummyBatchLinOp>(exec, size)
    {}

    int called = 0;

protected:
    void apply_impl(const gko::batch::BatchLinOp* b,
                    gko::batch::BatchLinOp* x) const override
    {
        this->called = 1;
    }

    void apply_impl(const gko::batch::BatchLinOp* alpha,
                    const gko::batch::BatchLinOp* b,
                    const gko::batch::BatchLinOp* beta,
                    gko::batch::BatchLinOp* x) const override
    {
        this->called = 2;
    }
};


class EnableBatchLinOp : public ::testing::Test {
protected:
    EnableBatchLinOp()
        : ref{gko::ReferenceExecutor::create()},
          ref2{gko::ReferenceExecutor::create()},
          op{DummyBatchLinOp::create(ref2,
                                     gko::batch_dim<2>(1, gko::dim<2>{3, 5}))},
          op2{DummyBatchLinOp::create(ref2,
                                      gko::batch_dim<2>(2, gko::dim<2>{3, 5}))},
          alpha{DummyBatchLinOp::create(
              ref, gko::batch_dim<2>(1, gko::dim<2>{1, 1}))},
          alpha2{DummyBatchLinOp::create(
              ref, gko::batch_dim<2>(2, gko::dim<2>{1, 1}))},
          beta{DummyBatchLinOp::create(
              ref, gko::batch_dim<2>(1, gko::dim<2>{1, 1}))},
          beta2{DummyBatchLinOp::create(
              ref, gko::batch_dim<2>(2, gko::dim<2>{1, 1}))},
          b{DummyBatchLinOp::create(ref,
                                    gko::batch_dim<2>(1, gko::dim<2>{5, 4}))},
          b2{DummyBatchLinOp::create(ref,
                                     gko::batch_dim<2>(2, gko::dim<2>{5, 4}))},
          x{DummyBatchLinOp::create(ref,
                                    gko::batch_dim<2>(1, gko::dim<2>{3, 4}))},
          x2{DummyBatchLinOp::create(ref,
                                     gko::batch_dim<2>(2, gko::dim<2>{3, 4}))}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::ReferenceExecutor> ref2;
    std::unique_ptr<DummyBatchLinOp> op;
    std::unique_ptr<DummyBatchLinOp> op2;
    std::unique_ptr<DummyBatchLinOp> alpha;
    std::unique_ptr<DummyBatchLinOp> alpha2;
    std::unique_ptr<DummyBatchLinOp> beta;
    std::unique_ptr<DummyBatchLinOp> beta2;
    std::unique_ptr<DummyBatchLinOp> b;
    std::unique_ptr<DummyBatchLinOp> b2;
    std::unique_ptr<DummyBatchLinOp> x;
    std::unique_ptr<DummyBatchLinOp> x2;
};


TEST_F(EnableBatchLinOp, KnowsNumBatchItems)
{
    ASSERT_EQ(op->get_num_batch_items(), 1);
    ASSERT_EQ(op2->get_num_batch_items(), 2);
}


TEST_F(EnableBatchLinOp, KnowsItsSizes)
{
    auto op1_sizes = gko::batch_dim<2>(1, gko::dim<2>{3, 5});
    auto op2_sizes = gko::batch_dim<2>(2, gko::dim<2>{3, 5});
    ASSERT_EQ(op->get_size(), op1_sizes);
    ASSERT_EQ(op2->get_size(), op2_sizes);
}


TEST_F(EnableBatchLinOp, CallsApplyImpl)
{
    op->apply(b, x);

    ASSERT_EQ(op->called, 1);
}


TEST_F(EnableBatchLinOp, CallsApplyImplForBatch)
{
    op2->apply(b2, x2);

    ASSERT_EQ(op2->called, 1);
}


TEST_F(EnableBatchLinOp, CallsExtendedApplyImpl)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(op->called, 2);
}


TEST_F(EnableBatchLinOp, CallsExtendedApplyImplBatch)
{
    op2->apply(alpha2, b2, beta2, x2);

    ASSERT_EQ(op2->called, 2);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongBatchSize)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 4}));

    ASSERT_THROW(op->apply(wrong, x), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongNumBatchItems)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 4}));

    ASSERT_THROW(op2->apply(wrong, x2), gko::ValueMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongSolutionRows)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{5, 4}));

    ASSERT_THROW(op->apply(b, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnOneBatchItemWrongSolutionRows)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(2, gko::dim<2>{5, 4}));

    ASSERT_THROW(op2->apply(b2, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongSolutionColumns)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 5}));

    ASSERT_THROW(op->apply(b, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnOneBatchItemWrongSolutionColumn)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(2, gko::dim<2>{3, 5}));

    ASSERT_THROW(op2->apply(b2, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongBatchSize)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 4}));

    ASSERT_THROW(op->apply(alpha, wrong, beta, x), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongSolutionRows)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{5, 4}));

    ASSERT_THROW(op->apply(alpha, b, beta, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongSolutionColumns)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 5}));

    ASSERT_THROW(op->apply(alpha, b, beta, wrong), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongAlphaDimension)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{2, 5}));

    ASSERT_THROW(op->apply(wrong, b, beta, x), gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongBetaDimension)
{
    auto wrong =
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{2, 5}));

    ASSERT_THROW(op->apply(alpha, b, wrong, x), gko::DimensionMismatch);
}


template <typename T = int>
class DummyBatchLinOpWithFactory
    : public gko::batch::EnableBatchLinOp<DummyBatchLinOpWithFactory<T>> {
public:
    DummyBatchLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::batch::EnableBatchLinOp<DummyBatchLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        T GKO_FACTORY_PARAMETER_SCALAR(value, T{5});
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(DummyBatchLinOpWithFactory, parameters,
                                    Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyBatchLinOpWithFactory(const Factory* factory,
                               std::shared_ptr<const gko::batch::BatchLinOp> op)
        : gko::batch::EnableBatchLinOp<DummyBatchLinOpWithFactory>(
              factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::batch::BatchLinOp> op_;

protected:
    void apply_impl(const gko::batch::BatchLinOp* b,
                    gko::batch::BatchLinOp* x) const override
    {}

    void apply_impl(const gko::batch::BatchLinOp* alpha,
                    const gko::batch::BatchLinOp* b,
                    const gko::batch::BatchLinOp* beta,
                    gko::batch::BatchLinOp* x) const override
    {}
};


class EnableBatchLinOpFactory : public ::testing::Test {
protected:
    EnableBatchLinOpFactory() : ref{gko::ReferenceExecutor::create()} {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};


TEST_F(EnableBatchLinOpFactory, CreatesDefaultFactory)
{
    auto factory = DummyBatchLinOpWithFactory<>::build().on(ref);

    ASSERT_EQ(factory->get_parameters().value, 5);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableBatchLinOpFactory, CreatesFactoryWithParameters)
{
    auto factory = DummyBatchLinOpWithFactory<>::build().with_value(7).on(ref);

    ASSERT_EQ(factory->get_parameters().value, 7);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableBatchLinOpFactory, PassesParametersToBatchLinOp)
{
    auto dummy = gko::share(
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 5})));
    auto factory = DummyBatchLinOpWithFactory<>::build().with_value(6).on(ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_.get(), dummy.get());
}


}  // namespace
