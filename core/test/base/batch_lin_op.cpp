/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>


namespace {


class DummyBatchLinOp : public gko::EnableBatchLinOp<DummyBatchLinOp>,
                        public gko::EnableCreateMethod<DummyBatchLinOp> {
public:
    DummyBatchLinOp(std::shared_ptr<const gko::Executor> exec,
                    std::vector<gko::dim<2>> size = std::vector<gko::dim<2>>{})
        : EnableBatchLinOp<DummyBatchLinOp>(exec, size)
    {}

    void access() const { last_access = this->get_executor(); }

    mutable std::shared_ptr<const gko::Executor> last_access;
    mutable std::shared_ptr<const gko::Executor> last_b_access;
    mutable std::shared_ptr<const gko::Executor> last_x_access;
    mutable std::shared_ptr<const gko::Executor> last_alpha_access;
    mutable std::shared_ptr<const gko::Executor> last_beta_access;

protected:
    void apply_impl(const gko::BatchLinOp* b, gko::BatchLinOp* x) const override
    {
        this->access();
        static_cast<const DummyBatchLinOp*>(b)->access();
        static_cast<const DummyBatchLinOp*>(x)->access();
        last_b_access = b->get_executor();
        last_x_access = x->get_executor();
    }

    void apply_impl(const gko::BatchLinOp* alpha, const gko::BatchLinOp* b,
                    const gko::BatchLinOp* beta,
                    gko::BatchLinOp* x) const override
    {
        this->access();
        static_cast<const DummyBatchLinOp*>(alpha)->access();
        static_cast<const DummyBatchLinOp*>(b)->access();
        static_cast<const DummyBatchLinOp*>(beta)->access();
        static_cast<const DummyBatchLinOp*>(x)->access();
        last_alpha_access = alpha->get_executor();
        last_b_access = b->get_executor();
        last_beta_access = beta->get_executor();
        last_x_access = x->get_executor();
    }
};


class EnableBatchLinOp : public ::testing::Test {
protected:
    EnableBatchLinOp()
        : ref{gko::ReferenceExecutor::create()},
          ref2{gko::ReferenceExecutor::create()},
          op{DummyBatchLinOp::create(
              ref2, std::vector<gko::dim<2>>{gko::dim<2>{3, 5}})},
          op2{DummyBatchLinOp::create(
              ref2,
              std::vector<gko::dim<2>>{gko::dim<2>{3, 5}, gko::dim<2>{3, 5}})},
          alpha{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{1}})},
          alpha2{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{1}, gko::dim<2>{1}})},
          beta{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{1}})},
          beta2{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{1}, gko::dim<2>{1}})},
          b{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{5, 4}})},
          b2{DummyBatchLinOp::create(
              ref,
              std::vector<gko::dim<2>>{gko::dim<2>{5, 4}, gko::dim<2>{5, 4}})},
          x{DummyBatchLinOp::create(
              ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 4}})},
          x2{DummyBatchLinOp::create(
              ref,
              std::vector<gko::dim<2>>{gko::dim<2>{3, 4}, gko::dim<2>{3, 4}})}
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


TEST_F(EnableBatchLinOp, KnowsNumBatches)
{
    ASSERT_EQ(op->get_num_batch_entries(), 1);
    ASSERT_EQ(op2->get_num_batch_entries(), 2);
}


TEST_F(EnableBatchLinOp, KnowsItsSizes)
{
    auto op1_sizes =
        gko::batch_dim<2>(std::vector<gko::dim<2>>{gko::dim<2>{3, 5}});
    auto op2_sizes = gko::batch_dim<2>(
        std::vector<gko::dim<2>>{gko::dim<2>{3, 5}, gko::dim<2>{3, 5}});
    ASSERT_EQ(op->get_size(), op1_sizes);
    ASSERT_EQ(op2->get_size(), op2_sizes);
}


TEST_F(EnableBatchLinOp, CallsApplyImpl)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(op->last_access, ref2);
}


TEST_F(EnableBatchLinOp, CallsApplyImplForBatch)
{
    op2->apply(gko::lend(b2), gko::lend(x2));

    ASSERT_EQ(op2->last_access, ref2);
}


TEST_F(EnableBatchLinOp, CallsExtendedApplyImpl)
{
    op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta), gko::lend(x));

    ASSERT_EQ(op->last_access, ref2);
}


TEST_F(EnableBatchLinOp, CallsExtendedApplyImplBatch)
{
    op2->apply(gko::lend(alpha2), gko::lend(b2), gko::lend(beta2),
               gko::lend(x2));

    ASSERT_EQ(op2->last_access, ref2);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongBSize)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 4}});

    ASSERT_THROW(op->apply(gko::lend(wrong), gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongBatchSize)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 4}});

    ASSERT_THROW(op2->apply(gko::lend(wrong), gko::lend(x2)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{5, 4}});

    ASSERT_THROW(op->apply(gko::lend(b), gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnOneBatchWrongSolutionRows)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{5, 4}, gko::dim<2>{5, 4}});

    ASSERT_THROW(op2->apply(gko::lend(b2), gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 5}});

    ASSERT_THROW(op->apply(gko::lend(b), gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongBSize)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 4}});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(wrong), gko::lend(beta),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{5, 4}});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta),
                           gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 5}});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta),
                           gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongAlphaDimension)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{2, 5}});

    ASSERT_THROW(op->apply(gko::lend(wrong), gko::lend(b), gko::lend(beta),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableBatchLinOp, ExtendedApplyFailsOnWrongBetaDimension)
{
    auto wrong = DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{2, 5}});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(wrong),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


// For tests between different memory, check cuda/test/base/batch_lin_op.cu
TEST_F(EnableBatchLinOp, ApplyDoesNotCopyBetweenSameMemory)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(op->last_b_access, ref);
    ASSERT_EQ(op->last_x_access, ref);
}


TEST_F(EnableBatchLinOp, ApplyNoCopyBackBetweenSameMemory)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(b->last_access, ref);
    ASSERT_EQ(x->last_access, ref);
}


TEST_F(EnableBatchLinOp, ExtendedApplyDoesNotCopyBetweenSameMemory)
{
    op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta), gko::lend(x));

    ASSERT_EQ(op->last_alpha_access, ref);
    ASSERT_EQ(op->last_b_access, ref);
    ASSERT_EQ(op->last_beta_access, ref);
    ASSERT_EQ(op->last_x_access, ref);
}


TEST_F(EnableBatchLinOp, ExtendedApplyNoCopyBackBetweenSameMemory)
{
    op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta), gko::lend(x));

    ASSERT_EQ(alpha->last_access, ref);
    ASSERT_EQ(b->last_access, ref);
    ASSERT_EQ(beta->last_access, ref);
    ASSERT_EQ(x->last_access, ref);
}


template <typename T = int>
class DummyBatchLinOpWithFactory
    : public gko::EnableBatchLinOp<DummyBatchLinOpWithFactory<T>> {
public:
    DummyBatchLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableBatchLinOp<DummyBatchLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        T GKO_FACTORY_PARAMETER_SCALAR(value, T{5});
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(DummyBatchLinOpWithFactory, parameters,
                                    Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyBatchLinOpWithFactory(const Factory* factory,
                               std::shared_ptr<const gko::BatchLinOp> op)
        : gko::EnableBatchLinOp<DummyBatchLinOpWithFactory>(
              factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::BatchLinOp> op_;

protected:
    void apply_impl(const gko::BatchLinOp* b, gko::BatchLinOp* x) const override
    {}

    void apply_impl(const gko::BatchLinOp* alpha, const gko::BatchLinOp* b,
                    const gko::BatchLinOp* beta,
                    gko::BatchLinOp* x) const override
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
    auto dummy = gko::share(DummyBatchLinOp::create(
        ref, std::vector<gko::dim<2>>{gko::dim<2>{3, 5}}));
    auto factory = DummyBatchLinOpWithFactory<>::build().with_value(6).on(ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_.get(), dummy.get());
}


}  // namespace
