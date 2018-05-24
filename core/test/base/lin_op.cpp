/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/lin_op.hpp>


#include <gtest/gtest.h>


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
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        this->access();
        static_cast<const DummyLinOp *>(b)->access();
        static_cast<const DummyLinOp *>(x)->access();
        last_b_access = b->get_executor();
        last_x_access = x->get_executor();
    }

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        this->access();
        static_cast<const DummyLinOp *>(alpha)->access();
        static_cast<const DummyLinOp *>(b)->access();
        static_cast<const DummyLinOp *>(beta)->access();
        static_cast<const DummyLinOp *>(x)->access();
        last_alpha_access = alpha->get_executor();
        last_b_access = b->get_executor();
        last_beta_access = beta->get_executor();
        last_x_access = x->get_executor();
    }
};


class EnableLinOp : public ::testing::Test {
protected:
    EnableLinOp()
        : ref{gko::ReferenceExecutor::create()},
          omp{gko::OmpExecutor::create()},
          op{DummyLinOp::create(omp, gko::dim<2>{3, 5})},
          alpha{DummyLinOp::create(ref, gko::dim<2>{1})},
          beta{DummyLinOp::create(ref, gko::dim<2>{1})},
          b{DummyLinOp::create(ref, gko::dim<2>{5, 4})},
          x{DummyLinOp::create(ref, gko::dim<2>{3, 4})}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;
    std::unique_ptr<DummyLinOp> op;
    std::unique_ptr<DummyLinOp> alpha;
    std::unique_ptr<DummyLinOp> beta;
    std::unique_ptr<DummyLinOp> b;
    std::unique_ptr<DummyLinOp> x;
};


TEST_F(EnableLinOp, CallsApplyImpl)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(op->last_access, omp);
}


TEST_F(EnableLinOp, CallsExtendedApplyImpl)
{
    op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta), gko::lend(x));

    ASSERT_EQ(op->last_access, omp);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongBSize)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 4});

    ASSERT_THROW(op->apply(gko::lend(wrong), gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{5, 4});

    ASSERT_THROW(op->apply(gko::lend(b), gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 5});

    ASSERT_THROW(op->apply(gko::lend(b), gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongBSize)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 4});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(wrong), gko::lend(beta),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{5, 4});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta),
                           gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 5});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta),
                           gko::lend(wrong)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongAlphaDimension)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{2, 5});

    ASSERT_THROW(op->apply(gko::lend(wrong), gko::lend(b), gko::lend(beta),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongBetaDimension)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{2, 5});

    ASSERT_THROW(op->apply(gko::lend(alpha), gko::lend(b), gko::lend(wrong),
                           gko::lend(x)),
                 gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ApplyCopiesDataToCorrectExecutor)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(op->last_b_access, omp);
    ASSERT_EQ(op->last_x_access, omp);
}


TEST_F(EnableLinOp, ApplyCopiesBackOnlyX)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(b->last_access, nullptr);
    ASSERT_EQ(x->last_access, omp);
}


TEST_F(EnableLinOp, ExtendedApplyCopiesDataToCorrectExecutor)
{
    op->apply(gko::lend(alpha), gko::lend(b), gko::lend(beta), gko::lend(x));

    ASSERT_EQ(op->last_alpha_access, omp);
    ASSERT_EQ(op->last_b_access, omp);
    ASSERT_EQ(op->last_beta_access, omp);
    ASSERT_EQ(op->last_x_access, omp);
}


TEST_F(EnableLinOp, ExtendedApplyCopiesBackOnlyX)
{
    op->apply(gko::lend(b), gko::lend(x));

    ASSERT_EQ(alpha->last_access, nullptr);
    ASSERT_EQ(b->last_access, nullptr);
    ASSERT_EQ(beta->last_access, nullptr);
    ASSERT_EQ(x->last_access, omp);
}


template <typename T = int>
class DummyLinOpWithFactory
    : public gko::EnableLinOp<DummyLinOpWithFactory<T>> {
public:
    DummyLinOpWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithFactory>(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        T GKO_FACTORY_PARAMETER(value, T{5});
    };
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);

    DummyLinOpWithFactory(const Factory *factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}
};


class EnableLinOpFactory : public ::testing::Test {
protected:
    EnableLinOpFactory() : ref{gko::ReferenceExecutor::create()} {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};


TEST_F(EnableLinOpFactory, CreatesDefaultFactory)
{
    auto factory = DummyLinOpWithFactory<>::Factory::create().on_executor(ref);

    ASSERT_EQ(factory->get_parameters().value, 5);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableLinOpFactory, CreatesFactoryWithParameters)
{
    auto factory =
        DummyLinOpWithFactory<>::Factory::create().with_value(7).on_executor(
            ref);

    ASSERT_EQ(factory->get_parameters().value, 7);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableLinOpFactory, PassesParametersToLinOp)
{
    auto dummy = gko::share(DummyLinOp::create(ref, gko::dim<2>{3, 5}));
    auto factory =
        DummyLinOpWithFactory<>::Factory::create().with_value(6).on_executor(
            ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_.get(), dummy.get());
}


}  // namespace
