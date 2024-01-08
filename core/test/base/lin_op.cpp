// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>


#include <complex>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


namespace {


struct DummyLogger : gko::log::Logger {
    DummyLogger()
        : gko::log::Logger(gko::log::Logger::linop_events_mask |
                           gko::log::Logger::linop_factory_events_mask)
    {}

    void on_linop_apply_started(const gko::LinOp*, const gko::LinOp*,
                                const gko::LinOp*) const override
    {
        linop_apply_started++;
    }

    void on_linop_apply_completed(const gko::LinOp*, const gko::LinOp*,
                                  const gko::LinOp*) const override
    {
        linop_apply_completed++;
    }

    void on_linop_advanced_apply_started(const gko::LinOp*, const gko::LinOp*,
                                         const gko::LinOp*, const gko::LinOp*,
                                         const gko::LinOp*) const override
    {
        linop_advanced_apply_started++;
    }

    void on_linop_advanced_apply_completed(const gko::LinOp*, const gko::LinOp*,
                                           const gko::LinOp*, const gko::LinOp*,
                                           const gko::LinOp*) const override
    {
        linop_advanced_apply_completed++;
    }

    void on_linop_factory_generate_started(const gko::LinOpFactory*,
                                           const gko::LinOp*) const override
    {
        linop_factory_generate_started++;
    }

    void on_linop_factory_generate_completed(const gko::LinOpFactory*,
                                             const gko::LinOp*,
                                             const gko::LinOp*) const override
    {
        linop_factory_generate_completed++;
    }

    int mutable linop_apply_started = 0;
    int mutable linop_apply_completed = 0;
    int mutable linop_advanced_apply_started = 0;
    int mutable linop_advanced_apply_completed = 0;
    int mutable linop_factory_generate_started = 0;
    int mutable linop_factory_generate_completed = 0;
};


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


class EnableLinOp : public ::testing::Test {
protected:
    EnableLinOp()
        : ref{gko::ReferenceExecutor::create()},
          ref2{gko::ReferenceExecutor::create()},
          op{DummyLinOp::create(ref2, gko::dim<2>{3, 5})},
          alpha{DummyLinOp::create(ref, gko::dim<2>{1})},
          beta{DummyLinOp::create(ref, gko::dim<2>{1})},
          b{DummyLinOp::create(ref, gko::dim<2>{5, 4})},
          x{DummyLinOp::create(ref, gko::dim<2>{3, 4})},
          logger{std::make_shared<DummyLogger>()}
    {
        op->add_logger(logger);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::ReferenceExecutor> ref2;
    std::unique_ptr<DummyLinOp> op;
    std::unique_ptr<DummyLinOp> alpha;
    std::unique_ptr<DummyLinOp> beta;
    std::unique_ptr<DummyLinOp> b;
    std::unique_ptr<DummyLinOp> x;
    std::shared_ptr<DummyLogger> logger;
};


TEST_F(EnableLinOp, CallsApplyImpl)
{
    op->apply(b, x);

    ASSERT_EQ(op->last_access, ref2);
}


TEST_F(EnableLinOp, CallsExtendedApplyImpl)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(op->last_access, ref2);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongBSize)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 4});

    ASSERT_THROW(op->apply(wrong, x), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{5, 4});

    ASSERT_THROW(op->apply(b, wrong), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 5});

    ASSERT_THROW(op->apply(b, wrong), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongBSize)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 4});

    ASSERT_THROW(op->apply(alpha, wrong, beta, x), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongSolutionRows)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{5, 4});

    ASSERT_THROW(op->apply(alpha, b, beta, wrong), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongSolutionColumns)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{3, 5});

    ASSERT_THROW(op->apply(alpha, b, beta, wrong), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongAlphaDimension)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{2, 5});

    ASSERT_THROW(op->apply(wrong, b, beta, x), gko::DimensionMismatch);
}


TEST_F(EnableLinOp, ExtendedApplyFailsOnWrongBetaDimension)
{
    auto wrong = DummyLinOp::create(ref, gko::dim<2>{2, 5});

    ASSERT_THROW(op->apply(alpha, b, wrong, x), gko::DimensionMismatch);
}


// For tests between different memory, check cuda/test/base/lin_op.cu
TEST_F(EnableLinOp, ApplyDoesNotCopyBetweenSameMemory)
{
    op->apply(b, x);

    ASSERT_EQ(op->last_b_access, ref);
    ASSERT_EQ(op->last_x_access, ref);
}


TEST_F(EnableLinOp, ApplyNoCopyBackBetweenSameMemory)
{
    op->apply(b, x);

    ASSERT_EQ(b->last_access, ref);
    ASSERT_EQ(x->last_access, ref);
}


TEST_F(EnableLinOp, ExtendedApplyDoesNotCopyBetweenSameMemory)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(op->last_alpha_access, ref);
    ASSERT_EQ(op->last_b_access, ref);
    ASSERT_EQ(op->last_beta_access, ref);
    ASSERT_EQ(op->last_x_access, ref);
}


TEST_F(EnableLinOp, ExtendedApplyNoCopyBackBetweenSameMemory)
{
    op->apply(alpha, b, beta, x);

    ASSERT_EQ(alpha->last_access, ref);
    ASSERT_EQ(b->last_access, ref);
    ASSERT_EQ(beta->last_access, ref);
    ASSERT_EQ(x->last_access, ref);
}


TEST_F(EnableLinOp, ApplyUsesInitialGuessReturnsFalse)
{
    ASSERT_FALSE(op->apply_uses_initial_guess());
}


TEST_F(EnableLinOp, ApplyIsLogged)
{
    auto before_logger = *logger;

    op->apply(b, x);

    ASSERT_EQ(logger->linop_apply_started,
              before_logger.linop_apply_started + 1);
    ASSERT_EQ(logger->linop_apply_completed,
              before_logger.linop_apply_completed + 1);
}


TEST_F(EnableLinOp, AdvancedApplyIsLogged)
{
    auto before_logger = *logger;

    op->apply(alpha, b, beta, x);

    ASSERT_EQ(logger->linop_advanced_apply_started,
              before_logger.linop_advanced_apply_started + 1);
    ASSERT_EQ(logger->linop_advanced_apply_completed,
              before_logger.linop_advanced_apply_completed + 1);
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
        T GKO_FACTORY_PARAMETER_SCALAR(value, T{5});
    };
    GKO_ENABLE_LIN_OP_FACTORY(DummyLinOpWithFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    DummyLinOpWithFactory(const Factory* factory,
                          std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyLinOpWithFactory>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          op_{op}
    {}

    std::shared_ptr<const gko::LinOp> op_;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


class EnableLinOpFactory : public ::testing::Test {
protected:
    EnableLinOpFactory()
        : ref{gko::ReferenceExecutor::create()},
          logger{std::make_shared<DummyLogger>()}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<DummyLogger> logger;
};


TEST_F(EnableLinOpFactory, CreatesDefaultFactory)
{
    auto factory = DummyLinOpWithFactory<>::build().on(ref);

    ASSERT_EQ(factory->get_parameters().value, 5);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableLinOpFactory, CreatesFactoryWithParameters)
{
    auto factory = DummyLinOpWithFactory<>::build().with_value(7).on(ref);

    ASSERT_EQ(factory->get_parameters().value, 7);
    ASSERT_EQ(factory->get_executor(), ref);
}


TEST_F(EnableLinOpFactory, PassesParametersToLinOp)
{
    auto dummy = gko::share(DummyLinOp::create(ref, gko::dim<2>{3, 5}));
    auto factory = DummyLinOpWithFactory<>::build().with_value(6).on(ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_.get(), dummy.get());
}


TEST_F(EnableLinOpFactory, FactoryGenerateIsLogged)
{
    auto before_logger = *logger;
    auto factory = DummyLinOpWithFactory<>::build().on(ref);
    factory->add_logger(logger);
    factory->generate(DummyLinOp::create(ref, gko::dim<2>{3, 5}));

    ASSERT_EQ(logger->linop_factory_generate_started,
              before_logger.linop_factory_generate_started + 1);
    ASSERT_EQ(logger->linop_factory_generate_completed,
              before_logger.linop_factory_generate_completed + 1);
}


TEST_F(EnableLinOpFactory, WithLoggersWorksAndPropagates)
{
    auto before_logger = *logger;
    auto factory =
        DummyLinOpWithFactory<>::build().with_loggers(logger).on(ref);
    auto op = factory->generate(DummyLinOp::create(ref, gko::dim<2>{3, 5}));
    op->apply(op, op);

    ASSERT_EQ(logger->linop_factory_generate_started,
              before_logger.linop_factory_generate_started + 1);
    ASSERT_EQ(logger->linop_factory_generate_completed,
              before_logger.linop_factory_generate_completed + 1);
    ASSERT_EQ(logger->linop_apply_started,
              before_logger.linop_apply_started + 1);
    ASSERT_EQ(logger->linop_apply_completed,
              before_logger.linop_apply_completed + 1);
}


TEST_F(EnableLinOpFactory, CopiesLinOpToOtherExecutor)
{
    auto ref2 = gko::ReferenceExecutor::create();
    auto dummy = gko::share(DummyLinOp::create(ref2, gko::dim<2>{3, 5}));
    auto factory = DummyLinOpWithFactory<>::build().with_value(6).on(ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_->get_executor(), ref);
    ASSERT_NE(op->op_.get(), dummy.get());
    ASSERT_TRUE(dynamic_cast<const DummyLinOp*>(op->op_.get()));
}


template <typename Type>
class DummyLinOpWithType
    : public gko::EnableLinOp<DummyLinOpWithType<Type>>,
      public gko::EnableCreateMethod<DummyLinOpWithType<Type>>,
      public gko::EnableAbsoluteComputation<
          gko::remove_complex<DummyLinOpWithType<Type>>> {
public:
    using absolute_type = gko::remove_complex<DummyLinOpWithType>;
    DummyLinOpWithType(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOpWithType>(exec)
    {}

    DummyLinOpWithType(std::shared_ptr<const gko::Executor> exec,
                       gko::dim<2> size, Type value)
        : gko::EnableLinOp<DummyLinOpWithType>(exec, size), value_(value)
    {}

    void compute_absolute_inplace() override { value_ = gko::abs(value_); }

    std::unique_ptr<absolute_type> compute_absolute() const override
    {
        return std::make_unique<absolute_type>(
            this->get_executor(), this->get_size(), gko::abs(value_));
    }

    Type get_value() const { return value_; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}

private:
    Type value_;
};


class EnableAbsoluteComputation : public ::testing::Test {
protected:
    using dummy_type = DummyLinOpWithType<std::complex<double>>;
    EnableAbsoluteComputation()
        : ref{gko::ReferenceExecutor::create()},
          op{dummy_type::create(ref, gko::dim<2>{1, 1},
                                std::complex<double>{-3.0, 4.0})}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<dummy_type> op;
};


TEST_F(EnableAbsoluteComputation, InplaceAbsoluteOnConcreteType)
{
    op->compute_absolute_inplace();

    ASSERT_EQ(op->get_value(), std::complex<double>{5.0});
}


TEST_F(EnableAbsoluteComputation, OutplaceAbsoluteOnConcreteType)
{
    auto abs_op = op->compute_absolute();

    static_assert(
        std::is_same<decltype(abs_op),
                     std::unique_ptr<gko::remove_complex<dummy_type>>>::value,
        "Types must match.");
    ASSERT_EQ(abs_op->get_value(), 5.0);
}


TEST_F(EnableAbsoluteComputation, InplaceAbsoluteOnAbsoluteComputable)
{
    auto linop = gko::as<gko::LinOp>(op);

    gko::as<gko::AbsoluteComputable>(linop)->compute_absolute_inplace();

    ASSERT_EQ(gko::as<dummy_type>(linop)->get_value(),
              std::complex<double>{5.0});
}


TEST_F(EnableAbsoluteComputation, OutplaceAbsoluteOnAbsoluteComputable)
{
    auto abs_op = op->compute_absolute();

    static_assert(
        std::is_same<decltype(abs_op),
                     std::unique_ptr<gko::remove_complex<dummy_type>>>::value,
        "Types must match.");
    ASSERT_EQ(abs_op->get_value(), 5.0);
}


TEST_F(EnableAbsoluteComputation, ThrowWithoutAbsoluteComputableInterface)
{
    std::shared_ptr<gko::LinOp> linop = DummyLinOp::create(ref);

    ASSERT_THROW(gko::as<gko::AbsoluteComputable>(linop), gko::NotSupported);
}


}  // namespace
