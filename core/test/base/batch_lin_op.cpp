// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/batch_lin_op.hpp>


#include <complex>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace {


struct DummyLogger : gko::log::Logger {
    DummyLogger()
        : gko::log::Logger(gko::log::Logger::batch_linop_factory_events_mask)
    {}

    void on_batch_linop_factory_generate_started(
        const gko::batch::BatchLinOpFactory*,
        const gko::batch::BatchLinOp*) const override
    {
        batch_linop_factory_generate_started++;
    }

    void on_batch_linop_factory_generate_completed(
        const gko::batch::BatchLinOpFactory*, const gko::batch::BatchLinOp*,
        const gko::batch::BatchLinOp*) const override
    {
        batch_linop_factory_generate_completed++;
    }

    int mutable batch_linop_factory_generate_started = 0;
    int mutable batch_linop_factory_generate_completed = 0;
};


class DummyBatchLinOp : public gko::batch::EnableBatchLinOp<DummyBatchLinOp>,
                        public gko::EnableCreateMethod<DummyBatchLinOp> {
public:
    DummyBatchLinOp(std::shared_ptr<const gko::Executor> exec,
                    gko::batch_dim<2> size = gko::batch_dim<2>{})
        : gko::batch::EnableBatchLinOp<DummyBatchLinOp>(exec, size)
    {}
};


class EnableBatchLinOp : public ::testing::Test {
protected:
    EnableBatchLinOp()
        : ref{gko::ReferenceExecutor::create()},
          op{DummyBatchLinOp::create(ref,
                                     gko::batch_dim<2>(1, gko::dim<2>{3, 5}))}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::unique_ptr<DummyBatchLinOp> op;
};


TEST_F(EnableBatchLinOp, KnowsNumBatchItems)
{
    ASSERT_EQ(op->get_num_batch_items(), 1);
}


TEST_F(EnableBatchLinOp, KnowsItsSizes)
{
    auto op1_sizes = gko::batch_dim<2>(1, gko::dim<2>{3, 5});
    ASSERT_EQ(op->get_size(), op1_sizes);
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
};


class EnableBatchLinOpFactory : public ::testing::Test {
protected:
    EnableBatchLinOpFactory()
        : ref{gko::ReferenceExecutor::create()},
          logger{std::make_shared<DummyLogger>()}

    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<DummyLogger> logger;
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


TEST_F(EnableBatchLinOpFactory, FactoryGenerateIsLogged)
{
    auto before_logger = *logger;
    auto factory = DummyBatchLinOpWithFactory<>::build().on(ref);
    factory->add_logger(logger);
    factory->generate(
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 5})));

    ASSERT_EQ(logger->batch_linop_factory_generate_started,
              before_logger.batch_linop_factory_generate_started + 1);
    ASSERT_EQ(logger->batch_linop_factory_generate_completed,
              before_logger.batch_linop_factory_generate_completed + 1);
}


TEST_F(EnableBatchLinOpFactory, WithLoggersWorksAndPropagates)
{
    auto before_logger = *logger;
    auto factory =
        DummyBatchLinOpWithFactory<>::build().with_loggers(logger).on(ref);
    auto op = factory->generate(
        DummyBatchLinOp::create(ref, gko::batch_dim<2>(1, gko::dim<2>{3, 5})));

    ASSERT_EQ(logger->batch_linop_factory_generate_started,
              before_logger.batch_linop_factory_generate_started + 1);
    ASSERT_EQ(logger->batch_linop_factory_generate_completed,
              before_logger.batch_linop_factory_generate_completed + 1);
}


TEST_F(EnableBatchLinOpFactory, CopiesLinOpToOtherExecutor)
{
    auto ref2 = gko::ReferenceExecutor::create();
    auto dummy = gko::share(
        DummyBatchLinOp::create(ref2, gko::batch_dim<2>(1, gko::dim<2>{3, 5})));
    auto factory = DummyBatchLinOpWithFactory<>::build().with_value(6).on(ref);

    auto op = factory->generate(dummy);

    ASSERT_EQ(op->get_executor(), ref);
    ASSERT_EQ(op->get_parameters().value, 6);
    ASSERT_EQ(op->op_->get_executor(), ref);
    ASSERT_NE(op->op_.get(), dummy.get());
    ASSERT_TRUE(dynamic_cast<const DummyBatchLinOp*>(op->op_.get()));
}


}  // namespace
