// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/stream.hpp>


#include <iomanip>
#include <sstream>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"


namespace {


constexpr int num_iters = 10;


template <typename T>
class Stream : public ::testing::Test {};

TYPED_TEST_SUITE(Stream, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Stream, CatchesAllocationStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::allocation_started_mask, out);

    logger->template on<gko::log::Logger::allocation_started>(exec.get(), 42);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "allocation started on");
    GKO_ASSERT_STR_CONTAINS(os, "42");
}


TYPED_TEST(Stream, CatchesAllocationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::allocation_completed_mask, out);
    int dummy = 1;
    std::stringstream ptrstream;
    ptrstream << std::hex << "0x" << reinterpret_cast<gko::uintptr>(&dummy);

    logger->template on<gko::log::Logger::allocation_completed>(
        exec.get(), 42, reinterpret_cast<gko::uintptr>(&dummy));

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "allocation completed on");
    GKO_ASSERT_STR_CONTAINS(os, "42");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TYPED_TEST(Stream, CatchesFreeStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::free_started_mask, out);
    int dummy = 1;
    std::stringstream ptrstream;
    ptrstream << std::hex << "0x" << reinterpret_cast<gko::uintptr>(&dummy);

    logger->template on<gko::log::Logger::free_started>(
        exec.get(), reinterpret_cast<gko::uintptr>(&dummy));

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "free started on");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TYPED_TEST(Stream, CatchesFreeCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::free_completed_mask, out);
    int dummy = 1;
    std::stringstream ptrstream;
    ptrstream << std::hex << "0x" << reinterpret_cast<gko::uintptr>(&dummy);

    logger->template on<gko::log::Logger::free_completed>(
        exec.get(), reinterpret_cast<gko::uintptr>(&dummy));

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "free completed on");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TYPED_TEST(Stream, CatchesCopyStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::copy_started_mask, out);
    int dummy_in = 1;
    int dummy_out = 1;
    std::stringstream ptrstream_in;
    ptrstream_in << std::hex << "0x"
                 << reinterpret_cast<gko::uintptr>(&dummy_in);
    std::stringstream ptrstream_out;
    ptrstream_out << std::hex << "0x"
                  << reinterpret_cast<gko::uintptr>(&dummy_out);

    logger->template on<gko::log::Logger::copy_started>(
        exec.get(), exec.get(), reinterpret_cast<gko::uintptr>(&dummy_in),
        reinterpret_cast<gko::uintptr>(&dummy_out), 42);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "copy started");
    GKO_ASSERT_STR_CONTAINS(os, "from Location[" + ptrstream_in.str());
    GKO_ASSERT_STR_CONTAINS(os, "to Location[" + ptrstream_out.str());
    GKO_ASSERT_STR_CONTAINS(os, "with Bytes[42]");
}


TYPED_TEST(Stream, CatchesCopyCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::copy_completed_mask, out);
    int dummy_in = 1;
    int dummy_out = 1;
    std::stringstream ptrstream_in;
    ptrstream_in << std::hex << "0x"
                 << reinterpret_cast<gko::uintptr>(&dummy_in);
    std::stringstream ptrstream_out;
    ptrstream_out << std::hex << "0x"
                  << reinterpret_cast<gko::uintptr>(&dummy_out);

    logger->template on<gko::log::Logger::copy_completed>(
        exec.get(), exec.get(), reinterpret_cast<gko::uintptr>(&dummy_in),
        reinterpret_cast<gko::uintptr>(&dummy_out), 42);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "copy completed");
    GKO_ASSERT_STR_CONTAINS(os, "from Location[" + ptrstream_in.str());
    GKO_ASSERT_STR_CONTAINS(os, "to Location[" + ptrstream_out.str());
    GKO_ASSERT_STR_CONTAINS(os, "with Bytes[42]");
}


TYPED_TEST(Stream, CatchesOperationLaunched)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::operation_launched_mask, out);
    gko::Operation op;
    std::stringstream ptrstream;
    ptrstream << &op;

    logger->template on<gko::log::Logger::operation_launched>(exec.get(), &op);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "started on");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TYPED_TEST(Stream, CatchesOperationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::operation_completed_mask, out);
    gko::Operation op;
    std::stringstream ptrstream;
    ptrstream << &op;

    logger->template on<gko::log::Logger::operation_completed>(exec.get(), &op);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "completed on");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectCreateStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_create_started_mask, out);
    auto po = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream;
    ptrstream << po.get();

    logger->template on<gko::log::Logger::polymorphic_object_create_started>(
        exec.get(), po.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
    GKO_ASSERT_STR_CONTAINS(os, "create started from");
}


TYPED_TEST(Stream, CatchesPolymorphicObjectCreateCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_create_completed_mask, out);
    auto po = gko::matrix::Dense<TypeParam>::create(exec);
    auto output = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream_in;
    ptrstream_in << po.get();
    std::stringstream ptrstream_out;
    ptrstream_out << output.get();

    logger->template on<gko::log::Logger::polymorphic_object_create_completed>(
        exec.get(), po.get(), output.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_in.str());
    GKO_ASSERT_STR_CONTAINS(os, "create completed from");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_out.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectCopyStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_copy_started_mask, out);
    auto from = gko::matrix::Dense<TypeParam>::create(exec);
    auto to = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->template on<gko::log::Logger::polymorphic_object_copy_started>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    GKO_ASSERT_STR_CONTAINS(os, "copy started to");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectCopyCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_copy_completed_mask, out);
    auto from = gko::matrix::Dense<TypeParam>::create(exec);
    auto to = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->template on<gko::log::Logger::polymorphic_object_copy_completed>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    GKO_ASSERT_STR_CONTAINS(os, "copy completed to");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectMoveStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_move_started_mask, out);
    auto from = gko::matrix::Dense<TypeParam>::create(exec);
    auto to = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->template on<gko::log::Logger::polymorphic_object_move_started>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    GKO_ASSERT_STR_CONTAINS(os, "move started to");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectMoveCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_move_completed_mask, out);
    auto from = gko::matrix::Dense<TypeParam>::create(exec);
    auto to = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->template on<gko::log::Logger::polymorphic_object_move_completed>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    GKO_ASSERT_STR_CONTAINS(os, "move completed to");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TYPED_TEST(Stream, CatchesPolymorphicObjectDeleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::polymorphic_object_deleted_mask, out);
    auto po = gko::matrix::Dense<TypeParam>::create(exec);
    std::stringstream ptrstream;
    ptrstream << po.get();

    logger->template on<gko::log::Logger::polymorphic_object_deleted>(
        exec.get(), po.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
    GKO_ASSERT_STR_CONTAINS(os, "deleted on");
}


TYPED_TEST(Stream, CatchesLinOpApplyStarted)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_apply_started_mask, out);
    auto A = Dense::create(exec);
    auto b = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->template on<gko::log::Logger::linop_apply_started>(A.get(), b.get(),
                                                               x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "apply started on A");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TYPED_TEST(Stream, CatchesLinOpApplyStartedWithVerbose)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_apply_started_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->template on<gko::log::Logger::linop_apply_started>(A.get(), b.get(),
                                                               x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "1.1");
    GKO_ASSERT_STR_CONTAINS(os, "-2.2");
    GKO_ASSERT_STR_CONTAINS(os, "3.3");
}


TYPED_TEST(Stream, CatchesLinOpApplyCompleted)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_apply_completed_mask, out);
    auto A = Dense::create(exec);
    auto b = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->template on<gko::log::Logger::linop_apply_completed>(
        A.get(), b.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "apply completed on A");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TYPED_TEST(Stream, CatchesLinOpApplyCompletedWithVerbose)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_apply_completed_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->template on<gko::log::Logger::linop_apply_completed>(
        A.get(), b.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "1.1");
    GKO_ASSERT_STR_CONTAINS(os, "-2.2");
    GKO_ASSERT_STR_CONTAINS(os, "3.3");
}


TYPED_TEST(Stream, CatchesLinOpAdvancedApplyStarted)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_advanced_apply_started_mask, out);
    auto A = Dense::create(exec);
    auto alpha = Dense::create(exec);
    auto b = Dense::create(exec);
    auto beta = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_alpha;
    ptrstream_alpha << alpha.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_beta;
    ptrstream_beta << beta.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->template on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "advanced apply started on A");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_alpha.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_beta.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TYPED_TEST(Stream, CatchesLinOpAdvancedApplyStartedWithVerbose)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_advanced_apply_started_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->template on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "1.1");
    GKO_ASSERT_STR_CONTAINS(os, "-4.4");
    GKO_ASSERT_STR_CONTAINS(os, "-2.2");
    GKO_ASSERT_STR_CONTAINS(os, "-5.5");
    GKO_ASSERT_STR_CONTAINS(os, "3.3");
}


TYPED_TEST(Stream, CatchesLinOpAdvancedApplyCompleted)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_advanced_apply_completed_mask, out);
    auto A = Dense::create(exec);
    auto alpha = Dense::create(exec);
    auto b = Dense::create(exec);
    auto beta = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_alpha;
    ptrstream_alpha << alpha.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_beta;
    ptrstream_beta << beta.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->template on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "advanced apply completed on A");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_alpha.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_beta.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TYPED_TEST(Stream, CatchesLinOpAdvancedApplyCompletedWithVerbose)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_advanced_apply_completed_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->template on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "1.1");
    GKO_ASSERT_STR_CONTAINS(os, "-4.4");
    GKO_ASSERT_STR_CONTAINS(os, "-2.2");
    GKO_ASSERT_STR_CONTAINS(os, "-5.5");
    GKO_ASSERT_STR_CONTAINS(os, "3.3");
}


TYPED_TEST(Stream, CatchesLinopFactoryGenerateStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_factory_generate_started_mask, out);
    auto factory =
        gko::solver::Bicgstab<TypeParam>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(exec);
    auto input = factory->generate(gko::matrix::Dense<TypeParam>::create(exec));
    std::stringstream ptrstream_factory;
    ptrstream_factory << factory.get();
    std::stringstream ptrstream_input;
    ptrstream_input << input.get();

    logger->template on<gko::log::Logger::linop_factory_generate_started>(
        factory.get(), input.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "generate started for");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_factory.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_input.str());
}


TYPED_TEST(Stream, CatchesLinopFactoryGenerateCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::linop_factory_generate_completed_mask, out);
    auto factory =
        gko::solver::Bicgstab<TypeParam>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(exec);
    auto input = factory->generate(gko::matrix::Dense<TypeParam>::create(exec));
    auto output =
        factory->generate(gko::matrix::Dense<TypeParam>::create(exec));
    std::stringstream ptrstream_factory;
    ptrstream_factory << factory.get();
    std::stringstream ptrstream_input;
    ptrstream_input << input.get();
    std::stringstream ptrstream_output;
    ptrstream_output << output.get();

    logger->template on<gko::log::Logger::linop_factory_generate_completed>(
        factory.get(), input.get(), output.get());

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "generate completed for");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_factory.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_input.str());
    GKO_ASSERT_STR_CONTAINS(os, ptrstream_output.str());
}


TYPED_TEST(Stream, CatchesCriterionCheckStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::criterion_check_started_mask, out);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    std::stringstream ptrstream;
    ptrstream << criterion.get();
    std::stringstream true_in_stream;
    true_in_stream << true;

    logger->template on<gko::log::Logger::criterion_check_started>(
        criterion.get(), 1, nullptr, nullptr, nullptr, RelativeStoppingId,
        true);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "check started for");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
    GKO_ASSERT_STR_CONTAINS(os, std::to_string(RelativeStoppingId));
    GKO_ASSERT_STR_CONTAINS(os, "finalized set to " + true_in_stream.str());
}


TYPED_TEST(Stream, CatchesCriterionCheckCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask, out);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    std::stringstream ptrstream;
    ptrstream << criterion.get();
    std::stringstream true_in_stream;
    true_in_stream << true;

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 1, nullptr, nullptr, nullptr, RelativeStoppingId, true,
        &stop_status, true, true);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "check completed for");
    GKO_ASSERT_STR_CONTAINS(os, ptrstream.str());
    GKO_ASSERT_STR_CONTAINS(os, std::to_string(RelativeStoppingId));
    GKO_ASSERT_STR_CONTAINS(os, "finalized set to " + true_in_stream.str());
    GKO_ASSERT_STR_CONTAINS(os, "changed one RHS " + true_in_stream.str());
    GKO_ASSERT_STR_CONTAINS(
        os, "stopped the iteration process " + true_in_stream.str());
}


TYPED_TEST(Stream, CatchesCriterionCheckCompletedWithVerbose)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask, out, true);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    std::stringstream true_in_stream;
    true_in_stream << true;

    stop_status.get_data()->reset();
    stop_status.get_data()->stop(RelativeStoppingId);
    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 1, nullptr, nullptr, nullptr, RelativeStoppingId, true,
        &stop_status, true, true);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "Stopped: " + true_in_stream.str());
    GKO_ASSERT_STR_CONTAINS(os,
                            "with id " + std::to_string(RelativeStoppingId));
    GKO_ASSERT_STR_CONTAINS(os, "Finalized: " + true_in_stream.str());
}


TYPED_TEST(Stream, CatchesIterationsWithoutStoppingStatus)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::iteration_complete_mask, out);
    auto solver = Dense::create(exec);
    auto right_hand_side = Dense::create(exec);
    auto residual = Dense::create(exec);
    auto solution = Dense::create(exec);
    auto residual_norm = Dense::create(exec);
    auto implicit_sq_residual_norm = Dense::create(exec);
    std::stringstream ptrstream_solver;
    ptrstream_solver << solver.get();
    std::stringstream ptrstream_residual;
    ptrstream_residual << residual.get();

    logger->template on<gko::log::Logger::iteration_complete>(
        solver.get(), right_hand_side.get(), solution.get(), num_iters,
        residual.get(), residual_norm.get(), implicit_sq_residual_norm.get(),
        nullptr, false);

    GKO_ASSERT_STR_CONTAINS(out.str(),
                            "iteration " + std::to_string(num_iters));
    GKO_ASSERT_STR_CONTAINS(out.str(), ptrstream_solver.str());
    GKO_ASSERT_STR_CONTAINS(out.str(), ptrstream_residual.str());
}


TYPED_TEST(Stream, CatchesIterationsWithStoppingStatus)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::iteration_complete_mask, out);
    auto solver = Dense::create(exec);
    auto right_hand_side = Dense::create(exec);
    auto residual = Dense::create(exec);
    auto solution = Dense::create(exec);
    auto residual_norm = Dense::create(exec);
    auto implicit_sq_residual_norm = Dense::create(exec);
    std::stringstream ptrstream_solver;
    ptrstream_solver << solver.get();
    std::stringstream ptrstream_residual;
    ptrstream_residual << residual.get();
    gko::array<gko::stopping_status> stop_status(exec, 1);

    logger->template on<gko::log::Logger::iteration_complete>(
        solver.get(), right_hand_side.get(), solution.get(), num_iters,
        residual.get(), residual_norm.get(), implicit_sq_residual_norm.get(),
        &stop_status, true);

    GKO_ASSERT_STR_CONTAINS(out.str(),
                            "iteration " + std::to_string(num_iters));
    GKO_ASSERT_STR_CONTAINS(out.str(), ptrstream_solver.str());
    GKO_ASSERT_STR_CONTAINS(out.str(), ptrstream_residual.str());
    GKO_ASSERT_STR_CONTAINS(out.str(), "Stopped the iteration process true");
}


TYPED_TEST(Stream, CatchesIterationsWithVerbose)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<TypeParam>::create(
        gko::log::Logger::iteration_complete_mask, out, true);

    auto factory =
        gko::solver::Bicgstab<TypeParam>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(exec);
    auto solver = factory->generate(gko::initialize<Dense>({1.1}, exec));
    auto right_hand_side = gko::initialize<Dense>({-5.5}, exec);
    auto residual = gko::initialize<Dense>({-4.4}, exec);
    auto solution = gko::initialize<Dense>({-2.2}, exec);
    auto residual_norm = gko::initialize<Dense>({-3.3}, exec);
    gko::array<gko::stopping_status> stop_status(exec, 1);

    logger->template on<gko::log::Logger::iteration_complete>(
        solver.get(), right_hand_side.get(), solution.get(), num_iters,
        residual.get(), residual_norm.get(), nullptr, &stop_status, true);

    auto os = out.str();
    GKO_ASSERT_STR_CONTAINS(os, "-5.5");
    GKO_ASSERT_STR_CONTAINS(os, "-4.4");
    GKO_ASSERT_STR_CONTAINS(os, "-2.2");
    GKO_ASSERT_STR_CONTAINS(os, "-3.3");
    GKO_ASSERT_STR_CONTAINS(os, "Finalized:")
}


}  // namespace
