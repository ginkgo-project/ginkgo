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

#include <core/log/stream.hpp>


#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>


#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/solver/bicgstab.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


constexpr int num_iters = 10;
const std::string apply_str = "DummyLoggedClass::apply";


TEST(Stream, CatchesApply)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger =
        gko::log::Stream<>::create(exec, gko::log::Logger::apply_mask, out);

    logger->on<gko::log::Logger::apply>(apply_str);

    ASSERT_STR_CONTAINS(out.str(), "starting apply function: " + apply_str);
}


TEST(Stream, CatchesIterations)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::iteration_complete_mask, out);
    auto solver = Dense::create(exec);
    auto residual = Dense::create(exec);
    auto solution = Dense::create(exec);
    auto residual_norm = Dense::create(exec);
    std::stringstream ptrstream_solver;
    ptrstream_solver << solver.get();
    std::stringstream ptrstream_residual;
    ptrstream_residual << residual.get();

    logger->on<gko::log::Logger::iteration_complete>(solver.get(), num_iters,
                                                     residual.get());

    ASSERT_STR_CONTAINS(out.str(), "iteration " + num_iters);
    ASSERT_STR_CONTAINS(out.str(), ptrstream_solver.str());
    ASSERT_STR_CONTAINS(out.str(), ptrstream_residual.str());
}


TEST(Stream, CatchesIterationsWithVerbose)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::iteration_complete_mask, out, true);

    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    auto solver = factory->generate(gko::initialize<Dense>({1.1}, exec));
    auto residual = gko::initialize<Dense>({-4.4}, exec);
    auto solution = gko::initialize<Dense>({-2.2}, exec);
    auto residual_norm = gko::initialize<Dense>({-3.3}, exec);

    logger->on<gko::log::Logger::iteration_complete>(
        solver.get(), num_iters, residual.get(), solution.get(),
        residual_norm.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "-4.4");
    ASSERT_STR_CONTAINS(os, "-2.2");
    ASSERT_STR_CONTAINS(os, "-3.3");
}


TEST(Stream, CatchesConvergence)
{
    std::stringstream out;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx =
        gko::initialize<gko::matrix::Dense<>>(4, {{1.0, 2.0, 3.0}}, exec);
    auto logger =
        gko::log::Stream<>::create(exec, gko::log::Logger::converged_mask, out);
    out << std::scientific << std::setprecision(4);

    logger->on<gko::log::Logger::converged>(num_iters, mtx.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "converged at iteration " + num_iters);
    ASSERT_STR_CONTAINS(os, "1.0");
    ASSERT_STR_CONTAINS(os, "2.0");
    ASSERT_STR_CONTAINS(os, "3.0");
}


TEST(Stream, CatchesAllocationStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::allocation_started_mask, out);

    logger->on<gko::log::Logger::allocation_started>(exec.get(), 42);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "allocation started on");
    ASSERT_STR_CONTAINS(os, "42");
}


TEST(Stream, CatchesAllocationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::allocation_completed_mask, out);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::allocation_completed>(exec.get(), 42, ptr);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "allocation completed on");
    ASSERT_STR_CONTAINS(os, "42");
    ASSERT_STR_CONTAINS(os, std::to_string(ptr));
}


TEST(Stream, CatchesFreeStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::free_started_mask, out);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::free_started>(exec.get(), ptr);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "free started on");
    ASSERT_STR_CONTAINS(os, std::to_string(ptr));
}


TEST(Stream, CatchesFreeCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::free_completed_mask, out);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::free_completed>(exec.get(), ptr);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "free completed on");
    ASSERT_STR_CONTAINS(os, std::to_string(ptr));
}


TEST(Stream, CatchesCopyStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::copy_started_mask, out);
    int dummy_in = 1;
    int dummy_out = 1;
    auto ptr_in = reinterpret_cast<gko::uintptr>(&dummy_in);
    auto ptr_out = reinterpret_cast<gko::uintptr>(&dummy_out);

    logger->on<gko::log::Logger::copy_started>(exec.get(), exec.get(), ptr_in,
                                               ptr_out, 42);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "copy started");
    ASSERT_STR_CONTAINS(os, "from Location[" + std::to_string(ptr_in));
    ASSERT_STR_CONTAINS(os, "to Location[" + std::to_string(ptr_out));
    ASSERT_STR_CONTAINS(os, "with Bytes[42]");
}


TEST(Stream, CatchesCopyCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::copy_completed_mask, out);
    int dummy_in = 1;
    int dummy_out = 1;
    auto ptr_in = reinterpret_cast<gko::uintptr>(&dummy_in);
    auto ptr_out = reinterpret_cast<gko::uintptr>(&dummy_out);

    logger->on<gko::log::Logger::copy_completed>(exec.get(), exec.get(), ptr_in,
                                                 ptr_out, 42);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "copy completed");
    ASSERT_STR_CONTAINS(os, "from Location[" + std::to_string(ptr_in));
    ASSERT_STR_CONTAINS(os, "to Location[" + std::to_string(ptr_out));
    ASSERT_STR_CONTAINS(os, "with Bytes[42]");
}


TEST(Stream, CatchesOperationLaunched)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::operation_launched_mask, out);
    gko::Operation op;
    std::stringstream ptrstream;
    ptrstream << &op;

    logger->on<gko::log::Logger::operation_launched>(exec.get(), &op);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "started on");
    ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TEST(Stream, CatchesOperationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::operation_completed_mask, out);
    gko::Operation op;
    std::stringstream ptrstream;
    ptrstream << &op;

    logger->on<gko::log::Logger::operation_completed>(exec.get(), &op);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "completed on");
    ASSERT_STR_CONTAINS(os, ptrstream.str());
}


TEST(Stream, CatchesPolymorphicObjectCreateStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::polymorphic_object_create_started_mask, out);
    auto po = gko::matrix::Dense<>::create(exec);
    std::stringstream ptrstream;
    ptrstream << po.get();

    logger->on<gko::log::Logger::polymorphic_object_create_started>(exec.get(),
                                                                    po.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, ptrstream.str());
    ASSERT_STR_CONTAINS(os, "create started from");
}


TEST(Stream, CatchesPolymorphicObjectCreateCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::polymorphic_object_create_completed_mask, out);
    auto po = gko::matrix::Dense<>::create(exec);
    auto output = gko::matrix::Dense<>::create(exec);
    std::stringstream ptrstream_in;
    ptrstream_in << po.get();
    std::stringstream ptrstream_out;
    ptrstream_out << output.get();

    logger->on<gko::log::Logger::polymorphic_object_create_completed>(
        exec.get(), po.get(), output.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, ptrstream_in.str());
    ASSERT_STR_CONTAINS(os, "create completed from");
    ASSERT_STR_CONTAINS(os, ptrstream_out.str());
}


TEST(Stream, CatchesPolymorphicObjectCopyStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::polymorphic_object_copy_started_mask, out);
    auto from = gko::matrix::Dense<>::create(exec);
    auto to = gko::matrix::Dense<>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->on<gko::log::Logger::polymorphic_object_copy_started>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    ASSERT_STR_CONTAINS(os, "copy started to");
    ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TEST(Stream, CatchesPolymorphicObjectCopyCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::polymorphic_object_copy_completed_mask, out);
    auto from = gko::matrix::Dense<>::create(exec);
    auto to = gko::matrix::Dense<>::create(exec);
    std::stringstream ptrstream_from;
    ptrstream_from << from.get();
    std::stringstream ptrstream_to;
    ptrstream_to << to.get();

    logger->on<gko::log::Logger::polymorphic_object_copy_completed>(
        exec.get(), from.get(), to.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, ptrstream_from.str());
    ASSERT_STR_CONTAINS(os, "copy completed to");
    ASSERT_STR_CONTAINS(os, ptrstream_to.str());
}


TEST(Stream, CatchesPolymorphicObjectDeleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::polymorphic_object_deleted_mask, out);
    auto po = gko::matrix::Dense<>::create(exec);
    std::stringstream ptrstream;
    ptrstream << po.get();

    logger->on<gko::log::Logger::polymorphic_object_deleted>(exec.get(),
                                                             po.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, ptrstream.str());
    ASSERT_STR_CONTAINS(os, "deleted on");
}


TEST(Stream, CatchesLinOpApplyStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_apply_started_mask, out);
    auto A = Dense::create(exec);
    auto b = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->on<gko::log::Logger::linop_apply_started>(A.get(), b.get(),
                                                      x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "apply started on A");
    ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TEST(Stream, CatchesLinOpApplyStartedWithVerbose)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_apply_started_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_apply_started>(A.get(), b.get(),
                                                      x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "1.1");
    ASSERT_STR_CONTAINS(os, "-2.2");
    ASSERT_STR_CONTAINS(os, "3.3");
}


TEST(Stream, CatchesLinOpApplyCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_apply_completed_mask, out);
    auto A = Dense::create(exec);
    auto b = Dense::create(exec);
    auto x = Dense::create(exec);
    std::stringstream ptrstream_A;
    ptrstream_A << A.get();
    std::stringstream ptrstream_b;
    ptrstream_b << b.get();
    std::stringstream ptrstream_x;
    ptrstream_x << x.get();

    logger->on<gko::log::Logger::linop_apply_completed>(A.get(), b.get(),
                                                        x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "apply completed on A");
    ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TEST(Stream, CatchesLinOpApplyCompletedWithVerbose)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_apply_completed_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_apply_completed>(A.get(), b.get(),
                                                        x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "1.1");
    ASSERT_STR_CONTAINS(os, "-2.2");
    ASSERT_STR_CONTAINS(os, "3.3");
}


TEST(Stream, CatchesLinOpAdvancedApplyStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_advanced_apply_started_mask, out);
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

    logger->on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "advanced apply started on A");
    ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    ASSERT_STR_CONTAINS(os, ptrstream_alpha.str());
    ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    ASSERT_STR_CONTAINS(os, ptrstream_beta.str());
    ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TEST(Stream, CatchesLinOpAdvancedApplyStartedWithVerbose)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_advanced_apply_started_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "1.1");
    ASSERT_STR_CONTAINS(os, "-4.4");
    ASSERT_STR_CONTAINS(os, "-2.2");
    ASSERT_STR_CONTAINS(os, "-5.5");
    ASSERT_STR_CONTAINS(os, "3.3");
}


TEST(Stream, CatchesLinOpAdvancedApplyCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_advanced_apply_completed_mask, out);
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

    logger->on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "advanced apply completed on A");
    ASSERT_STR_CONTAINS(os, ptrstream_A.str());
    ASSERT_STR_CONTAINS(os, ptrstream_alpha.str());
    ASSERT_STR_CONTAINS(os, ptrstream_b.str());
    ASSERT_STR_CONTAINS(os, ptrstream_beta.str());
    ASSERT_STR_CONTAINS(os, ptrstream_x.str());
}


TEST(Stream, CatchesLinOpAdvancedApplyCompletedWithVerbose)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_advanced_apply_completed_mask, out, true);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "1.1");
    ASSERT_STR_CONTAINS(os, "-4.4");
    ASSERT_STR_CONTAINS(os, "-2.2");
    ASSERT_STR_CONTAINS(os, "-5.5");
    ASSERT_STR_CONTAINS(os, "3.3");
}


TEST(Stream, CatchesLinopFactoryGenerateStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_factory_generate_started_mask, out);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    auto input = factory->generate(gko::matrix::Dense<>::create(exec));
    std::stringstream ptrstream_factory;
    ptrstream_factory << factory.get();
    std::stringstream ptrstream_input;
    ptrstream_input << input.get();

    logger->on<gko::log::Logger::linop_factory_generate_started>(factory.get(),
                                                                 input.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "generate started for");
    ASSERT_STR_CONTAINS(os, ptrstream_factory.str());
    ASSERT_STR_CONTAINS(os, ptrstream_input.str());
}


TEST(Stream, CatchesLinopFactoryGenerateCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::linop_factory_generate_completed_mask, out);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    auto input = factory->generate(gko::matrix::Dense<>::create(exec));
    auto output = factory->generate(gko::matrix::Dense<>::create(exec));
    std::stringstream ptrstream_factory;
    ptrstream_factory << factory.get();
    std::stringstream ptrstream_input;
    ptrstream_input << input.get();
    std::stringstream ptrstream_output;
    ptrstream_output << output.get();

    logger->on<gko::log::Logger::linop_factory_generate_completed>(
        factory.get(), input.get(), output.get());

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "generate completed for");
    ASSERT_STR_CONTAINS(os, ptrstream_factory.str());
    ASSERT_STR_CONTAINS(os, ptrstream_input.str());
    ASSERT_STR_CONTAINS(os, ptrstream_output.str());
}


TEST(Stream, CatchesCriterionCheckStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::criterion_check_started_mask, out);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    std::stringstream ptrstream;
    ptrstream << criterion.get();
    std::stringstream true_in_stream;
    true_in_stream << true;

    logger->on<gko::log::Logger::criterion_check_started>(
        criterion.get(), RelativeStoppingId, true);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "check started for");
    ASSERT_STR_CONTAINS(os, ptrstream.str());
    ASSERT_STR_CONTAINS(os, std::to_string(RelativeStoppingId));
    ASSERT_STR_CONTAINS(os, "finalized set to " + true_in_stream.str());
}


TEST(Stream, CatchesCriterionCheckCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::criterion_check_completed_mask, out);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::Array<gko::stopping_status> stop_status(exec, 1);
    std::stringstream ptrstream;
    ptrstream << criterion.get();
    std::stringstream true_in_stream;
    true_in_stream << true;

    logger->on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), RelativeStoppingId, true, &stop_status, true, true);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "check completed for");
    ASSERT_STR_CONTAINS(os, ptrstream.str());
    ASSERT_STR_CONTAINS(os, std::to_string(RelativeStoppingId));
    ASSERT_STR_CONTAINS(os, "finalized set to " + true_in_stream.str());
    ASSERT_STR_CONTAINS(os, "changed one RHS " + true_in_stream.str());
    ASSERT_STR_CONTAINS(
        os, "stopped the iteration process " + true_in_stream.str());
}


TEST(Stream, CatchesCriterionCheckCompletedWithVerbose)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::criterion_check_completed_mask, out, true);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::Array<gko::stopping_status> stop_status(exec, 1);
    std::stringstream true_in_stream;
    true_in_stream << true;

    stop_status.get_data()->clear();
    stop_status.get_data()->stop(RelativeStoppingId);
    logger->on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), RelativeStoppingId, true, &stop_status, true, true);

    auto os = out.str();
    ASSERT_STR_CONTAINS(os, "Stopped: " + true_in_stream.str());
    ASSERT_STR_CONTAINS(os, "with id " + std::to_string(RelativeStoppingId));
    ASSERT_STR_CONTAINS(os, "Finalized: " + true_in_stream.str());
}


}  // namespace
