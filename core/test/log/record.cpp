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

#include <core/log/record.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>
#include <core/base/utils.hpp>
#include <core/solver/bicgstab.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


constexpr int num_iters = 10;
const std::string apply_str = "Dummy::apply";


TEST(Record, CanGetData)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::iteration_complete_mask);

    ASSERT_EQ(logger->get().allocation_started.size(), 0);
}


TEST(Record, CatchesAllocationStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::allocation_started_mask);

    logger->on<gko::log::Logger::allocation_started>(exec.get(), 42);

    auto data = logger->get().allocation_started.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.num_bytes, 42);
    ASSERT_EQ(data.location, 0);
}


TEST(Record, CatchesAllocationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::allocation_completed_mask);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::allocation_completed>(exec.get(), 42, ptr);

    auto data = logger->get().allocation_completed.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.num_bytes, 42);
    ASSERT_EQ(data.location, ptr);
}


TEST(Record, CatchesFreeStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger =
        gko::log::Record::create(exec, gko::log::Logger::free_started_mask);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::free_started>(exec.get(), ptr);

    auto data = logger->get().free_started.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.num_bytes, 0);
    ASSERT_EQ(data.location, ptr);
}


TEST(Record, CatchesFreeCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger =
        gko::log::Record::create(exec, gko::log::Logger::free_completed_mask);
    int dummy = 1;
    auto ptr = reinterpret_cast<gko::uintptr>(&dummy);

    logger->on<gko::log::Logger::free_completed>(exec.get(), ptr);

    auto data = logger->get().free_completed.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.num_bytes, 0);
    ASSERT_EQ(data.location, ptr);
}


TEST(Record, CatchesCopyStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger =
        gko::log::Record::create(exec, gko::log::Logger::copy_started_mask);
    int dummy_from = 1;
    int dummy_to = 1;
    auto ptr_from = reinterpret_cast<gko::uintptr>(&dummy_from);
    auto ptr_to = reinterpret_cast<gko::uintptr>(&dummy_to);

    logger->on<gko::log::Logger::copy_started>(exec.get(), exec.get(), ptr_from,
                                               ptr_to, 42);

    auto data = logger->get().copy_started.back();
    auto data_from = std::get<0>(data);
    auto data_to = std::get<1>(data);
    ASSERT_EQ(data_from.exec, exec.get());
    ASSERT_EQ(data_from.num_bytes, 42);
    ASSERT_EQ(data_from.location, ptr_from);
    ASSERT_EQ(data_to.exec, exec.get());
    ASSERT_EQ(data_to.num_bytes, 42);
    ASSERT_EQ(data_to.location, ptr_to);
}


TEST(Record, CatchesCopyCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger =
        gko::log::Record::create(exec, gko::log::Logger::copy_completed_mask);
    int dummy_from = 1;
    int dummy_to = 1;
    auto ptr_from = reinterpret_cast<gko::uintptr>(&dummy_from);
    auto ptr_to = reinterpret_cast<gko::uintptr>(&dummy_to);

    logger->on<gko::log::Logger::copy_completed>(exec.get(), exec.get(),
                                                 ptr_from, ptr_to, 42);

    auto data = logger->get().copy_completed.back();
    auto data_from = std::get<0>(data);
    auto data_to = std::get<1>(data);
    ASSERT_EQ(data_from.exec, exec.get());
    ASSERT_EQ(data_from.num_bytes, 42);
    ASSERT_EQ(data_from.location, ptr_from);
    ASSERT_EQ(data_to.exec, exec.get());
    ASSERT_EQ(data_to.num_bytes, 42);
    ASSERT_EQ(data_to.location, ptr_to);
}


TEST(Record, CatchesOperationLaunched)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::operation_launched_mask);
    gko::Operation op;

    logger->on<gko::log::Logger::operation_launched>(exec.get(), &op);

    auto data = logger->get().operation_launched.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.operation, &op);
}


TEST(Record, CatchesOperationCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::operation_completed_mask);
    gko::Operation op;

    logger->on<gko::log::Logger::operation_completed>(exec.get(), &op);

    auto data = logger->get().operation_completed.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_EQ(data.operation, &op);
}


TEST(Record, CatchesPolymorphicObjectCreateStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::polymorphic_object_create_started_mask);
    auto po = gko::matrix::Dense<>::create(exec);

    logger->on<gko::log::Logger::polymorphic_object_create_started>(exec.get(),
                                                                    po.get());


    auto data = logger->get().polymorphic_object_create_started.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_MTX_NEAR(gko::as<Dense>(data.input.get()), po.get(), 0);
    ASSERT_EQ(data.output.get(), nullptr);
}


TEST(Record, CatchesPolymorphicObjectCreateCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::polymorphic_object_create_completed_mask);
    auto po = gko::matrix::Dense<>::create(exec);
    auto output = gko::matrix::Dense<>::create(exec);

    logger->on<gko::log::Logger::polymorphic_object_create_completed>(
        exec.get(), po.get(), output.get());

    auto data = logger->get().polymorphic_object_create_completed.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_MTX_NEAR(gko::as<Dense>(data.input.get()), po.get(), 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.output.get()), output.get(), 0);
}


TEST(Record, CatchesPolymorphicObjectCopyStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::polymorphic_object_copy_started_mask);
    auto from = gko::matrix::Dense<>::create(exec);
    auto to = gko::matrix::Dense<>::create(exec);

    logger->on<gko::log::Logger::polymorphic_object_copy_started>(
        exec.get(), from.get(), to.get());

    auto data = logger->get().polymorphic_object_copy_started.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_MTX_NEAR(gko::as<Dense>(data.input.get()), from.get(), 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.output.get()), to.get(), 0);
}


TEST(Record, CatchesPolymorphicObjectCopyCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::polymorphic_object_copy_completed_mask);
    auto from = gko::matrix::Dense<>::create(exec);
    auto to = gko::matrix::Dense<>::create(exec);

    logger->on<gko::log::Logger::polymorphic_object_copy_completed>(
        exec.get(), from.get(), to.get());


    auto data = logger->get().polymorphic_object_copy_completed.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_MTX_NEAR(gko::as<Dense>(data.input.get()), from.get(), 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.output.get()), to.get(), 0);
}


TEST(Record, CatchesPolymorphicObjectDeleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::polymorphic_object_deleted_mask);
    auto po = gko::matrix::Dense<>::create(exec);

    logger->on<gko::log::Logger::polymorphic_object_deleted>(exec.get(),
                                                             po.get());


    auto data = logger->get().polymorphic_object_deleted.back();
    ASSERT_EQ(data.exec, exec.get());
    ASSERT_MTX_NEAR(gko::as<Dense>(data.input.get()), po.get(), 0);
    ASSERT_EQ(data.output, nullptr);
}


TEST(Record, CatchesLinOpApplyStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_apply_started_mask);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_apply_started>(A.get(), b.get(),
                                                      x.get());

    auto data = logger->get().linop_apply_started.back();
    ASSERT_MTX_NEAR(gko::as<Dense>(data.A.get()), A, 0);
    ASSERT_EQ(data.alpha, nullptr);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.b.get()), b, 0);
    ASSERT_EQ(data.beta, nullptr);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.x.get()), x, 0);
}


TEST(Record, CatchesLinOpApplyCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_apply_completed_mask);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_apply_completed>(A.get(), b.get(),
                                                        x.get());

    auto data = logger->get().linop_apply_completed.back();
    ASSERT_MTX_NEAR(gko::as<Dense>(data.A.get()), A, 0);
    ASSERT_EQ(data.alpha, nullptr);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.b.get()), b, 0);
    ASSERT_EQ(data.beta, nullptr);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.x.get()), x, 0);
}


TEST(Record, CatchesLinOpAdvancedApplyStarted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_advanced_apply_started_mask);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto data = logger->get().linop_advanced_apply_started.back();
    ASSERT_MTX_NEAR(gko::as<Dense>(data.A.get()), A, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.alpha.get()), alpha, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.b.get()), b, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.beta.get()), beta, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.x.get()), x, 0);
}


TEST(Record, CatchesLinOpAdvancedApplyCompleted)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_advanced_apply_completed_mask);
    auto A = gko::initialize<Dense>({1.1}, exec);
    auto alpha = gko::initialize<Dense>({-4.4}, exec);
    auto b = gko::initialize<Dense>({-2.2}, exec);
    auto beta = gko::initialize<Dense>({-5.5}, exec);
    auto x = gko::initialize<Dense>({3.3}, exec);

    logger->on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), alpha.get(), b.get(), beta.get(), x.get());

    auto data = logger->get().linop_advanced_apply_completed.back();
    ASSERT_MTX_NEAR(gko::as<Dense>(data.A.get()), A, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.alpha.get()), alpha, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.b.get()), b, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.beta.get()), beta, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.x.get()), x, 0);
}


TEST(Record, CatchesLinopFactoryGenerateStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_factory_generate_started_mask);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    auto input = factory->generate(gko::matrix::Dense<>::create(exec));

    logger->on<gko::log::Logger::linop_factory_generate_started>(factory.get(),
                                                                 input.get());

    auto data = logger->get().linop_factory_generate_started.back();
    ASSERT_EQ(data.factory, factory.get());
    ASSERT_NE(data.input.get(), nullptr);
    ASSERT_EQ(data.output.get(), nullptr);
}


TEST(Record, CatchesLinopFactoryGenerateCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::linop_factory_generate_completed_mask);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    auto input = factory->generate(gko::matrix::Dense<>::create(exec));
    auto output = factory->generate(gko::matrix::Dense<>::create(exec));

    logger->on<gko::log::Logger::linop_factory_generate_completed>(
        factory.get(), input.get(), output.get());

    auto data = logger->get().linop_factory_generate_completed.back();
    ASSERT_EQ(data.factory, factory.get());
    ASSERT_NE(data.input.get(), nullptr);
    ASSERT_NE(data.output.get(), nullptr);
}


TEST(Record, CatchesCriterionCheckStarted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::criterion_check_started_mask);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};

    logger->on<gko::log::Logger::criterion_check_started>(
        criterion.get(), RelativeStoppingId, true);

    auto data = logger->get().criterion_check_started.back();
    ASSERT_NE(data.updater, nullptr);
    ASSERT_EQ(data.stoppingId, RelativeStoppingId);
    ASSERT_EQ(data.setFinalized, true);
    ASSERT_EQ(data.oneChanged, false);
    ASSERT_EQ(data.converged, false);
}


TEST(Record, CatchesCriterionCheckCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::criterion_check_completed_mask);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::Array<gko::stopping_status> stop_status(exec, 1);

    logger->on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), RelativeStoppingId, true, &stop_status, true, true);

    stop_status.get_data()->clear();
    stop_status.get_data()->stop(RelativeStoppingId);
    auto data = logger->get().criterion_check_completed.back();
    ASSERT_NE(data.updater, nullptr);
    ASSERT_EQ(data.stoppingId, RelativeStoppingId);
    ASSERT_EQ(data.setFinalized, true);
    ASSERT_EQ(data.status->get_const_data()->has_stopped(), true);
    ASSERT_EQ(data.status->get_const_data()->get_id(),
              stop_status.get_const_data()->get_id());
    ASSERT_EQ(data.status->get_const_data()->is_finalized(), true);
    ASSERT_EQ(data.oneChanged, true);
    ASSERT_EQ(data.converged, true);
}


TEST(Record, CatchesIterations)
{
    using Dense = gko::matrix::Dense<>;
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Record::create(
        exec, gko::log::Logger::iteration_complete_mask);
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

    auto data = logger->get().iteration_completed.back();
    ASSERT_NE(data.solver.get(), nullptr);
    ASSERT_EQ(data.num_iterations, num_iters);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.residual.get()), residual, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.solution.get()), solution, 0);
    ASSERT_MTX_NEAR(gko::as<Dense>(data.residual_norm.get()), residual_norm, 0);
}


}  // namespace
