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


#include <core/log/papi.hpp>


#include <gtest/gtest.h>
#include <papi.h>


#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/solver/bicgstab.hpp>
#include <core/stop/iteration.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Papi : public ::testing::Test {
protected:
    using Dense = gko::matrix::Dense<>;

    Papi() : exec(gko::ReferenceExecutor::create()), eventset(PAPI_NULL) {}

    void SetUp()
    {
        int ret_val = PAPI_library_init(PAPI_VER_CURRENT);
        if (ret_val != PAPI_VER_CURRENT) {
            fprintf(stderr, "Error at PAPI_library_init()\n");
            exit(-1);
        }
        ret_val = PAPI_create_eventset(&eventset);
        if (PAPI_OK != ret_val) {
            fprintf(stderr, "Error at PAPI_create_eventset()\n");
            exit(-1);
        }
    }

    void TearDown() { eventset = PAPI_NULL; }

    void add_event(const std::string &event_name)
    {
        int code;
        int ret_val = PAPI_event_name_to_code(event_name.c_str(), &code);
        if (PAPI_OK != ret_val) {
            fprintf(stderr, "Error at PAPI_name_to_code()\n");
            exit(-1);
        }

        ret_val = PAPI_add_event(eventset, code);
        if (PAPI_OK != ret_val) {
            fprintf(stderr, "Error at PAPI_name_to_code()\n");
            exit(-1);
        }
    }

    void start()
    {
        int ret_val = PAPI_start(eventset);
        if (PAPI_OK != ret_val) {
            fprintf(stderr, "Error at PAPI_start()\n");
            exit(-1);
        }
    }

    void stop(long long int *values)
    {
        int ret_val = PAPI_stop(eventset, values);
        if (PAPI_OK != ret_val) {
            fprintf(stderr, "Error at PAPI_stop()\n");
            exit(-1);
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    int eventset;
};


TEST_F(Papi, CatchesAllocationStarted)
{
    int logged_value = 42;
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::allocation_started_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name() << "::allocation_started::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::allocation_started>(exec.get(), logged_value);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, logged_value);
}


TEST_F(Papi, CatchesAllocationCompleted)
{
    int logged_value = 42;
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::allocation_completed_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name() << "::allocation_completed::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::allocation_completed>(exec.get(), logged_value,
                                                       0);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, logged_value);
}


TEST_F(Papi, CatchesFreeStarted)
{
    auto logger =
        gko::log::Papi<>::create(exec, gko::log::Logger::free_started_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::free_started::" << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::free_started>(exec.get(), 0);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesFreeCompleted)
{
    auto logger =
        gko::log::Papi<>::create(exec, gko::log::Logger::free_completed_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::free_completed::" << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::free_completed>(exec.get(), 0);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesCopyStarted)
{
    auto logged_value = 42;
    auto logger =
        gko::log::Papi<>::create(exec, gko::log::Logger::copy_started_mask);
    std::ostringstream os_in;
    os_in << "sde:::" << logger->get_handle_name() << "::copy_started_from::"
          << reinterpret_cast<gko::uintptr>(exec.get());
    std::ostringstream os_out;
    os_out << "sde:::" << logger->get_handle_name() << "::copy_started_to::"
           << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os_in.str());
    add_event(os_out.str());
    start();
    logger->on<gko::log::Logger::copy_started>(exec.get(), exec.get(), 0, 0,
                                               logged_value);
    long long int values[2];
    stop(values);

    ASSERT_EQ(values[0], logged_value);
    ASSERT_EQ(values[1], logged_value);
}


TEST_F(Papi, CatchesCopyCompleted)
{
    auto logged_value = 42;
    auto logger =
        gko::log::Papi<>::create(exec, gko::log::Logger::copy_completed_mask);
    std::ostringstream os_in;
    os_in << "sde:::" << logger->get_handle_name() << "::copy_completed_from::"
          << reinterpret_cast<gko::uintptr>(exec.get());
    std::ostringstream os_out;
    os_out << "sde:::" << logger->get_handle_name() << "::copy_completed_to::"
           << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os_in.str());
    add_event(os_out.str());
    start();
    logger->on<gko::log::Logger::copy_completed>(exec.get(), exec.get(), 0, 0,
                                                 logged_value);
    long long int values[2];
    stop(values);

    ASSERT_EQ(values[0], logged_value);
    ASSERT_EQ(values[1], logged_value);
}


TEST_F(Papi, CatchesOperationLaunched)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::operation_launched_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name() << "::operation_launched::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::operation_launched>(exec.get(), nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesOperationCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::operation_completed_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name() << "::operation_completed::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::operation_completed>(exec.get(), nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesPolymorphicObjectCreateStarted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::polymorphic_object_create_started_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::polymorphic_object_create_started::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::polymorphic_object_create_started>(exec.get(),
                                                                    nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesPolymorphicObjectCreateCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::polymorphic_object_create_completed_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::polymorphic_object_create_completed::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::polymorphic_object_create_completed>(
        exec.get(), nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesPolymorphicObjectCopyStarted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::polymorphic_object_copy_started_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::polymorphic_object_copy_started::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::polymorphic_object_copy_started>(
        exec.get(), nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesPolymorphicObjectCopyCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::polymorphic_object_copy_completed_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::polymorphic_object_copy_completed::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::polymorphic_object_copy_completed>(
        exec.get(), nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesPolymorphicObjectDeleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::polymorphic_object_deleted_mask);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::polymorphic_object_deleted::"
       << reinterpret_cast<gko::uintptr>(exec.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::polymorphic_object_deleted>(exec.get(),
                                                             nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpApplyStarted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_apply_started_mask);
    auto A = Dense::create(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::linop_apply_started::" << reinterpret_cast<gko::uintptr>(A.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_apply_started>(A.get(), nullptr,
                                                      nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpApplyCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_apply_completed_mask);
    auto A = Dense::create(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name() << "::linop_apply_completed::"
       << reinterpret_cast<gko::uintptr>(A.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_apply_completed>(A.get(), nullptr,
                                                        nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpAdvancedApplyStarted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_advanced_apply_started_mask);
    auto A = Dense::create(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::linop_advanced_apply_started::"
       << reinterpret_cast<gko::uintptr>(A.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), nullptr, nullptr, nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpAdvancedApplyCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_advanced_apply_completed_mask);
    auto A = Dense::create(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::linop_advanced_apply_completed::"
       << reinterpret_cast<gko::uintptr>(A.get());

    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), nullptr, nullptr, nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpFactoryGenerateStarted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_factory_generate_started_mask);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::linop_factory_generate_started::"
       << reinterpret_cast<gko::uintptr>(factory.get());


    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_factory_generate_started>(factory.get(),
                                                                 nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesLinOpFactoryGenerateCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_factory_generate_completed_mask);
    auto factory = gko::solver::Bicgstab<>::Factory::create()
                       .with_criterion(gko::stop::Iteration::Factory::create()
                                           .with_max_iters(3u)
                                           .on_executor(exec))
                       .on_executor(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::linop_factory_generate_completed::"
       << reinterpret_cast<gko::uintptr>(factory.get());


    add_event(os.str());
    start();
    logger->on<gko::log::Logger::linop_factory_generate_completed>(
        factory.get(), nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, 1);
}


TEST_F(Papi, CatchesIterationComplete)
{
    int logged_value = 42;
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::iteration_complete_mask);
    auto A = Dense::create(exec);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::iteration_complete::" << reinterpret_cast<gko::uintptr>(A.get());


    add_event(os.str());
    start();
    logger->on<gko::log::Logger::iteration_complete>(A.get(), 42, nullptr,
                                                     nullptr, nullptr);
    long long int value = 0;
    stop(&value);

    ASSERT_EQ(value, logged_value);
}


}  // namespace
