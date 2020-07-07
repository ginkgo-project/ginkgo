/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/log/papi.hpp>


#include <stdexcept>


#include <gtest/gtest.h>
#include <papi.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Papi : public ::testing::Test {
protected:
    using Dense = gko::matrix::Dense<T>;

    Papi() : exec(gko::ReferenceExecutor::create()), eventset(PAPI_NULL) {}

    void SetUp()
    {
        int ret_val = PAPI_library_init(PAPI_VER_CURRENT);
        if (PAPI_VER_CURRENT != ret_val) {
            throw std::runtime_error("Error at PAPI_library_init()");
        }
        ret_val = PAPI_create_eventset(&eventset);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_create_eventset()");
        }
    }

    void TearDown() { eventset = PAPI_NULL; }

    template <typename U>
    const std::string init(const gko::log::Logger::mask_type &event,
                           const std::string &event_name, U *ptr)
    {
        logger = gko::log::Papi<T>::create(exec, event);
        std::ostringstream os;
        os << "sde:::" << logger->get_handle_name() << "::" << event_name
           << "::" << reinterpret_cast<gko::uintptr>(ptr);
        return os.str();
    }

    void add_event(const std::string &event_name)
    {
        int code;
        int ret_val = PAPI_event_name_to_code(event_name.c_str(), &code);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_name_to_code()\n");
        }

        ret_val = PAPI_add_event(eventset, code);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_name_to_code()\n");
        }
    }

    void start()
    {
        int ret_val = PAPI_start(eventset);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_start()\n");
        }
    }

    void stop(long long int *values)
    {
        int ret_val = PAPI_stop(eventset, values);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_stop()\n");
        }
    }

    std::shared_ptr<const gko::log::Papi<T>> logger;
    std::shared_ptr<const gko::Executor> exec;
    int eventset;
};

TYPED_TEST_CASE(Papi, gko::test::ValueTypes);


TYPED_TEST(Papi, CatchesAllocationStarted)
{
    int logged_value = 42;
    auto str = this->init(gko::log::Logger::allocation_started_mask,
                          "allocation_started", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::allocation_started>(
        this->exec.get(), logged_value);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, logged_value);
}


TYPED_TEST(Papi, CatchesAllocationCompleted)
{
    int logged_value = 42;
    auto str = this->init(gko::log::Logger::allocation_completed_mask,
                          "allocation_completed", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::allocation_completed>(
        this->exec.get(), logged_value, 0);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, logged_value);
}


TYPED_TEST(Papi, CatchesFreeStarted)
{
    auto str = this->init(gko::log::Logger::free_started_mask, "free_started",
                          this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::free_started>(this->exec.get(),
                                                              0);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesFreeCompleted)
{
    auto str = this->init(gko::log::Logger::free_completed_mask,
                          "free_completed", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::free_completed>(
        this->exec.get(), 0);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesCopyStarted)
{
    auto logged_value = 42;
    auto str = this->init(gko::log::Logger::copy_started_mask,
                          "copy_started_from", this->exec.get());
    std::ostringstream os_out;
    os_out << "sde:::" << this->logger->get_handle_name()
           << "::copy_started_to::"
           << reinterpret_cast<gko::uintptr>(this->exec.get());
    this->add_event(str);
    this->add_event(os_out.str());

    this->start();
    this->logger->template on<gko::log::Logger::copy_started>(
        this->exec.get(), this->exec.get(), 0, 0, logged_value);
    long long int values[2];
    this->stop(values);

    ASSERT_EQ(values[0], logged_value);
    ASSERT_EQ(values[1], logged_value);
}


TYPED_TEST(Papi, CatchesCopyCompleted)
{
    auto logged_value = 42;
    auto str = this->init(gko::log::Logger::copy_completed_mask,
                          "copy_completed_from", this->exec.get());
    std::ostringstream os_out;
    os_out << "sde:::" << this->logger->get_handle_name()
           << "::copy_completed_to::"
           << reinterpret_cast<gko::uintptr>(this->exec.get());
    this->add_event(str);
    this->add_event(os_out.str());

    this->start();
    this->logger->template on<gko::log::Logger::copy_completed>(
        this->exec.get(), this->exec.get(), 0, 0, logged_value);
    long long int values[2];
    this->stop(values);

    ASSERT_EQ(values[0], logged_value);
    ASSERT_EQ(values[1], logged_value);
}


TYPED_TEST(Papi, CatchesOperationLaunched)
{
    auto str = this->init(gko::log::Logger::operation_launched_mask,
                          "operation_launched", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::operation_launched>(
        this->exec.get(), nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesOperationCompleted)
{
    auto str = this->init(gko::log::Logger::operation_completed_mask,
                          "operation_completed", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::operation_completed>(
        this->exec.get(), nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesPolymorphicObjectCreateStarted)
{
    auto str =
        this->init(gko::log::Logger::polymorphic_object_create_started_mask,
                   "polymorphic_object_create_started", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger
        ->template on<gko::log::Logger::polymorphic_object_create_started>(
            this->exec.get(), nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesPolymorphicObjectCreateCompleted)
{
    auto str =
        this->init(gko::log::Logger::polymorphic_object_create_completed_mask,
                   "polymorphic_object_create_completed", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger
        ->template on<gko::log::Logger::polymorphic_object_create_completed>(
            this->exec.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesPolymorphicObjectCopyStarted)
{
    auto str =
        this->init(gko::log::Logger::polymorphic_object_copy_started_mask,
                   "polymorphic_object_copy_started", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger
        ->template on<gko::log::Logger::polymorphic_object_copy_started>(
            this->exec.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesPolymorphicObjectCopyCompleted)
{
    auto str =
        this->init(gko::log::Logger::polymorphic_object_copy_completed_mask,
                   "polymorphic_object_copy_completed", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger
        ->template on<gko::log::Logger::polymorphic_object_copy_completed>(
            this->exec.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesPolymorphicObjectDeleted)
{
    auto str = this->init(gko::log::Logger::polymorphic_object_deleted_mask,
                          "polymorphic_object_deleted", this->exec.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::polymorphic_object_deleted>(
        this->exec.get(), nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpApplyStarted)
{
    using Dense = typename TestFixture::Dense;
    auto A = Dense::create(this->exec);
    auto str = this->init(gko::log::Logger::linop_apply_started_mask,
                          "linop_apply_started", A.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::linop_apply_started>(
        A.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpApplyCompleted)
{
    using Dense = typename TestFixture::Dense;
    auto A = Dense::create(this->exec);
    auto str = this->init(gko::log::Logger::linop_apply_completed_mask,
                          "linop_apply_completed", A.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::linop_apply_completed>(
        A.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpAdvancedApplyStarted)
{
    using Dense = typename TestFixture::Dense;
    auto A = Dense::create(this->exec);
    auto str = this->init(gko::log::Logger::linop_advanced_apply_started_mask,
                          "linop_advanced_apply_started", A.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::linop_advanced_apply_started>(
        A.get(), nullptr, nullptr, nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpAdvancedApplyCompleted)
{
    using Dense = typename TestFixture::Dense;
    auto A = Dense::create(this->exec);
    auto str = this->init(gko::log::Logger::linop_advanced_apply_completed_mask,
                          "linop_advanced_apply_completed", A.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::linop_advanced_apply_completed>(
        A.get(), nullptr, nullptr, nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpFactoryGenerateStarted)
{
    auto factory =
        gko::solver::Bicgstab<TypeParam>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    auto str = this->init(gko::log::Logger::linop_factory_generate_started_mask,
                          "linop_factory_generate_started", factory.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::linop_factory_generate_started>(
        factory.get(), nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesLinOpFactoryGenerateCompleted)
{
    auto factory =
        gko::solver::Bicgstab<TypeParam>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    TypeParam dummy;
    auto str =
        this->init(gko::log::Logger::linop_factory_generate_completed_mask,
                   "linop_factory_generate_completed", factory.get());
    this->add_event(str);

    this->start();
    this->logger
        ->template on<gko::log::Logger::linop_factory_generate_completed>(
            factory.get(), nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, 1);
}


TYPED_TEST(Papi, CatchesIterationComplete)
{
    using Dense = typename TestFixture::Dense;
    int logged_value = 42;
    auto A = Dense::create(this->exec);
    auto str = this->init(gko::log::Logger::iteration_complete_mask,
                          "iteration_complete", A.get());
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::iteration_complete>(
        A.get(), 42, nullptr, nullptr, nullptr);
    long long int value = 0;
    this->stop(&value);

    ASSERT_EQ(value, logged_value);
}


}  // namespace
