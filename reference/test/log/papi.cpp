// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <papi.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/papi.hpp>
#include <ginkgo/core/matrix/dense.hpp>
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
        if (ret_val != PAPI_VER_CURRENT) {
            throw std::runtime_error("Error at PAPI_library_init()");
        }
        ret_val = PAPI_create_eventset(&eventset);
        if (PAPI_OK != ret_val) {
            throw std::runtime_error("Error at PAPI_create_eventset()");
        }
    }

    void TearDown() { eventset = PAPI_NULL; }

    template <typename U>
    const std::string init(const gko::log::Logger::mask_type& event,
                           const std::string& event_name, U* ptr)
    {
        logger = gko::log::Papi<T>::create(event);
        std::ostringstream os;
        os << "sde:::" << logger->get_handle_name() << "::" << event_name << "_"
           << reinterpret_cast<gko::uintptr>(ptr);
        return os.str();
    }

    void add_event(const std::string& event_name)
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

    void stop(long long int* values)
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

TYPED_TEST_SUITE(Papi, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Papi, CatchesCriterionCheckCompleted)
{
    using Dense = typename TestFixture::Dense;
    auto residual_norm = gko::initialize<Dense>({4.0}, this->exec);
    auto criterion = gko::stop::Iteration::build()
                         .with_max_iters(3u)
                         .on(this->exec)
                         ->generate(nullptr, nullptr, nullptr);
    auto str = this->init(gko::log::Logger::criterion_check_completed_mask,
                          "criterion_check_completed", criterion.get());
    this->add_event(str + ":CNT");
    this->add_event(str);

    this->start();
    this->logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 0, nullptr, residual_norm.get(), nullptr, 0, false,
        nullptr, false, false);
    long long int values[2];
    this->stop(values);
    double* sde_ptr = GET_SDE_RECORDER_ADDRESS(values[1], double);

    ASSERT_EQ(values[0], 1);
    ASSERT_EQ(sde_ptr[0], 4.0);
}


}  // namespace
