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


#include <gtest/gtest.h>
#include <papi.h>


#include <ginkgo/core/base/executor.hpp>
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
    const std::string init(const gko::log::Logger::mask_type &event,
                           const std::string &event_name, U *ptr)
    {
        logger = gko::log::Papi<T>::create(exec, event);
        std::ostringstream os;
        os << "sde:::" << logger->get_handle_name() << "::" << event_name << "_"
           << reinterpret_cast<gko::uintptr>(ptr);
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
    double *sde_ptr = GET_SDE_RECORDER_ADDRESS(values[1], double);

    ASSERT_EQ(values[0], 1);
    ASSERT_EQ(sde_ptr[0], 4.0);
}


}  // namespace
