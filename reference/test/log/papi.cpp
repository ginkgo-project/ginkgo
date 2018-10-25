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


TEST_F(Papi, CatchesCriterionCheckCompleted)
{
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::criterion_check_completed_mask);
    auto criterion = gko::stop::Iteration::Factory::create()
                         .with_max_iters(3u)
                         .on_executor(exec)
                         ->generate(nullptr, nullptr, nullptr);
    std::ostringstream os;
    os << "sde:::" << logger->get_handle_name()
       << "::criterion_check_completed::"
       << reinterpret_cast<gko::uintptr>(criterion.get());
    auto residual_norm = gko::initialize<Dense>({4.0}, exec);


    logger->on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 0, nullptr, residual_norm.get(), nullptr, 0, false,
        nullptr, false, false);

    // TODO:  How do I access the record?
}


}  // namespace
