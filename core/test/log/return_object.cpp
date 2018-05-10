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

#include <core/log/logger.hpp>
#include <core/log/return_object.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>


namespace {


const int NUM_ITERS = 10;


struct DummyLoggedClass : public gko::log::EnableLogging {
    void apply()
    {
        this->log<gko::log::Logger::iteration_complete>(NUM_ITERS);
        this->log<gko::log::Logger::converged>(NUM_ITERS, nullptr);
    }

    void apply_with_data()
    {
        auto exec = gko::ReferenceExecutor::create();
        auto mtx =
            gko::initialize<gko::matrix::Dense<>>(4, {{1.0, 2.0, 3.0}}, exec);
        this->log<gko::log::Logger::converged>(NUM_ITERS, mtx.get());
    }
};


TEST(ReturnObject, CanGetData)
{
    auto logger = gko::log::ReturnObject::create(
        gko::log::Logger::iteration_complete_mask);

    ASSERT_TRUE(logger->get_return_object() != nullptr);
    ASSERT_TRUE(logger->get_return_object()->num_iterations == 0);
}


TEST(ReturnObject, CatchesIterations)
{
    DummyLoggedClass c;
    auto logger = gko::log::ReturnObject::create(
        gko::log::Logger::iteration_complete_mask);

    c.add_logger(logger);
    c.apply();

    ASSERT_EQ(NUM_ITERS, logger->get_return_object()->num_iterations);
}


TEST(ReturnObject, CatchesConvergence)
{
    DummyLoggedClass c;
    auto logger =
        gko::log::ReturnObject::create(gko::log::Logger::converged_mask);

    c.add_logger(logger);
    c.apply();

    ASSERT_EQ(NUM_ITERS, logger->get_return_object()->converged_at_iteration);
    ASSERT_TRUE(logger->get_return_object()->residual == nullptr);

    c.apply_with_data();
    ASSERT_EQ(NUM_ITERS, logger->get_return_object()->converged_at_iteration);
    auto residual = logger->get_return_object()->residual.get();
    ASSERT_EQ(residual->at(0), 1.0);
    ASSERT_EQ(residual->at(1), 2.0);
    ASSERT_EQ(residual->at(2), 3.0);
}


}  // namespace
