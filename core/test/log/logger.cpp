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
#include <core/log/ostream.hpp>
#include <core/log/return_object.hpp>


#include <gtest/gtest.h>
#include <iostream>
#include <memory>


namespace {


const int NUM_ITERS = 10;


struct DummyLoggedClass : public gko::log::EnableLogging {
    int get_loggers_size() { return loggers_.size(); }

    void apply() { this->log<gko::log::Logger::iteration_complete>(NUM_ITERS); }
};


TEST(DummyLogged, CanAddLoggers)
{
    DummyLoggedClass c;

    c.add_logger(
        gko::log::ReturnObject::create(gko::log::Logger::all_events_mask));
    ASSERT_EQ(c.get_loggers_size(), 1);

    c.add_logger(gko::log::Ostream::create(gko::log::Logger::all_events_mask,
                                           std::cout));
    ASSERT_EQ(c.get_loggers_size(), 2);
}


struct DummyData {
    gko::size_type num_iterations;
};


struct DummyLogger : public gko::log::Logger {
    static std::shared_ptr<DummyLogger> create(const mask_type &enabled_events)
    {
        return std::shared_ptr<DummyLogger>(new DummyLogger(enabled_events));
    }


    explicit DummyLogger(const mask_type &enabled_events)
        : Logger(enabled_events)
    {
        data_ = std::make_shared<DummyData>();
    }


    void on_iteration_complete(
        const gko::size_type num_iterations) const override;
    std::shared_ptr<DummyData> data_;
};


void DummyLogger::on_iteration_complete(
    const gko::size_type num_iterations) const
{
    data_->num_iterations = num_iterations;
}


TEST(DummyLogged, CanLogEvents)
{
    DummyLoggedClass c;
    auto logger =
        DummyLogger::create(gko::log::Logger::iteration_complete_mask);

    c.add_logger(logger);
    c.apply();

    ASSERT_EQ(NUM_ITERS, logger->data_->num_iterations);
}


}  // namespace
