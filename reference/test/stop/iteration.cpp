/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/stop/iteration.hpp>


#include <gtest/gtest.h>


namespace {


constexpr gko::size_type test_iterations = 10;


class Iteration : public ::testing::Test {
protected:
    Iteration() : exec_{gko::ReferenceExecutor::create()}
    {
        factory_ = gko::stop::Iteration::build()
                       .with_max_iters(test_iterations)
                       .on(exec_);
    }

    std::unique_ptr<gko::stop::Iteration::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Iteration, WaitsTillIteration)
{
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);

    ASSERT_FALSE(
        criterion->update()
            .num_iterations(test_iterations - 1)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_TRUE(
        criterion->update()
            .num_iterations(test_iterations)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_TRUE(
        criterion->update()
            .num_iterations(test_iterations + 1)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
}


}  // namespace
