/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <gtest/gtest.h>


#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "test/utils/executor.hpp"


constexpr gko::size_type test_iterations = 10;


class Combined : public CommonTestFixture {
protected:
    Combined()
    {
        // Actually use an iteration stopping criterion because Criterion is an
        // abstract class
        factory = gko::stop::Combined::build()
                      .with_criteria(gko::stop::Iteration::build().on(ref),
                                     gko::stop::Iteration::build().on(ref),
                                     gko::stop::Iteration::build().on(ref))
                      .on(ref);
    }

    std::unique_ptr<gko::stop::Combined::Factory> factory;
};


TEST_F(Combined, CopyPropagatesExecutor)
{
    auto dev_factory = gko::clone(exec, factory.get());

    for (const auto& c : dev_factory->get_parameters().criteria) {
        ASSERT_TRUE(c->get_executor());
        ASSERT_EQ(exec.get(), c->get_executor().get());
    }
}


TEST_F(Combined, MovePropagatesExecutor)
{
    auto dev_factory = factory->create_default(exec);

    dev_factory->move_from(factory);

    for (const auto& c : dev_factory->get_parameters().criteria) {
        ASSERT_TRUE(c->get_executor());
        ASSERT_EQ(exec.get(), c->get_executor().get());
    }
}
