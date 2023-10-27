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

#include <ginkgo/core/base/abstract_factory.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>


// Note: the following Factory structure is not identical to Ginkgo Factory
// struture, but it is easier setup without too much dependences and
// inheritances.

struct DummyFactory {
    struct param
        : public gko::solver::enable_iterative_solver_factory_parameters<
              param, DummyFactory> {};

    const param& get_parameters() const noexcept { return parameters_; };

    void add_logger(std::shared_ptr<const gko::log::Logger> logger) {}

    DummyFactory(std::shared_ptr<const gko::Executor>, const param& parameters)
        : parameters_(parameters)
    {}

private:
    param parameters_;
};


using DF = DummyFactory;


TEST(IterativeFactoryParameter, WithCriteria)
{
    auto exec = gko::ReferenceExecutor::create();
    std::vector<std::shared_ptr<gko::stop::CriterionFactory>> criteria{
        gko::stop::Iteration::build().on(exec)};
    DummyFactory::param{}.with_criteria(criteria);
}
