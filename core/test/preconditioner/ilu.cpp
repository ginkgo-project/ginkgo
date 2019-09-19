/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/ilu.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class IluFactory : public ::testing::Test {
protected:
    using value_type = double;
    using Mtx = gko::matrix::Dense<value_type>;
    using default_ilu = gko::preconditioner::Ilu<>;
    using l_solver_type = gko::solver::Bicgstab<value_type>;
    using u_solver_type = gko::solver::Bicgstab<value_type>;
    using ilu_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, false>;
    static constexpr int size_matrix{10};

    IluFactory()
        : exec(gko::ReferenceExecutor::create()),
          l_factory(l_solver_type::build().on(exec)),
          u_factory(u_solver_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<l_solver_type::Factory> l_factory;
    std::shared_ptr<u_solver_type::Factory> u_factory;
};


TEST_F(IluFactory, KnowsItsExecutor)
{
    auto ilu_factory = ilu_prec_type::build().on(exec);

    ASSERT_EQ(ilu_factory->get_executor(), exec);
}


TEST_F(IluFactory, CanSetLSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_l_solver_factory(l_factory).on(exec);

    ASSERT_EQ(ilu_factory->get_parameters().l_solver_factory, l_factory);
}


TEST_F(IluFactory, CanSetUSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_u_solver_factory(u_factory).on(exec);

    ASSERT_EQ(ilu_factory->get_parameters().u_solver_factory, u_factory);
}


}  // namespace
