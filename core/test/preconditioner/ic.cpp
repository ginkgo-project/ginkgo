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

#include <ginkgo/core/preconditioner/ic.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


namespace {


class IcFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using ic_prec_type = gko::preconditioner::Ic<solver_type>;
    using ic_type = gko::factorization::ParIc<value_type, index_type>;

    IcFactory()
        : exec(gko::ReferenceExecutor::create()),
          l_factory(solver_type::build().on(exec)),
          fact_factory(ic_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename solver_type::Factory> l_factory;
    std::shared_ptr<typename ic_type::Factory> fact_factory;
};


TEST_F(IcFactory, KnowsItsExecutor)
{
    auto ic_factory = ic_prec_type::build().on(this->exec);

    ASSERT_EQ(ic_factory->get_executor(), this->exec);
}


TEST_F(IcFactory, CanSetLSolverFactory)
{
    auto ic_factory = ic_prec_type::build()
                          .with_l_solver_factory(this->l_factory)
                          .on(this->exec);

    ASSERT_EQ(ic_factory->get_parameters().l_solver_factory, this->l_factory);
}


TEST_F(IcFactory, CanSetFactorizationFactory)
{
    auto ic_factory = ic_prec_type::build()
                          .with_factorization_factory(this->fact_factory)
                          .on(this->exec);

    ASSERT_EQ(ic_factory->get_parameters().factorization_factory,
              this->fact_factory);
}


}  // namespace
