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

#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


namespace {


class BatchIluFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using batch_ilu_prec =
        gko::preconditioner::BatchIlu<value_type, index_type>;

    BatchIluFactory()
        : exec(gko::ReferenceExecutor::create()),
          type(gko::preconditioner::batch_ilu_type::parilu),
          skip_sorting(true),
          parilu_num_sweeps(20)
    {}

    std::shared_ptr<const gko::Executor> exec;
    const enum gko::preconditioner::batch_ilu_type type;
    const bool skip_sorting;
    const int parilu_num_sweeps;
};


TEST_F(BatchIluFactory, KnowsItsExecutor)
{
    auto batch_ilu_factory = batch_ilu_prec::build().on(this->exec);

    ASSERT_EQ(batch_ilu_factory->get_executor(), this->exec);
}


TEST_F(BatchIluFactory, CanSetIluType)
{
    auto batch_ilu_factory =
        batch_ilu_prec::build().with_ilu_type(this->type).on(this->exec);

    ASSERT_EQ(batch_ilu_factory->get_parameters().ilu_type, this->type);
}

TEST_F(BatchIluFactory, CanSetSorting)
{
    auto batch_ilu_factory = batch_ilu_prec::build()
                                 .with_skip_sorting(this->skip_sorting)
                                 .on(this->exec);

    ASSERT_EQ(batch_ilu_factory->get_parameters().skip_sorting,
              this->skip_sorting);
}


TEST_F(BatchIluFactory, CanSetNumSweeps)
{
    auto batch_ilu_factory =
        batch_ilu_prec::build()
            .with_parilu_num_sweeps(this->parilu_num_sweeps)
            .on(this->exec);

    ASSERT_EQ(batch_ilu_factory->get_parameters().parilu_num_sweeps,
              this->parilu_num_sweeps);
}

}  // namespace
