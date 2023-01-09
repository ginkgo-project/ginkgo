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

#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


namespace {


class BatchIluIsaiFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using batch_ilu_isai_prec =
        gko::preconditioner::BatchIluIsai<value_type, index_type>;

    BatchIluIsaiFactory()
        : exec(gko::ReferenceExecutor::create()),
          apply_type(gko::preconditioner::batch_ilu_isai_apply::
                         relaxation_steps_isai_simple),
          ilu_type(gko::preconditioner::batch_ilu_type::parilu),
          parilu_num_sweeps(20),
          skip_sorting(true),
          lower_factor_isai_spy_power(2),
          upper_factor_isai_spy_power(3),
          num_relaxation_steps(5)
    {}

    std::shared_ptr<const gko::Executor> exec;
    const enum gko::preconditioner::batch_ilu_isai_apply apply_type;
    const enum gko::preconditioner::batch_ilu_type ilu_type;
    const int parilu_num_sweeps;
    const bool skip_sorting;
    const int lower_factor_isai_spy_power;
    const int upper_factor_isai_spy_power;
    const int num_relaxation_steps;
};

TEST_F(BatchIluIsaiFactory, KnowsItsExecutor)
{
    auto batch_ilu_isai_factory = batch_ilu_isai_prec::build().on(this->exec);
    ASSERT_EQ(batch_ilu_isai_factory->get_executor(), this->exec);
}

TEST_F(BatchIluIsaiFactory, CanSetApplyType)
{
    auto batch_ilu_isai_factory =
        batch_ilu_isai_prec::build()
            .with_apply_type(gko::preconditioner::batch_ilu_isai_apply::
                                 relaxation_steps_isai_simple)
            .on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters().apply_type,
              this->apply_type);
}

TEST_F(BatchIluIsaiFactory, CanSetIluType)
{
    auto batch_ilu_isai_factory =
        batch_ilu_isai_prec::build()
            .with_ilu_type(gko::preconditioner::batch_ilu_type::parilu)
            .on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters().ilu_type,
              this->ilu_type);
}

TEST_F(BatchIluIsaiFactory, CanSetNumSweeps)
{
    auto batch_ilu_isai_factory =
        batch_ilu_isai_prec::build().with_parilu_num_sweeps(20).on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters().parilu_num_sweeps,
              this->parilu_num_sweeps);
}

TEST_F(BatchIluIsaiFactory, CanSetSorting)
{
    auto batch_ilu_isai_factory =
        batch_ilu_isai_prec::build().with_skip_sorting(true).on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters().skip_sorting,
              this->skip_sorting);
}

TEST_F(BatchIluIsaiFactory, CanSetLowerIsaiSpy)
{
    auto batch_ilu_isai_factory = batch_ilu_isai_prec::build()
                                      .with_lower_factor_isai_sparsity_power(2)
                                      .on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters()
                  .lower_factor_isai_sparsity_power,
              this->lower_factor_isai_spy_power);
}


TEST_F(BatchIluIsaiFactory, CanSetUpperIsaiSpy)
{
    auto batch_ilu_isai_factory = batch_ilu_isai_prec::build()
                                      .with_upper_factor_isai_sparsity_power(3)
                                      .on(this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters()
                  .upper_factor_isai_sparsity_power,
              this->upper_factor_isai_spy_power);
}

TEST_F(BatchIluIsaiFactory, CanSetRelaxationSteps)
{
    auto batch_ilu_isai_factory =
        batch_ilu_isai_prec::build().with_num_relaxation_steps(5).on(
            this->exec);

    ASSERT_EQ(batch_ilu_isai_factory->get_parameters().num_relaxation_steps,
              this->num_relaxation_steps);
}


}  // namespace
