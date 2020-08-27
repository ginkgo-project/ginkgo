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

#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Rcm : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;

    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          rcm_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<reorder_type::Factory> rcm_factory;
};

TEST_F(Rcm, RcmFactoryKnowsItsExecutor)
{
    ASSERT_EQ(rcm_factory->get_executor(), this->exec);
}

TEST_F(Rcm, CanBeCopied)
{
    auto rcm =
        rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));
    auto rcm_copy =
        rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}},
            this->exec));

    rcm_copy->copy_from(rcm.get());

    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[2], 0);
    ASSERT_NE(rcm->get_adjacency_matrix(), rcm_copy->get_adjacency_matrix());
}


TEST_F(Rcm, CanBeMoved)
{
    auto rcm =
        rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));
    auto rcm_move =
        rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}},
            this->exec));

    rcm->move_to(rcm_move.get());

    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[2], 0);
}


TEST_F(Rcm, CanBeCloned)
{
    auto rcm =
        rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));

    auto rcm_clone = rcm->clone();

    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[2], 0);
    ASSERT_NE(rcm->get_adjacency_matrix(), rcm_clone->get_adjacency_matrix());
}

TEST_F(Rcm, HasSensibleDefaults)
{
    auto rcm = reorder_type::build()
                   .on(this->exec)
                   ->generate(gko::initialize<gko::matrix::Dense<v_type>>(
                       3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                       this->exec));

    ASSERT_EQ(rcm->get_parameters().construct_inverse_permutation, false);
    ASSERT_EQ(rcm->get_parameters().strategy,
              gko::reorder::starting_strategy::pseudo_peripheral);
}

TEST_F(Rcm, CanBeCreatedWithStartingStrategy)
{
    auto rcm =
        reorder_type::build()
            .with_strategy(gko::reorder::starting_strategy::minimum_degree)
            .on(this->exec)
            ->generate(gko::initialize<gko::matrix::Dense<v_type>>(
                3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                this->exec));

    ASSERT_EQ(rcm->get_parameters().strategy,
              gko::reorder::starting_strategy::minimum_degree);
}

TEST_F(Rcm, CanBeCreatedWithConstructInversePermutation)
{
    auto rcm = reorder_type::build()
                   .with_construct_inverse_permutation(true)
                   .on(this->exec)
                   ->generate(gko::initialize<gko::matrix::Dense<v_type>>(
                       3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                       this->exec));

    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm->get_inverse_permutation()->get_const_permutation()[2], 0);
    ASSERT_EQ(rcm->get_parameters().construct_inverse_permutation, true);
}

}  // namespace
