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


#include "core/test/utils.hpp"


namespace {

template <typename ValueIndexType>
class Rcm : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;

    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          rcm_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename reorder_type::Factory> rcm_factory;
};

TYPED_TEST_CASE(Rcm, gko::test::ValueIndexTypes);

TYPED_TEST(Rcm, RcmFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->rcm_factory->get_executor(), this->exec);
}

TYPED_TEST(Rcm, CanBeCopied)
{
    using v_type = typename TestFixture::v_type;
    auto rcm =
        this->rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));
    auto rcm_copy =
        this->rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}},
            this->exec));

    rcm_copy->copy_from(rcm.get());

    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_copy->get_permutation()->get_const_permutation()[2], 0);
}


TYPED_TEST(Rcm, CanBeMoved)
{
    using v_type = typename TestFixture::v_type;
    auto rcm =
        this->rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));
    auto rcm_move =
        this->rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}},
            this->exec));

    rcm->move_to(rcm_move.get());

    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_move->get_permutation()->get_const_permutation()[2], 0);
}


TYPED_TEST(Rcm, CanBeCloned)
{
    using v_type = typename TestFixture::v_type;
    auto rcm =
        this->rcm_factory->generate(gko::initialize<gko::matrix::Dense<v_type>>(
            3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
            this->exec));

    auto rcm_clone = rcm->clone();

    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[0], 2);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[1], 1);
    ASSERT_EQ(rcm_clone->get_permutation()->get_const_permutation()[2], 0);
}

TYPED_TEST(Rcm, HasSensibleDefaults)
{
    using v_type = typename TestFixture::v_type;
    using reorder_type = typename TestFixture::reorder_type;

    auto rcm = reorder_type::build()
                   .on(this->exec)
                   ->generate(gko::initialize<gko::matrix::Dense<v_type>>(
                       3, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                       this->exec));

    ASSERT_EQ(rcm->get_parameters().construct_inverse_permutation, false);
    ASSERT_EQ(rcm->get_parameters().strategy,
              gko::reorder::starting_strategy::pseudo_peripheral);
}

TYPED_TEST(Rcm, CanBeCreatedWithStartingStrategy)
{
    using v_type = typename TestFixture::v_type;
    using reorder_type = typename TestFixture::reorder_type;
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

TYPED_TEST(Rcm, CanBeCreatedWithConstructInversePermutation)
{
    using v_type = typename TestFixture::v_type;
    using reorder_type = typename TestFixture::reorder_type;
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
