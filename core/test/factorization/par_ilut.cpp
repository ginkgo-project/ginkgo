/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


namespace {


class ParIlut : public ::testing::Test {
public:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using ilut_factory_type =
        gko::factorization::ParIlut<value_type, index_type>;

protected:
    ParIlut() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};


TEST_F(ParIlut, SetIterations)
{
    auto factory = ilut_factory_type::build().with_iterations(5u).on(ref);

    ASSERT_EQ(factory->get_parameters().iterations, 5u);
}


TEST_F(ParIlut, SetSkip)
{
    auto factory = ilut_factory_type::build().with_skip_sorting(true).on(ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TEST_F(ParIlut, SetApprox)
{
    auto factory =
        ilut_factory_type::build().with_approximate_select(false).on(ref);

    ASSERT_EQ(factory->get_parameters().approximate_select, false);
}


TEST_F(ParIlut, SetDeterministic)
{
    auto factory =
        ilut_factory_type::build().with_deterministic_sample(true).on(ref);

    ASSERT_EQ(factory->get_parameters().deterministic_sample, true);
}


TEST_F(ParIlut, SetFillIn)
{
    auto factory = ilut_factory_type::build().with_fill_in_limit(1.2).on(ref);

    ASSERT_EQ(factory->get_parameters().fill_in_limit, 1.2);
}


TEST_F(ParIlut, SetDefaults)
{
    auto factory = ilut_factory_type::build().on(ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().iterations, 5u);
    ASSERT_EQ(factory->get_parameters().approximate_select, true);
    ASSERT_EQ(factory->get_parameters().deterministic_sample, false);
    ASSERT_EQ(factory->get_parameters().fill_in_limit, 2.0);
}


TEST_F(ParIlut, SetEverything)
{
    auto factory = ilut_factory_type::build()
                       .with_skip_sorting(true)
                       .with_iterations(7u)
                       .with_approximate_select(false)
                       .with_deterministic_sample(true)
                       .with_fill_in_limit(1.2)
                       .on(ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().approximate_select, false);
    ASSERT_EQ(factory->get_parameters().deterministic_sample, true);
    ASSERT_EQ(factory->get_parameters().fill_in_limit, 1.2);
}


}  // namespace
