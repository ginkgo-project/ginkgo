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

#include <ginkgo/core/factorization/bilu.hpp>
#include <ginkgo/core/factorization/par_bilu.hpp>


#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>


#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace {


template <typename ValueIndexType>
class ParBilu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using par_bilu_t = gko::factorization::ParBilu<value_type, index_type>;
    using bilu_t = gko::factorization::Bilu<value_type, index_type>;

    ParBilu() : refexec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> refexec;
};

TYPED_TEST_SUITE(ParBilu, gko::test::ValueIndexTypes);


TYPED_TEST(ParBilu, FactorizationSortedBS4)
{
    using Fbcsr = typename TestFixture::Fbcsr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using bilu_t = typename TestFixture::bilu_t;
    using par_bilu_t = typename TestFixture::par_bilu_t;
    const int nbrows = 50, bs = 4;
    std::shared_ptr<const Fbcsr> mtx =
        gko::test::generate_random_fbcsr<value_type, index_type>(
            this->refexec, std::ranlux48(42), nbrows, nbrows, bs, true, false);
    auto parbilu_factory_skip =
        par_bilu_t::build().with_iterations(1).with_skip_sorting(true).on(
            this->refexec);
    auto bilu_factory_skip =
        bilu_t::build().with_skip_sorting(true).on(this->refexec);

    auto bfacts = bilu_factory_skip->generate(mtx);
    auto parbfacts = parbilu_factory_skip->generate(mtx);

    auto bfL = gko::as<const Fbcsr>(bfacts->get_l_factor());
    auto pbfL = gko::as<const Fbcsr>(parbfacts->get_l_factor());
    auto bfU = gko::as<const Fbcsr>(bfacts->get_u_factor());
    auto pbfU = gko::as<const Fbcsr>(parbfacts->get_u_factor());
    constexpr auto eps =
        4 * std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_EQ_SPARSITY(bfL, pbfL);
    GKO_ASSERT_MTX_EQ_SPARSITY(bfU, pbfU);
    GKO_ASSERT_MTX_NEAR(bfL, pbfL, eps);
    GKO_ASSERT_MTX_NEAR(bfU, pbfU, eps);
}

TYPED_TEST(ParBilu, FactorizationUnsortedBS3)
{
    using Fbcsr = typename TestFixture::Fbcsr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using bilu_t = typename TestFixture::bilu_t;
    using par_bilu_t = typename TestFixture::par_bilu_t;
    const int nbrows = 100, bs = 3;
    std::shared_ptr<const Fbcsr> mtx =
        gko::test::generate_random_fbcsr<value_type, index_type>(
            this->refexec, std::ranlux48(42), nbrows, nbrows, bs, true, true);
    EXPECT_FALSE(mtx->is_sorted_by_column_index());
    auto parbilu_factory_skip =
        par_bilu_t::build().with_iterations(1).with_skip_sorting(false).on(
            this->refexec);
    auto bilu_factory_skip =
        bilu_t::build().with_skip_sorting(false).on(this->refexec);

    auto bfacts = bilu_factory_skip->generate(mtx);
    auto parbfacts = parbilu_factory_skip->generate(mtx);

    auto bfL = bfacts->get_l_factor();
    auto pbfL = parbfacts->get_l_factor();
    auto bfU = bfacts->get_u_factor();
    auto pbfU = parbfacts->get_u_factor();
    constexpr auto eps =
        4 * std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_EQ_SPARSITY(bfL, pbfL);
    GKO_ASSERT_MTX_EQ_SPARSITY(bfU, pbfU);
    GKO_ASSERT_MTX_NEAR(bfL, pbfL, eps);
    GKO_ASSERT_MTX_NEAR(bfU, pbfU, eps);
}

}  // namespace
