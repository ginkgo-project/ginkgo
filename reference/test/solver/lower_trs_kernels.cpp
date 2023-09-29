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

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/lower_trs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class LowerTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::LowerTrs<value_type, index_type>;
    LowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          ref(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1, 0.0, 0.0}, {3.0, 1, 0.0}, {1.0, 2.0, 1}}, exec)),
          mtx2(gko::initialize<Mtx>(
              {{2, 0.0, 0.0}, {3.0, 3, 0.0}, {1.0, 2.0, 4}}, exec)),
          mtx_big_lower(gko::initialize<Mtx>({{124.0, 0.0, 0.0, 0.0, 0.0},
                                              {43.0, -789.0, 0.0, 0.0, 0.0},
                                              {134.5, -651.0, 654.0, 0.0, 0.0},
                                              {-642.0, 684.0, 68.0, 387.0, 0.0},
                                              {365.0, 97.0, -654.0, 8.0, 91.0}},
                                             exec)),
          mtx_big_general(
              gko::initialize<Mtx>({{124.0, 4.0, -4.0, 0.0, 2.0},
                                    {43.0, -789.0, 0.0, 2.0, 1.0},
                                    {134.5, -651.0, 654.0, 0.0, 0.5},
                                    {-642.0, 684.0, 68.0, 387.0, 0.0},
                                    {365.0, 97.0, -654.0, 8.0, 91.0}},
                                   exec)),
          lower_trs_factory(Solver::build().on(exec)),
          lower_trs_factory_mrhs(Solver::build().with_num_rhs(2u).on(exec)),
          lower_trs_factory_unit(
              Solver::build().with_unit_diagonal(true).on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx2;
    std::shared_ptr<Mtx> mtx_big_lower;
    std::shared_ptr<Mtx> mtx_big_general;
    std::unique_ptr<typename Solver::Factory> lower_trs_factory;
    std::unique_ptr<typename Solver::Factory> lower_trs_factory_mrhs;
    std::unique_ptr<typename Solver::Factory> lower_trs_factory_unit;
};

TYPED_TEST_SUITE(LowerTrs, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(LowerTrs, RefLowerTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::reference::lower_trs::should_perform_transpose(this->ref,
                                                                 trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemMixed)
{
    using other_value_type = typename TestFixture::value_type;
    using value_type = next_precision<other_value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 2.0}),
                        (r_mixed<value_type, other_value_type>()));
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{-1.0, 2.0},
                           value_type{2.0, -4.0}}),
                        r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemMixedComplex)
{
    using other_value_type = typename TestFixture::value_type;
    using Scalar = gko::matrix::Dense<next_precision<other_value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{0.0, 0.0}, value_type{0.0, 0.0}, value_type{0.0, 0.0}},
        this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{-1.0, 2.0},
                           value_type{2.0, -4.0}}),
                        (r_mixed<value_type, other_value_type>()));
}


TYPED_TEST(LowerTrs, SolvesMultipleTriangularSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {I<T>{3.0, 4.0}, I<T>{1.0, 0.0}, I<T>{1.0, -1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{0.0, 0.0}, I<T>{0.0, 0.0}, I<T>{0.0, 0.0}}, this->exec);
    auto solver = this->lower_trs_factory_mrhs->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{3.0, 4.0}, {-8.0, -12.0}, {14.0, 19.0}}),
                        r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesNonUnitTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({2.0, 12.0, 3.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx2);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, -1.0}), r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 3.0}), r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemUsingAdvancedApplyMixed)
{
    using other_value_type = typename TestFixture::value_type;
    using value_type = next_precision<other_value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 3.0}),
                        (r_mixed<value_type, other_value_type>()));
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemUsingAdvancedApplyComplex)
{
    using Scalar = typename TestFixture::Mtx;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{-1.0, 2.0}, value_type{1.0, -2.0}},
        this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{-1.0, 2.0},
                           value_type{3.0, -6.0}}),
                        r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesTriangularSystemUsingAdvancedApplyMixedComplex)
{
    using other_value_type = typename TestFixture::value_type;
    using Scalar = gko::matrix::Dense<next_precision<other_value_type>>;
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using value_type = typename Mtx::value_type;
    auto alpha = gko::initialize<Scalar>({2.0}, this->exec);
    auto beta = gko::initialize<Scalar>({-1.0}, this->exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}, value_type{1.0, -2.0}},
        this->exec);
    auto x = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{-1.0, 2.0}, value_type{1.0, -2.0}},
        this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({value_type{1.0, -2.0}, value_type{-1.0, 2.0},
                           value_type{3.0, -6.0}}),
                        (r_mixed<value_type, other_value_type>()));
}


TYPED_TEST(LowerTrs, SolvesMultipleTriangularSystemsUsingAdvancedApply)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {I<T>{3.0, 4.0}, I<T>{1.0, 0.0}, I<T>{1.0, -1.0}}, this->exec);
    auto x = gko::initialize<Mtx>(
        {I<T>{1.0, 2.0}, I<T>{-1.0, -1.0}, I<T>{0.0, -2.0}}, this->exec);
    auto solver = this->lower_trs_factory_mrhs->generate(this->mtx);

    solver->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-1.0, 0.0}, {6.0, 10.0}, {-14.0, -23.0}}),
                        r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesBigDenseSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {-124.0, -3199.0, 3147.5, 5151.0, -6021.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx_big_lower);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-1.0, 4.0, 9.0, 3.0, -2.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(LowerTrs, SolvesBigDenseSystemWithUnitDiagonal)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {-1.0, -39.0, -2729.5, 3993.0, -5841.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory_unit->generate(this->mtx_big_lower);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-1.0, 4.0, 9.0, 3.0, -2.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(LowerTrs, SolveBigDenseSystemIgnoresNonTriangleEntries)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
        {-124.0, -3199.0, 3147.5, 5151.0, -6021.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx_big_general);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-1.0, 4.0, 9.0, 3.0, -2.0}),
                        r<value_type>::value * 1e3);
}


TYPED_TEST(LowerTrs, SolvesTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({0.0, 0.0, 1.0}), r<value_type>::value);
}


TYPED_TEST(LowerTrs, SolvesConjTransposedTriangularSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    auto solver = this->lower_trs_factory->generate(this->mtx);

    solver->conj_transpose()->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({0.0, 0.0, 1.0}), r<value_type>::value);
}


}  // namespace
