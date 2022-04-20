/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/ic_wrapper.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class IcWrapper : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ltr = gko::solver::LowerTrs<value_type, index_type>;
    using isai = gko::preconditioner::LowerIsai<value_type, index_type>;
    using ic_type = gko::preconditioner::IcWrapper;
    using Composition = gko::Composition<value_type>;
    using par_ic = gko::factorization::ParIc<value_type, index_type>;

    IcWrapper()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{1., 1., 0.}, {1., 2., 1.}, {0., 1., 2.}},
                                   exec)),
          l_factor(gko::initialize<Mtx>(
              {{1., 0., 0.}, {1., 1., 0.}, {0., 1., 1.}}, exec)),
          lh_factor(gko::as<Mtx>(l_factor->conj_transpose())),
          l_isai_factor(gko::initialize<Mtx>(
              {{1., 0., 0.}, {-1., 1., 0.}, {0., -1., 1.}}, exec)),
          lh_isai_factor(gko::as<Mtx>(l_isai_factor->conj_transpose())),
          l_composition(Composition::create(l_factor)),
          l_lh_composition(Composition::create(l_factor, lh_factor)),
          ic_pre_factory(
              ic_type::build()
                  .with_l_solver_factory(
                      ic_type::generate_default_solver<ltr>(exec))
                  .with_factorization_factory(
                      par_ic::build().with_both_factors(false).on(exec))
                  .on(exec)),
          ic_isai_pre_factory(
              ic_type::build()
                  .with_l_solver_factory(
                      ic_type::generate_default_solver<isai>(exec))
                  .with_factorization_factory(
                      par_ic::build().with_both_factors(false).on(exec))
                  .on(exec)),
          tol{r<value_type>::value}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> l_factor;
    std::shared_ptr<Mtx> lh_factor;
    std::shared_ptr<Mtx> l_isai_factor;
    std::shared_ptr<Mtx> lh_isai_factor;
    std::shared_ptr<Composition> l_composition;
    std::shared_ptr<Composition> l_lh_composition;
    std::shared_ptr<typename ic_type::Factory> ic_pre_factory;
    std::shared_ptr<typename ic_type::Factory> ic_isai_pre_factory;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(IcWrapper, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(IcWrapper, BuildsTwoFactorComposition)
{
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto precond = this->ic_pre_factory->generate(this->l_lh_composition);

    GKO_ASSERT_MTX_NEAR(
        gko::as<ltr>(precond->get_l_solver())->get_system_matrix(),
        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(
        gko::as<utr>(precond->get_lh_solver())->get_system_matrix(),
        this->lh_factor, this->tol);
}


TYPED_TEST(IcWrapper, BuildsOneFactorComposition)
{
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto precond = this->ic_pre_factory->generate(this->l_composition);

    GKO_ASSERT_MTX_NEAR(
        gko::as<ltr>(precond->get_l_solver())->get_system_matrix(),
        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(
        gko::as<utr>(precond->get_lh_solver())->get_system_matrix(),
        this->lh_factor, this->tol);
}


TYPED_TEST(IcWrapper, BuildsSymmetricTwoFactor)
{
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto precond = this->ic_pre_factory->generate(this->l_lh_composition);

    // the first factor should be identical, the second not as it was transposed
    ASSERT_EQ(gko::as<ltr>(precond->get_l_solver())->get_system_matrix().get(),
              this->l_factor.get());
    ASSERT_NE(gko::as<utr>(precond->get_lh_solver())->get_system_matrix().get(),
              this->lh_factor.get());
}


TYPED_TEST(IcWrapper, BuildsFromMatrix)
{
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto precond = this->ic_pre_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(
        gko::as<ltr>(precond->get_l_solver())->get_system_matrix(),
        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(
        gko::as<utr>(precond->get_lh_solver())->get_system_matrix(),
        this->lh_factor, this->tol);
}


TYPED_TEST(IcWrapper, BuildsIsaiFromMatrix)
{
    using isai = typename TestFixture::isai;
    using isai_trans = typename isai::transposed_type;
    auto precond = this->ic_isai_pre_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(
        gko::as<isai>(precond->get_l_solver())->get_approximate_inverse(),
        this->l_isai_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(gko::as<isai_trans>(precond->get_lh_solver())
                            ->get_approximate_inverse(),
                        this->lh_isai_factor, this->tol);
}


TYPED_TEST(IcWrapper, ThrowOnWrongCompositionInput)
{
    using Composition = typename TestFixture::Composition;
    std::shared_ptr<Composition> composition =
        Composition::create(this->l_factor, this->l_factor, this->l_factor);

    ASSERT_THROW(this->ic_pre_factory->generate(composition),
                 gko::NotSupported);
}


TYPED_TEST(IcWrapper, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Composition = typename TestFixture::Composition;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto lh_l_composition =
        gko::share(Composition::create(this->lh_factor, this->l_factor));
    auto copied = this->ic_pre_factory->generate(lh_l_composition);

    copied->copy_from(ic.get());

    ASSERT_EQ(before_l_solver, copied->get_l_solver());
    ASSERT_EQ(before_lh_solver, copied->get_lh_solver());
}


TYPED_TEST(IcWrapper, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Composition = typename TestFixture::Composition;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto lh_l_composition =
        gko::share(Composition::create(this->lh_factor, this->l_factor));
    auto moved = this->ic_pre_factory->generate(lh_l_composition);

    moved->copy_from(std::move(ic));

    ASSERT_EQ(before_l_solver, moved->get_l_solver());
    ASSERT_EQ(before_lh_solver, moved->get_lh_solver());
}


TYPED_TEST(IcWrapper, CanBeCloned)
{
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();

    auto clone = ic->clone();

    ASSERT_EQ(before_l_solver, clone->get_l_solver());
    ASSERT_EQ(before_lh_solver, clone->get_lh_solver());
}


TYPED_TEST(IcWrapper, CanBeTransposed)
{
    using IcWrapper = typename TestFixture::ic_type;
    using Mtx = typename TestFixture::Mtx;
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto l_ref =
        gko::as<Mtx>(gko::as<ltr>(ic->get_l_solver())->get_system_matrix());
    auto lh_ref =
        gko::as<Mtx>(gko::as<utr>(ic->get_lh_solver())->get_system_matrix());

    auto transp = gko::as<IcWrapper>(ic->transpose());

    auto l_transp =
        gko::as<Mtx>(gko::as<ltr>(transp->get_l_solver())->get_system_matrix());
    auto lh_transp = gko::as<Mtx>(
        gko::as<utr>(transp->get_lh_solver())->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(l_transp, l_ref, 0);
    GKO_ASSERT_MTX_NEAR(lh_transp, lh_ref, 0);
}


TYPED_TEST(IcWrapper, CanBeConjTransposed)
{
    using IcWrapper = typename TestFixture::ic_type;
    using Mtx = typename TestFixture::Mtx;
    using ltr = typename TestFixture::ltr;
    using utr = typename ltr::transposed_type;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto l_ref =
        gko::as<Mtx>(gko::as<ltr>(ic->get_l_solver())->get_system_matrix());
    auto lh_ref =
        gko::as<Mtx>(gko::as<utr>(ic->get_lh_solver())->get_system_matrix());

    auto transp = gko::as<IcWrapper>(ic->conj_transpose());

    auto l_transp =
        gko::as<Mtx>(gko::as<ltr>(transp->get_l_solver())->get_system_matrix());
    auto lh_transp = gko::as<Mtx>(
        gko::as<utr>(transp->get_lh_solver())->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(l_transp, l_ref, 0);
    GKO_ASSERT_MTX_NEAR(lh_transp, lh_ref, 0);
}


TYPED_TEST(IcWrapper, SolvesSingleRhs)
{
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({3.0, -2.0, 4.0}), this->tol);
}


TYPED_TEST(IcWrapper, SolvesSingleRhsMixed)
{
    using T = typename TestFixture::value_type;
    using Vec = gko::matrix::Dense<gko::next_precision<T>>;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({3.0, -2.0, 4.0}), this->tol);
}


TYPED_TEST(IcWrapper, SolvesSingleRhsComplex)
{
    using Vec = gko::to_complex<typename TestFixture::Vec>;
    using T = typename Vec::value_type;
    const auto b = gko::initialize<Vec>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({T{3.0, 6.0}, T{-2.0, -4.0}, T{4.0, 8.0}}),
                        this->tol);
}


TYPED_TEST(IcWrapper, SolvesSingleRhsComplexMixed)
{
    using Vec = gko::matrix::Dense<
        gko::next_precision<gko::to_complex<typename TestFixture::value_type>>>;
    using T = typename Vec::value_type;
    const auto b = gko::initialize<Vec>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({T{3.0, 6.0}, T{-2.0, -4.0}, T{4.0, 8.0}}),
                        this->tol);
}


TYPED_TEST(IcWrapper, AdvancedSolvesSingleRhs)
{
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({5.0, -6.0, 5.0}), this->tol);
}


TYPED_TEST(IcWrapper, AdvancedSolvesSingleRhsMixed)
{
    using T = typename TestFixture::value_type;
    using Vec = gko::matrix::Dense<gko::next_precision<T>>;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({5.0, -6.0, 5.0}), this->tol);
}


TYPED_TEST(IcWrapper, AdvancedSolvesSingleRhsComplex)
{
    using Dense = typename TestFixture::Vec;
    using DenseComplex = gko::to_complex<Dense>;
    using T = typename DenseComplex::value_type;
    const auto b = gko::initialize<DenseComplex>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    const auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    const auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = gko::initialize<DenseComplex>(
        {T{1.0, 2.0}, T{2.0, 4.0}, T{3.0, 6.0}}, this->exec);
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({T{5.0, 10.0}, T{-6.0, -12.0}, T{5.0, 10.0}}),
                        this->tol);
}


TYPED_TEST(IcWrapper, AdvancedSolvesSingleRhsComplexMixed)
{
    using MixedDense = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using MixedDenseComplex = gko::to_complex<MixedDense>;
    using T = typename MixedDenseComplex::value_type;
    const auto b = gko::initialize<MixedDenseComplex>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    const auto alpha = gko::initialize<MixedDense>({2.0}, this->exec);
    const auto beta = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {T{1.0, 2.0}, T{2.0, 4.0}, T{3.0, 6.0}}, this->exec);
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({T{5.0, 10.0}, T{-6.0, -12.0}, T{5.0, 10.0}}),
                        this->tol);
}


TYPED_TEST(IcWrapper, SolvesMultipleRhs)
{
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>(
        {{1.0, 2.0, 3.0}, {3.0, 6.0, 9.0}, {6.0, 12.0, 18.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 3});
    auto preconditioner = this->ic_pre_factory->generate(this->mtx);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{3.0, 6.0, 9.0}, {-2.0, -4.0, -6.0}, {4.0, 8.0, 12.0}}),
        this->tol);
}


}  // namespace
