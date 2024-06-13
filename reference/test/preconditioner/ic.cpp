// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ic.hpp>


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
class Ic : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ic_prec_type =
        gko::preconditioner::Ic<gko::solver::LowerTrs<value_type, index_type>,
                                index_type>;
    using ic_isai_prec_type = gko::preconditioner::Ic<
        gko::preconditioner::LowerIsai<value_type, index_type>, index_type>;
    using Composition = gko::Composition<value_type>;

    Ic()
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
          ic_pre_factory(ic_prec_type::build().on(exec)),
          ic_isai_pre_factory(ic_isai_prec_type::build().on(exec)),
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
    std::shared_ptr<typename ic_prec_type::Factory> ic_pre_factory;
    std::shared_ptr<typename ic_isai_prec_type::Factory> ic_isai_pre_factory;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(Ic, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ic, BuildsTwoFactorComposition)
{
    auto precond = this->ic_pre_factory->generate(this->l_lh_composition);

    GKO_ASSERT_MTX_NEAR(precond->get_l_solver()->get_system_matrix(),
                        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(precond->get_lh_solver()->get_system_matrix(),
                        this->lh_factor, this->tol);
}


TYPED_TEST(Ic, BuildsOneFactorComposition)
{
    auto precond = this->ic_pre_factory->generate(this->l_composition);

    GKO_ASSERT_MTX_NEAR(precond->get_l_solver()->get_system_matrix(),
                        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(precond->get_lh_solver()->get_system_matrix(),
                        this->lh_factor, this->tol);
}


TYPED_TEST(Ic, BuildsSymmetricTwoFactor)
{
    auto precond = this->ic_pre_factory->generate(this->l_lh_composition);

    // the first factor should be identical, the second not as it was transposed
    ASSERT_EQ(precond->get_l_solver()->get_system_matrix().get(),
              this->l_factor.get());
    ASSERT_NE(precond->get_lh_solver()->get_system_matrix().get(),
              this->lh_factor.get());
}


TYPED_TEST(Ic, BuildsFromMatrix)
{
    auto precond = this->ic_pre_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(precond->get_l_solver()->get_system_matrix(),
                        this->l_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(precond->get_lh_solver()->get_system_matrix(),
                        this->lh_factor, this->tol);
}


TYPED_TEST(Ic, BuildsIsaiFromMatrix)
{
    auto precond = this->ic_isai_pre_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(precond->get_l_solver()->get_approximate_inverse(),
                        this->l_isai_factor, this->tol);
    GKO_ASSERT_MTX_NEAR(precond->get_lh_solver()->get_approximate_inverse(),
                        this->lh_isai_factor, this->tol);
}


TYPED_TEST(Ic, ThrowOnWrongCompositionInput)
{
    using Composition = typename TestFixture::Composition;
    std::shared_ptr<Composition> composition =
        Composition::create(this->l_factor, this->l_factor, this->l_factor);

    ASSERT_THROW(this->ic_pre_factory->generate(composition),
                 gko::NotSupported);
}


TYPED_TEST(Ic, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Composition = typename TestFixture::Composition;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto lh_l_composition =
        gko::share(Composition::create(this->lh_factor, this->l_factor));
    auto copied =
        ic_prec_type::build().on(this->exec)->generate(lh_l_composition);

    copied->copy_from(ic);

    ASSERT_EQ(before_l_solver, copied->get_l_solver());
    ASSERT_EQ(before_lh_solver, copied->get_lh_solver());
}


TYPED_TEST(Ic, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Composition = typename TestFixture::Composition;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto lh_l_composition =
        gko::share(Composition::create(this->lh_factor, this->l_factor));
    auto moved =
        ic_prec_type::build().on(this->exec)->generate(lh_l_composition);

    moved->move_from(ic);

    ASSERT_EQ(before_l_solver, moved->get_l_solver());
    ASSERT_EQ(before_lh_solver, moved->get_lh_solver());
}


TYPED_TEST(Ic, CanBeCloned)
{
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto before_l_solver = ic->get_l_solver();
    auto before_lh_solver = ic->get_lh_solver();

    auto clone = ic->clone();

    ASSERT_EQ(before_l_solver, clone->get_l_solver());
    ASSERT_EQ(before_lh_solver, clone->get_lh_solver());
}


TYPED_TEST(Ic, CanBeTransposed)
{
    using Ic = typename TestFixture::ic_prec_type;
    using Mtx = typename TestFixture::Mtx;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto l_ref = gko::as<Mtx>(ic->get_l_solver()->get_system_matrix());
    auto lh_ref = gko::as<Mtx>(ic->get_lh_solver()->get_system_matrix());

    auto transp = gko::as<Ic>(ic->transpose());

    auto l_transp = gko::as<Mtx>(transp->get_l_solver()->get_system_matrix());
    auto lh_transp = gko::as<Mtx>(transp->get_lh_solver()->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(l_transp, l_ref, 0);
    GKO_ASSERT_MTX_NEAR(lh_transp, lh_ref, 0);
}


TYPED_TEST(Ic, CanBeConjTransposed)
{
    using Ic = typename TestFixture::ic_prec_type;
    using Mtx = typename TestFixture::Mtx;
    auto ic = this->ic_pre_factory->generate(this->l_lh_composition);
    auto l_ref = gko::as<Mtx>(ic->get_l_solver()->get_system_matrix());
    auto lh_ref = gko::as<Mtx>(ic->get_lh_solver()->get_system_matrix());

    auto transp = gko::as<Ic>(ic->conj_transpose());

    auto l_transp = gko::as<Mtx>(transp->get_l_solver()->get_system_matrix());
    auto lh_transp = gko::as<Mtx>(transp->get_lh_solver()->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(l_transp, l_ref, 0);
    GKO_ASSERT_MTX_NEAR(lh_transp, lh_ref, 0);
}


TYPED_TEST(Ic, SolvesSingleRhs)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({3.0, -2.0, 4.0}), this->tol);
}


TYPED_TEST(Ic, SolvesSingleRhsMixed)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using T = typename TestFixture::value_type;
    using Vec = gko::matrix::Dense<gko::next_precision<T>>;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({3.0, -2.0, 4.0}), this->tol);
}


TYPED_TEST(Ic, SolvesSingleRhsComplex)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Vec = gko::to_complex<typename TestFixture::Vec>;
    using T = typename Vec::value_type;
    const auto b = gko::initialize<Vec>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({T{3.0, 6.0}, T{-2.0, -4.0}, T{4.0, 8.0}}),
                        this->tol);
}


TYPED_TEST(Ic, SolvesSingleRhsComplexMixed)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Vec = gko::matrix::Dense<
        gko::next_precision<gko::to_complex<typename TestFixture::value_type>>>;
    using T = typename Vec::value_type;
    const auto b = gko::initialize<Vec>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 1});
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({T{3.0, 6.0}, T{-2.0, -4.0}, T{4.0, 8.0}}),
                        this->tol);
}


TYPED_TEST(Ic, AdvancedSolvesSingleRhs)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({5.0, -6.0, 5.0}), this->tol);
}


TYPED_TEST(Ic, AdvancedSolvesSingleRhsMixed)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using T = typename TestFixture::value_type;
    using Vec = gko::matrix::Dense<gko::next_precision<T>>;
    const auto b = gko::initialize<Vec>({1.0, 3.0, 6.0}, this->exec);
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({5.0, -6.0, 5.0}), this->tol);
}


TYPED_TEST(Ic, AdvancedSolvesSingleRhsComplex)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Dense = typename TestFixture::Vec;
    using DenseComplex = gko::to_complex<Dense>;
    using T = typename DenseComplex::value_type;
    const auto b = gko::initialize<DenseComplex>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    const auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    const auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = gko::initialize<DenseComplex>(
        {T{1.0, 2.0}, T{2.0, 4.0}, T{3.0, 6.0}}, this->exec);
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({T{5.0, 10.0}, T{-6.0, -12.0}, T{5.0, 10.0}}),
                        this->tol);
}


TYPED_TEST(Ic, AdvancedSolvesSingleRhsComplexMixed)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
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
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({T{5.0, 10.0}, T{-6.0, -12.0}, T{5.0, 10.0}}),
                        this->tol);
}


TYPED_TEST(Ic, SolvesMultipleRhs)
{
    using ic_prec_type = typename TestFixture::ic_prec_type;
    using Vec = typename TestFixture::Vec;
    const auto b = gko::initialize<Vec>(
        {{1.0, 2.0, 3.0}, {3.0, 6.0, 9.0}, {6.0, 12.0, 18.0}}, this->exec);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 3});
    auto preconditioner =
        ic_prec_type::build().on(this->exec)->generate(this->mtx);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({{3.0, 6.0, 9.0}, {-2.0, -4.0, -6.0}, {4.0, 8.0, 12.0}}),
        this->tol);
}


}  // namespace
