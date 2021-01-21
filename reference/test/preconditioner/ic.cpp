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

TYPED_TEST_SUITE(Ic, gko::test::ValueIndexTypes);


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
        Composition::create(this->lh_factor, this->l_factor);
    auto copied = ic_prec_type::build()
                      .on(this->exec)
                      ->generate(gko::share(lh_l_composition));

    copied->copy_from(ic.get());

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
        Composition::create(this->lh_factor, this->l_factor);
    auto moved = ic_prec_type::build()
                     .on(this->exec)
                     ->generate(gko::share(lh_l_composition));

    moved->copy_from(std::move(ic));

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

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({3.0, -2.0, 4.0}), this->tol);
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

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{3.0, 6.0, 9.0}, {-2.0, -4.0, -6.0}, {4.0, 8.0, 12.0}}),
        this->tol);
}


}  // namespace
