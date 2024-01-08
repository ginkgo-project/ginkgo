// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fbcsr_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"
#include "test/utils/executor.hpp"


template <typename T>
class Fbcsr : public CommonTestFixture {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    Fbcsr() : distb(), engine(42)
    {
        const index_type rand_brows = 100;
        const index_type rand_bcols = 70;
        const int block_size = 3;
        rsorted = gko::test::generate_random_fbcsr<value_type, index_type>(
            ref, rand_brows, rand_bcols, block_size, false, false,
            std::default_random_engine(43));
    }

    std::unique_ptr<const Mtx> rsorted;

    std::normal_distribution<gko::remove_complex<T>> distb;
    std::default_random_engine engine;

    value_type get_random_value()
    {
        return gko::test::detail::get_rand_value<T>(distb, engine);
    }

    void generate_sin(gko::ptr_param<Dense> x)
    {
        value_type* const xarr = x->get_values();
        for (index_type i = 0; i < x->get_size()[0] * x->get_size()[1]; i++) {
            xarr[i] =
                static_cast<real_type>(2.0) *
                std::sin(static_cast<real_type>(i / 2.0) + get_random_value());
        }
    }
};

#ifdef GKO_COMPILING_HIP
TYPED_TEST_SUITE(Fbcsr, gko::test::RealValueTypes, TypenameNameGenerator);
#else
TYPED_TEST_SUITE(Fbcsr, gko::test::ValueTypes, TypenameNameGenerator);
#endif

TYPED_TEST(Fbcsr, CanWriteFromMatrixOnDevice)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;
    using MatData = gko::matrix_data<value_type, index_type>;
    gko::testing::FbcsrSample<value_type, index_type> sample(this->ref);
    auto mat = sample.generate_fbcsr();
    auto dmat = gko::clone(this->exec, mat);
    MatData refdata;
    MatData devdata;

    mat->write(refdata);
    dmat->write(devdata);

    ASSERT_TRUE(refdata.nonzeros == devdata.nonzeros);
}


TYPED_TEST(Fbcsr, TransposeIsEquivalentToRefSortedBS3)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;
    auto drand = gko::clone(this->exec, this->rsorted);

    auto trans = gko::as<Mtx>(this->rsorted->transpose());
    auto dtrans = gko::as<Mtx>(drand->transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(trans, dtrans);
    GKO_ASSERT_MTX_NEAR(trans, dtrans, 0.0);
}


TYPED_TEST(Fbcsr, TransposeIsEquivalentToRefSortedBS7)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;
    auto drand = Mtx::create(this->exec);
    const index_type rand_brows = 50;
    const index_type rand_bcols = 40;
    const int block_size = 7;
    auto rsorted2 = gko::test::generate_random_fbcsr<value_type, index_type>(
        this->ref, rand_brows, rand_bcols, block_size, false, false,
        std::default_random_engine(43));
    drand->copy_from(rsorted2);

    auto trans = gko::as<Mtx>(rsorted2->transpose());
    auto dtrans = gko::as<Mtx>(drand->transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(trans, dtrans);
    GKO_ASSERT_MTX_NEAR(trans, dtrans, 0.0);
}


TYPED_TEST(Fbcsr, SpmvIsEquivalentToRefSorted)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename Mtx::value_type;
    auto drand = gko::clone(this->exec, this->rsorted);
    auto x =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[1], 1));
    this->generate_sin(x);
    auto dx = gko::clone(this->exec, x);
    auto prod =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[0], 1));
    auto dprod = Dense::create(this->exec, prod->get_size());

    drand->apply(dx, dprod);
    this->rsorted->apply(x, prod);

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(prod, dprod, 5 * tol);
}


TYPED_TEST(Fbcsr, SpmvMultiIsEquivalentToRefSorted)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename Mtx::value_type;
    auto drand = gko::clone(this->exec, this->rsorted);
    auto x =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[1], 3));
    this->generate_sin(x);
    auto dx = gko::clone(this->exec, x);
    auto prod =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[0], 3));
    auto dprod = Dense::create(this->exec, prod->get_size());

    drand->apply(dx, dprod);
    this->rsorted->apply(x, prod);

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(prod, dprod, 5 * tol);
}


TYPED_TEST(Fbcsr, AdvancedSpmvIsEquivalentToRefSorted)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    auto drand = gko::clone(this->exec, this->rsorted);
    auto x =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[1], 1));
    this->generate_sin(x);
    auto dx = gko::clone(this->exec, x);
    auto prod =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[0], 1));
    this->generate_sin(prod);
    auto dprod = gko::clone(this->exec, prod);
    auto alpha = Dense::create(this->ref, gko::dim<2>(1, 1));
    alpha->at(0, 0) = static_cast<real_type>(2.4) + this->get_random_value();
    auto beta = Dense::create(this->ref, gko::dim<2>(1, 1));
    beta->at(0, 0) = -1.2;
    auto dalpha = gko::clone(this->exec, alpha);
    auto dbeta = gko::clone(this->exec, beta);

    drand->apply(dalpha, dx, dbeta, dprod);
    this->rsorted->apply(alpha, x, beta, prod);

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(prod, dprod, 5 * tol);
}


TYPED_TEST(Fbcsr, AdvancedSpmvMultiIsEquivalentToRefSorted)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    auto drand = gko::clone(this->exec, this->rsorted);
    auto x =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[1], 3));
    this->generate_sin(x);
    auto dx = gko::clone(this->exec, x);
    auto prod =
        Dense::create(this->ref, gko::dim<2>(this->rsorted->get_size()[0], 3));
    this->generate_sin(prod);
    auto dprod = gko::clone(this->exec, prod);
    auto alpha = Dense::create(this->ref, gko::dim<2>(1, 1));
    alpha->at(0, 0) = static_cast<real_type>(2.4) + this->get_random_value();
    auto beta = Dense::create(this->ref, gko::dim<2>(1, 1));
    beta->at(0, 0) = -1.2;
    auto dalpha = gko::clone(this->exec, alpha);
    auto dbeta = gko::clone(this->exec, beta);

    drand->apply(dalpha, dx, dbeta, dprod);
    this->rsorted->apply(alpha, x, beta, prod);

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(prod, dprod, 5 * tol);
}


TYPED_TEST(Fbcsr, ConjTransposeIsEquivalentToRefSortedBS3)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;
    auto drand = gko::clone(this->exec, this->rsorted);

    auto trans = gko::as<Mtx>(this->rsorted->conj_transpose());
    auto dtrans = gko::as<Mtx>(drand->conj_transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(trans, dtrans);
    GKO_ASSERT_MTX_NEAR(trans, dtrans, 0.0);
}


TYPED_TEST(Fbcsr, RecognizeSortedMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    auto drand = gko::clone(this->exec, this->rsorted);

    ASSERT_TRUE(drand->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, RecognizeUnsortedMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto mat = this->rsorted->clone();
    index_type* const colinds = mat->get_col_idxs();
    std::swap(colinds[0], colinds[1]);
    auto dunsrt = gko::clone(this->exec, mat);

    ASSERT_FALSE(dunsrt->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    auto rand = gko::clone(this->ref, this->rsorted);
    auto drand = gko::clone(this->exec, this->rsorted);

    rand->compute_absolute_inplace();
    drand->compute_absolute_inplace();

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(rand, drand, tol);
}


TYPED_TEST(Fbcsr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    auto drand = gko::clone(this->exec, this->rsorted);

    auto abs_mtx = this->rsorted->compute_absolute();
    auto dabs_mtx = drand->compute_absolute();

    const double tol = r<value_type>::value;
    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, tol);
}
