// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"


TEST(AMPTypes, NarrowTypesWorksCorrectly)
{
#if GINKGO_HAVE_AMP_HALF
    using mytypesd = gko::amp::narrow_types<double>::type;
    constexpr auto chkd =
        std::is_same<mytypesd,
                     std::tuple<double, float, gko::amp::half>>::value;
    static_assert(chkd, "Wrong types_list<double>!");
    using mytypescd = gko::amp::narrow_types<std::complex<double>>::type;
    constexpr auto chkcd =
        std::is_same<mytypescd,
                     std::tuple<std::complex<double>, std::complex<float>,
                                std::complex<gko::amp::half>>>::value;
    static_assert(chkcd, "Wrong types_list<cdouble>!");
    using mytypescf = gko::amp::narrow_types<std::complex<float>>::type;
    constexpr auto chkcf = std::is_same<
        mytypescf,
        std::tuple<std::complex<float>, std::complex<gko::amp::half>>>::value;
    static_assert(chkcf, "Wrong types_list<cfloat>!");
    static_assert(gko::amp::narrow_types<double>::num_types == 3);
    static_assert(gko::amp::narrow_types<std::complex<float>>::num_types == 2);
    static_assert(gko::amp::narrow_types<gko::amp::half>::num_types == 1);
    static_assert(
        gko::amp::narrow_types<std::complex<gko::amp::half>>::num_types == 1);
#else
    using mytypesd = gko::amp::narrow_types<double>::type;
    constexpr auto chkd =
        std::is_same<mytypesd, std::tuple<double, float>>::value;
    static_assert(chkd, "Wrong types_list<double>!");
    using mytypescd = gko::amp::narrow_types<std::complex<double>>::type;
    constexpr auto chkcd =
        std::is_same<mytypescd, std::tuple<std::complex<double>,
                                           std::complex<float>>>::value;
    static_assert(chkcd, "Wrong types_list<cdouble>!");
    using mytypescf = gko::amp::narrow_types<std::complex<float>>::type;
    constexpr auto chkcf =
        std::is_same<mytypescf, std::tuple<std::complex<float>>>::value;
    static_assert(chkcf, "Wrong types_list<cfloat>!");
    static_assert(gko::amp::narrow_types<double>::num_types == 2);
    static_assert(gko::amp::narrow_types<std::complex<double>>::num_types == 2);
    static_assert(gko::amp::narrow_types<float>::num_types == 1);
    static_assert(gko::amp::narrow_types<std::complex<float>>::num_types == 1);
#endif
}

template <typename T, typename I>
using Ell = gko::matrix::Ell<T, I>;


TEST(AMPHelpers, AllocatesEllBinsCorrectlyDouble)
{
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
#if GINKGO_HAVE_AMP_HALF
    auto mnpr = gko::amp::array_prec<int, double>{3, 4, 5};

    auto bins = gko::amp::allocate_bins<double, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 3,
                  "wrong number of bins!");
    auto p = dynamic_cast<Ell<double, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 3);
    EXPECT_TRUE(p->get_col_idxs()[0] = 1);
    auto q = dynamic_cast<Ell<float, int>*>(bins[1].get());
    EXPECT_TRUE(q);
    EXPECT_EQ(q->get_size(), ds);
    EXPECT_EQ(q->get_num_stored_elements_per_row(), 4);
    EXPECT_TRUE(q->get_col_idxs());
    auto r = dynamic_cast<Ell<gko::amp::half, int>*>(bins[2].get());
    EXPECT_TRUE(r);
    EXPECT_EQ(r->get_size(), ds);
    EXPECT_EQ(r->get_num_stored_elements_per_row(), 5);
    EXPECT_TRUE(r->get_col_idxs());
#else
    auto mnpr = gko::amp::array_prec<int, double>{3, 4};

    auto bins = gko::amp::allocate_bins<double, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    auto p = dynamic_cast<Ell<double, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 3);
    EXPECT_TRUE(p->get_col_idxs()[0] = 1);
    auto q = dynamic_cast<Ell<float, int>*>(bins[1].get());
    EXPECT_TRUE(q);
    EXPECT_EQ(q->get_size(), ds);
    EXPECT_EQ(q->get_num_stored_elements_per_row(), 4);
    EXPECT_TRUE(q->get_col_idxs());
#endif
}

#if GINKGO_HAVE_AMP_HALF
TEST(AMPHelpers, AllocatesEllBinsCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    using half = gko::amp::half;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::array_prec<int, value_type>{4, 5};

    auto bins = gko::amp::allocate_bins<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    auto p = dynamic_cast<gko::matrix::Ell<value_type, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 4);
    auto r =
        dynamic_cast<gko::matrix::Ell<std::complex<half>, int>*>(bins[1].get());
    EXPECT_TRUE(r);
    EXPECT_EQ(r->get_size(), ds);
    EXPECT_EQ(r->get_num_stored_elements_per_row(), 5);
}
#else
TEST(AMPHelpers, AllocatesEllBinsCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::array_prec<int, value_type>{4};

    auto bins = gko::amp::allocate_bins<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 1,
                  "wrong number of bins!");
    auto p = dynamic_cast<gko::matrix::Ell<value_type, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 4);
}
#endif

#if GINKGO_HAVE_AMP_HALF

TEST(AMPHelpers, AllocatesEllBinsTupleCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    using half = gko::amp::half;

    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::array_prec<int, value_type>{4, 5};

    auto bins = gko::amp::allocate_bins_tuple<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    using bin0type = decltype(std::get<0>(bins));
    static_assert(
        std::is_same<bin0type,
                     std::unique_ptr<Ell<std::complex<float>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<0>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<0>(bins)->get_num_stored_elements_per_row(), 4);
    static_assert(
        std::is_same<decltype(std::get<1>(bins)),
                     std::unique_ptr<Ell<std::complex<half>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<1>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<1>(bins)->get_num_stored_elements_per_row(), 5);
}

#else

TEST(AMPHelpers, AllocatesEllBinsTupleCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;

    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::array_prec<int, value_type>{4};

    auto bins = gko::amp::allocate_bins_tuple<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 1,
                  "wrong number of bins!");
    using bin0type = decltype(std::get<0>(bins));
    static_assert(
        std::is_same<bin0type,
                     std::unique_ptr<Ell<std::complex<float>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<0>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<0>(bins)->get_num_stored_elements_per_row(), 4);
}

#endif


template <typename ValueIndexType>
class Amp : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;

    Amp() : exec(gko::ReferenceExecutor::create())
    {
        // Static tests
#if GINKGO_ENABLE_BFLOAT16 || GINKGO_ENABLE_HALF
        static_assert(gko::matrix::AMP<double, int>::num_precisions == 3,
                      "Wrong number of supported precisions for AMP<double>!");
        static_assert(gko::matrix::AMP<float, int>::num_precisions == 2,
                      "Wrong number of supported precisions for AMP<float>!");
        static_assert(
            gko::matrix::AMP<std::complex<float>, int>::num_precisions == 2,
            "Wrong number of supported precisions for AMP<complex float>!");
#else
        static_assert(gko::matrix::AMP<double, int>::num_precisions == 2,
                      "Wrong number of supported precisions for AMP<double>!");
        static_assert(gko::matrix::AMP<float, int>::num_precisions == 1,
                      "Wrong number of supported precisions for AMP<float>!");
        static_assert(
            gko::matrix::AMP<std::complex<float>, int>::num_precisions == 1,
            "Wrong number of supported precisions for AMP<complex float>!");
#endif
    }

    std::unique_ptr<Mtx> create_amp_from_one_dense(gko::dim<2> size)
    {
        auto input = gko::share(Dense::create(exec, size));
        input->fill(gko::one<value_type>());
        auto inell = gko::share(Ell::create(exec));
        input->convert_to(inell.get());
        auto factory = Mtx::build().on(exec);
        return factory->generate(inell);
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        for (int i = 0; i < Mtx::num_precisions; ++i) {
            ASSERT_EQ(m->get_bin_matrix(i), nullptr);
        }
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(Amp, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(Amp, HasCorrectExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    auto empty_input = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto factory = Mtx::build().on(this->exec);
    auto mtx = factory->generate(empty_input);

    ASSERT_EQ(mtx->get_executor()->get_description(),
              this->exec->get_description());
}


TYPED_TEST(Amp, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    auto empty_input = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto factory = Mtx::build().on(this->exec);
    auto mtx = factory->generate(empty_input);

    ASSERT_EQ(mtx->get_size(), gko::dim<2>(0, 0));
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithDefaultParameters)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    ASSERT_NE(factory, nullptr);
    ASSERT_EQ(factory->get_executor(), this->exec);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithCustomTolerance)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().with_tolerance(1e-6f).on(this->exec);

    EXPECT_EQ(factory->get_parameters().tolerance, 1e-6f);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithNormwiseStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build()
                       .with_strategy(Mtx::tolerance_type::normwise)
                       .on(this->exec);

    EXPECT_EQ(factory->get_parameters().strategy,
              Mtx::tolerance_type::normwise);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithComponentwiseStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build()
                       .with_strategy(Mtx::tolerance_type::componentwise)
                       .on(this->exec);

    EXPECT_EQ(factory->get_parameters().strategy,
              Mtx::tolerance_type::componentwise);
}


TYPED_TEST(Amp, FactoryGenerateCompletesWithoutError)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using Ell = typename TestFixture::Ell;
    auto dinput = Dense::create(this->exec, gko::dim<2>{3, 3});
    dinput->fill(gko::one<typename TestFixture::value_type>());
    auto input = gko::share(Ell::create(this->exec));
    dinput->convert_to(input.get());
    auto factory = Mtx::build().on(this->exec);

    ASSERT_NO_THROW(auto mtx = factory->generate(input));
}


TYPED_TEST(Amp, GeneratedMatrixHasCorrectSize)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    auto input = gko::share(Ell::create(this->exec, gko::dim<2>{4, 5}));
    auto factory = Mtx::build().on(this->exec);

    auto mtx = factory->generate(input);

    EXPECT_EQ(mtx->get_size(), gko::dim<2>(4, 5));
    gko::constexpr_for<0, Mtx::num_precisions, 1>([&](auto k) {
        using types_list = typename gko::amp::narrow_types<value_type>::type;
        using vtype = typename std::tuple_element<k, types_list>::type;
        auto mell = dynamic_cast<const gko::matrix::Ell<vtype, index_type>*>(
            mtx->get_bin_matrix(k));
        EXPECT_EQ(mell->get_size(), input->get_size());
        EXPECT_GE(mell->get_num_stored_elements_per_row(), 0);
    });
    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(-1), nullptr);
}


TYPED_TEST(Amp, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{2, 3});

    auto copy = mtx->clone();

    auto copy_mtx = dynamic_cast<Mtx*>(copy.get());
    ASSERT_NE(copy_mtx, nullptr);
    EXPECT_EQ(copy_mtx->get_size(), mtx->get_size());
}


TYPED_TEST(Amp, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{2, 3});
    auto original_size = mtx->get_size();
    auto moved = mtx->clone();

    moved->move_from(mtx);

    auto moved_mtx = dynamic_cast<Mtx*>(moved.get());
    ASSERT_NE(moved_mtx, nullptr);
    EXPECT_EQ(moved_mtx->get_size(), original_size);
}


TYPED_TEST(Amp, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{3, 4});

    auto clone = mtx->clone();

    auto cloned_mtx = dynamic_cast<Mtx*>(clone.get());
    ASSERT_NE(cloned_mtx, nullptr);
    EXPECT_EQ(cloned_mtx->get_size(), mtx->get_size());
}


TYPED_TEST(Amp, CanBeCleared)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{2, 3});

    mtx->clear();

    this->assert_empty(mtx.get());
}


TYPED_TEST(Amp, GetBinMatrixReturnsNullForInvalidIndex)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{2, 3});

    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions + 1), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(-1), nullptr);
}


TYPED_TEST(Amp, ApplyCompletesWithoutError)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{3, 4});
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});

    ASSERT_NO_THROW(mtx->apply(x, y));
}


TYPED_TEST(Amp, AdvancedApplyCompletesWithoutError)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{3, 4});
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});
    y->fill(gko::one<value_type>());

    ASSERT_NO_THROW(mtx->apply(alpha, x, beta, y));
}


TYPED_TEST(Amp, CanConvertToDense)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto size = gko::dim<2>{2, 3};
    auto mtx = this->create_amp_from_one_dense(size);
    auto dense = Dense::create(this->exec);
    auto ref = gko::share(Dense::create(this->exec, size));
    ref->fill(gko::one<value_type>());

    mtx->convert_to(dense.get());

    EXPECT_EQ(dense->get_size(), mtx->get_size());
    GKO_ASSERT_MTX_NEAR(dense, ref, 0.0);
}


TYPED_TEST(Amp, CanMoveToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{2, 3});
    const auto original_size = mtx->get_size();
    const auto zero_size = gko::dim<2>{0, 0};
    auto dense = Dense::create(this->exec);

    mtx->move_to(dense.get());

    EXPECT_EQ(dense->get_size(), original_size);
}


TYPED_TEST(Amp, CanExtractDiagonal)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_one_dense(gko::dim<2>{3, 4});

    auto diag = mtx->extract_diagonal();

    ASSERT_NE(diag, nullptr);
    EXPECT_EQ(diag->get_size()[0],
              std::min(mtx->get_size()[0], mtx->get_size()[1]));
}
