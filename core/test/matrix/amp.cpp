// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Amp : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

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

    std::unique_ptr<Mtx> create_amp_from_dense(gko::dim<2> size)
    {
        auto input = gko::share(Dense::create(exec, size));
        input->fill(gko::one<value_type>());
        auto factory = Mtx::build().on(exec);
        return factory->generate(input);
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_bins(), 0);
        for (int i = 0; i < Mtx::num_precisions; ++i) {
            ASSERT_EQ(m->get_bin_matrix(i), nullptr);
        }
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(Amp, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(Amp, KnowsNumPrecisions)
{
    using Mtx = typename TestFixture::Mtx;

    // num_precisions is a compile-time constant based on supported_precisions
    // tuple. Typically 2 (float, double) or up to 4 with half/bfloat16 enabled.
    ASSERT_GE(Mtx::num_precisions, 2);
    ASSERT_LE(Mtx::num_precisions, 4);
}


TYPED_TEST(Amp, HasCorrectExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto empty_input = gko::share(Dense::create(this->exec, gko::dim<2>{0, 0}));
    auto factory = Mtx::build().on(this->exec);
    auto mtx = factory->generate(empty_input);

    ASSERT_EQ(mtx->get_executor()->get_description(),
              this->exec->get_description());
}


TYPED_TEST(Amp, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto empty_input = gko::share(Dense::create(this->exec, gko::dim<2>{0, 0}));
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

    auto input = gko::share(Dense::create(this->exec, gko::dim<2>{3, 3}));
    input->fill(gko::one<typename TestFixture::value_type>());
    auto factory = Mtx::build().on(this->exec);

    ASSERT_NO_THROW(auto mtx = factory->generate(input));
}


TYPED_TEST(Amp, GeneratedMatrixHasCorrectSize)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto input = gko::share(Dense::create(this->exec, gko::dim<2>{4, 5}));
    auto factory = Mtx::build().on(this->exec);

    auto mtx = factory->generate(input);

    EXPECT_EQ(mtx->get_size(), gko::dim<2>(4, 5));
}


TYPED_TEST(Amp, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});
    auto copy = mtx->clone();

    auto copy_mtx = dynamic_cast<Mtx*>(copy.get());
    ASSERT_NE(copy_mtx, nullptr);
    EXPECT_EQ(copy_mtx->get_size(), mtx->get_size());
    EXPECT_EQ(copy_mtx->get_num_bins(), mtx->get_num_bins());
}


TYPED_TEST(Amp, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});
    auto original_size = mtx->get_size();
    auto original_bins = mtx->get_num_bins();

    auto moved = mtx->clone();
    moved->move_from(mtx);

    auto moved_mtx = dynamic_cast<Mtx*>(moved.get());
    ASSERT_NE(moved_mtx, nullptr);
    EXPECT_EQ(moved_mtx->get_size(), original_size);
    EXPECT_EQ(moved_mtx->get_num_bins(), original_bins);
}


TYPED_TEST(Amp, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{3, 4});

    auto clone = mtx->clone();

    auto cloned_mtx = dynamic_cast<Mtx*>(clone.get());
    ASSERT_NE(cloned_mtx, nullptr);
    EXPECT_EQ(cloned_mtx->get_size(), mtx->get_size());
    EXPECT_EQ(cloned_mtx->get_num_bins(), mtx->get_num_bins());
}


TYPED_TEST(Amp, CanBeCleared)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});

    mtx->clear();

    this->assert_empty(mtx.get());
}


TYPED_TEST(Amp, GetNumBinsReturnsValidValue)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});

    EXPECT_GE(mtx->get_num_bins(), 0);
    EXPECT_LE(mtx->get_num_bins(), Mtx::num_precisions);
}


TYPED_TEST(Amp, GetBinMatrixReturnsNullForInvalidIndex)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});

    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions + 1), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(-1), nullptr);
}


TYPED_TEST(Amp, CanConvertToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});
    auto dense = Dense::create(this->exec);

    mtx->convert_to(dense.get());

    EXPECT_EQ(dense->get_size(), mtx->get_size());
}


TYPED_TEST(Amp, CanMoveToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{2, 3});
    auto original_size = mtx->get_size();
    auto dense = Dense::create(this->exec);

    mtx->move_to(dense.get());

    EXPECT_EQ(dense->get_size(), original_size);
}


TYPED_TEST(Amp, CanExtractDiagonal)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_amp_from_dense(gko::dim<2>{3, 4});

    auto diag = mtx->extract_diagonal();

    ASSERT_NE(diag, nullptr);
    EXPECT_EQ(diag->get_size()[0],
              std::min(mtx->get_size()[0], mtx->get_size()[1]));
}
