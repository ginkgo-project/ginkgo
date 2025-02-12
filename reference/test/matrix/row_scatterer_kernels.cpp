// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/row_scatterer_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/row_scatterer.hpp>

#include "core/test/utils.hpp"


template <typename InValueOutValueIndexType>
class RowScatter : public ::testing::Test {
protected:
    using in_value_type = std::tuple_element_t<0, InValueOutValueIndexType>;
    using out_value_type = std::tuple_element_t<1, InValueOutValueIndexType>;
    using index_type = std::tuple_element_t<2, InValueOutValueIndexType>;
    using DenseIn = gko::matrix::Dense<in_value_type>;
    using DenseOut = gko::matrix::Dense<out_value_type>;
    using Scatterer = gko::matrix::RowScatterer<index_type>;


    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();

    std::unique_ptr<DenseIn> in =
        gko::initialize<DenseIn>(I<I<in_value_type>>{{1, 2}, {3, 4}}, exec);
    std::unique_ptr<DenseIn> in_repeated =
        gko::initialize<DenseIn>(I<I<in_value_type>>{{1, 2}, {3, 4}, {5, 6}}, exec);
    std::unique_ptr<DenseOut> out = gko::initialize<DenseOut>(
        I<I<out_value_type>>{{11, 22}, {33, 44}, {55, 66}, {77, 88}}, exec);

    gko::array<index_type> idxs = {exec, {3, 1}};
    gko::array<index_type> idxs_repeated = {exec, {3, 3, 1}};

    std::unique_ptr<Scatterer> scatterer =
        Scatterer::create(exec, idxs, out->get_size()[0]);
};

#ifdef GINKGO_MIXED_PRECISION
TYPED_TEST_SUITE(RowScatter, gko::test::MixedPresisionValueIndexTypes,
                 TupleTypenameNameGenerator);
#else
TYPED_TEST_SUITE(RowScatter, gko::test::MixedPresisionValueIndexTypes,
                 TupleTypenameNameGenerator);
#endif


TYPED_TEST(RowScatter, CanRowScatter)
{
    bool invalid_access = false;

    gko::kernels::reference::row_scatter::row_scatter(
        this->exec, &this->idxs, this->in.get(), this->out.get(),
        invalid_access);

    ASSERT_FALSE(invalid_access);
    auto expected = gko::initialize<typename TestFixture::DenseOut>(
        I<I<typename TestFixture::out_value_type>>{
            {11, 22}, {3, 4}, {55, 66}, {1, 2}},
        this->exec);
    GKO_ASSERT_MTX_NEAR(this->out, expected, 0.0);
}


TYPED_TEST(RowScatter, SimpleRowScatterIsAdditive)
{
    bool invalid_access = false;

    gko::kernels::reference::row_scatter::row_scatter(
        this->exec, &this->idxs_repeated, this->in_repeated.get(), this->out.get(),
        invalid_access);

    ASSERT_FALSE(invalid_access);
    auto expected = gko::initialize<typename TestFixture::DenseOut>(
        I<I<typename TestFixture::out_value_type>>{
            {11, 22}, {5, 6}, {55, 66}, {4, 6}},
        this->exec);
    GKO_ASSERT_MTX_NEAR(this->out, expected, 0.0);
}


TYPED_TEST(RowScatter, CanDetectInvalidAccess)
{
    bool invalid_access = false;
    gko::array<typename TestFixture::index_type> idxs{this->exec, {300, 1}};

    gko::kernels::reference::row_scatter::row_scatter(
        this->exec, &idxs, this->in.get(), this->out.get(), invalid_access);

    ASSERT_TRUE(invalid_access);
}


TYPED_TEST(RowScatter, CanRowScatterSimpleApply)
{
    this->scatterer->apply(this->in.get(), this->out.get());

    auto expected = gko::initialize<typename TestFixture::DenseOut>(
        I<I<typename TestFixture::out_value_type>>{
            {11, 22}, {3, 4}, {55, 66}, {1, 2}},
        this->exec);
    GKO_ASSERT_MTX_NEAR(this->out, expected, 0.0);
}
