// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/permutation.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Permutation : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<v_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;
    Permutation()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Permutation<i_type>::create(
              exec, gko::dim<2>{4, 3}, gko::array<i_type>{exec, {1, 0, 2, 3}}))
    {}


    static void assert_equal_to_original_mtx(
        gko::ptr_param<gko::matrix::Permutation<i_type>> m)
    {
        auto perm = m->get_permutation();
        ASSERT_EQ(m->get_size(), gko::dim<2>(4, 3));
        ASSERT_EQ(m->get_permutation_size(), 4);
        ASSERT_EQ(perm[0], 1);
        ASSERT_EQ(perm[1], 0);
        ASSERT_EQ(perm[2], 2);
        ASSERT_EQ(perm[3], 3);
    }

    static void assert_empty(gko::matrix::Permutation<i_type>* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_permutation_size(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Permutation<i_type>> mtx;
};

TYPED_TEST_SUITE(Permutation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Permutation, CanBeEmpty)
{
    using i_type = typename TestFixture::i_type;
    auto empty = gko::matrix::Permutation<i_type>::create(this->exec);

    this->assert_empty(empty.get());
}


TYPED_TEST(Permutation, ReturnsNullValuesArrayWhenEmpty)
{
    using i_type = typename TestFixture::i_type;
    auto empty = gko::matrix::Permutation<i_type>::create(this->exec);

    ASSERT_EQ(empty->get_const_permutation(), nullptr);
}


TYPED_TEST(Permutation, CanBeConstructedWithSize)
{
    using i_type = typename TestFixture::i_type;
    auto m =
        gko::matrix::Permutation<i_type>::create(this->exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_permutation_size(), 2);
}


TYPED_TEST(Permutation, FactorySetsCorrectPermuteMask)
{
    using i_type = typename TestFixture::i_type;
    auto m = gko::matrix::Permutation<i_type>::create(this->exec);
    auto mask = m->get_permute_mask();

    ASSERT_EQ(mask, gko::matrix::row_permute);
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingData)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3, 5},
        gko::make_array_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, PermutationCanBeConstructedFromExistingConstData)
{
    using i_type = typename TestFixture::i_type;
    using i_type = typename TestFixture::i_type;
    const i_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<i_type>::create_const(
        this->exec, 3, gko::array<i_type>::const_view(this->exec, 3, data));

    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, CanBeConstructedWithSizeAndMask)
{
    using i_type = typename TestFixture::i_type;
    auto m = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2, 3}, gko::matrix::column_permute);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_permutation_size(), 2);
    ASSERT_EQ(m->get_permute_mask(), gko::matrix::column_permute);
}


TYPED_TEST(Permutation, CanExplicitlyOverrideSetPermuteMask)
{
    using i_type = typename TestFixture::i_type;
    auto m = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2, 3}, gko::matrix::column_permute);

    auto mask = m->get_permute_mask();
    ASSERT_EQ(mask, gko::matrix::column_permute);

    m->set_permute_mask(gko::matrix::row_permute |
                        gko::matrix::inverse_permute);

    auto s_mask = m->get_permute_mask();
    ASSERT_EQ(s_mask, gko::matrix::row_permute | gko::matrix::inverse_permute);
}


TYPED_TEST(Permutation, PermutationThrowsforWrongRowPermDimensions)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {0, 2, 1};

    ASSERT_THROW(gko::matrix::Permutation<i_type>::create(
                     this->exec, gko::dim<2>{4, 2},
                     gko::make_array_view(this->exec, 3, data)),
                 gko::ValueMismatch);
}


TYPED_TEST(Permutation, SettingMaskDoesNotModifyData)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {1, 0, 2};

    auto m = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3, 5},
        gko::make_array_view(this->exec, 3, data));

    auto mask = m->get_permute_mask();
    ASSERT_EQ(m->get_const_permutation(), data);
    ASSERT_EQ(mask, gko::matrix::row_permute);

    m->set_permute_mask(gko::matrix::row_permute |
                        gko::matrix::inverse_permute);

    auto s_mask = m->get_permute_mask();
    ASSERT_EQ(s_mask, gko::matrix::row_permute | gko::matrix::inverse_permute);
    ASSERT_EQ(m->get_const_permutation(), data);
}


TYPED_TEST(Permutation, PermutationThrowsforWrongColPermDimensions)
{
    using i_type = typename TestFixture::i_type;
    i_type data[] = {0, 2, 1};

    ASSERT_THROW(gko::matrix::Permutation<i_type>::create(
                     this->exec, gko::dim<2>{3, 4},
                     gko::make_array_view(this->exec, 3, data),
                     gko::matrix::column_permute),
                 gko::ValueMismatch);
}


TYPED_TEST(Permutation, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Permutation, CanBeCopied)
{
    using i_type = typename TestFixture::i_type;
    auto mtx_copy = gko::matrix::Permutation<i_type>::create(this->exec);

    mtx_copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_permutation()[0] = 3;
    this->assert_equal_to_original_mtx(mtx_copy);
}


TYPED_TEST(Permutation, CanBeMoved)
{
    using i_type = typename TestFixture::i_type;
    auto mtx_copy = gko::matrix::Permutation<i_type>::create(this->exec);

    mtx_copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(mtx_copy);
}


TYPED_TEST(Permutation, CopyingPreservesMask)
{
    using i_type = typename TestFixture::i_type;
    auto mtx_copy = gko::matrix::Permutation<i_type>::create(this->exec);

    mtx_copy->copy_from(this->mtx);

    auto o_mask = this->mtx->get_permute_mask();
    auto n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, gko::matrix::row_permute);
    ASSERT_EQ(o_mask, n_mask);

    this->mtx->set_permute_mask(gko::matrix::column_permute);

    o_mask = this->mtx->get_permute_mask();
    n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, gko::matrix::column_permute);
    ASSERT_NE(o_mask, n_mask);

    mtx_copy->copy_from(this->mtx);

    n_mask = mtx_copy->get_permute_mask();
    ASSERT_EQ(o_mask, n_mask);
}


TYPED_TEST(Permutation, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Permutation, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


}  // namespace
