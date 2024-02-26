// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sellp.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Sellp : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Sellp<value_type, index_type>;

    index_type invalid_index = gko::invalid_index<index_type>();

    Sellp()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Sellp<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 3))
    {
        mtx->read(
            {{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(gko::ptr_param<const Mtx> m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto l = m->get_const_slice_lengths();
        auto s = m->get_const_slice_sets();
        auto slice_size = m->get_slice_size();
        auto stride_factor = m->get_stride_factor();
        auto total_cols = m->get_total_cols();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 192);
        ASSERT_EQ(m->get_slice_size(), gko::matrix::default_slice_size);
        ASSERT_EQ(m->get_stride_factor(), gko::matrix::default_stride_factor);
        ASSERT_EQ(m->get_total_cols(), 3);
        EXPECT_EQ(l[0], 3);
        EXPECT_EQ(s[0], 0);
        EXPECT_EQ(s[1], 3);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
        EXPECT_EQ(c[gko::matrix::default_slice_size + 1], invalid_index);
        EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
        EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[gko::matrix::default_slice_size], value_type{0.0});
        EXPECT_EQ(v[gko::matrix::default_slice_size + 1], value_type{0.0});
        EXPECT_EQ(v[2 * gko::matrix::default_slice_size], value_type{2.0});
        EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], value_type{0.0});
    }

    void assert_equal_to_original_mtx_with_slice_size_and_stride_factor(
        const Mtx* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto l = m->get_const_slice_lengths();
        auto s = m->get_const_slice_sets();
        auto slice_size = m->get_slice_size();
        auto stride_factor = m->get_stride_factor();
        auto total_cols = m->get_total_cols();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 8);
        ASSERT_EQ(m->get_slice_size(), 2);
        ASSERT_EQ(m->get_stride_factor(), 2);
        ASSERT_EQ(m->get_total_cols(), 4);
        EXPECT_EQ(l[0], 4);
        EXPECT_EQ(s[0], 0);
        EXPECT_EQ(s[1], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], invalid_index);
        EXPECT_EQ(c[4], 2);
        EXPECT_EQ(c[5], invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[2], value_type{0.0});
        EXPECT_EQ(v[3], value_type{0.0});
        EXPECT_EQ(v[4], value_type{2.0});
        EXPECT_EQ(v[5], value_type{0.0});
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_total_cols(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_slice_lengths(), nullptr);
        ASSERT_NE(m->get_const_slice_sets(), nullptr);
    }
};

TYPED_TEST_SUITE(Sellp, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Sellp, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 192);
    ASSERT_EQ(this->mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(this->mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    ASSERT_EQ(this->mtx->get_total_cols(), 3);
}


TYPED_TEST(Sellp, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Sellp, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Sellp, CanBeConstructedWithSliceSizeAndStrideFactor)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec, gko::dim<2>{2, 3}, 2, 2, 3);

    ASSERT_EQ(mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(mtx->get_slice_size(), 2);
    ASSERT_EQ(mtx->get_stride_factor(), 2);
    ASSERT_EQ(mtx->get_total_cols(), 3);
}


TYPED_TEST(Sellp, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Sellp, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Sellp, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Sellp, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Sellp, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Sellp, CanBeReadFromMatrixDataWithSliceSizeAndStrideFactor)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec, gko::dim<2>{2, 3}, 2, 2, 3);
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx_with_slice_size_and_stride_factor(
        m.get());
}


TYPED_TEST(Sellp, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{0.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


TYPED_TEST(Sellp, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 0.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Sellp, CanBeReadFromMatrixAssemblyDataWithSliceSizeAndStrideFactor)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec, gko::dim<2>{2, 3}, 2, 2, 3);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 0.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx_with_slice_size_and_stride_factor(
        m.get());
}
