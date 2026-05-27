// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<gko::matrix::Dense<value_type>>(
              4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::ptr_param<gko::matrix::Dense<value_type>> m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * m->get_stride());
        EXPECT_EQ(m->at(0, 0), value_type{1.0});
        EXPECT_EQ(m->at(0, 1), value_type{2.0});
        EXPECT_EQ(m->at(0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 0), value_type{1.5});
        EXPECT_EQ(m->at(1, 1), value_type{2.5});
        ASSERT_EQ(m->at(1, 2), value_type{3.5});
    }

    static void assert_empty(gko::ptr_param<gko::matrix::Dense<value_type>> m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<value_type>> mtx;
};

TYPED_TEST_SUITE(Dense, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Dense, CanBeEmpty)
{
    auto empty = gko::matrix::Dense<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(Dense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::Dense<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(Dense, CanBeConstructedWithSize)
{
    auto m =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
}


TYPED_TEST(Dense, CanBeConstructedWithSizeAndStride)
{
    auto m =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 3}, 4);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TYPED_TEST(Dense, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    value_type data[] = {
        1.0, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on

    auto m = gko::matrix::Dense<TypeParam>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(2, 1), value_type{6.0});
}


TYPED_TEST(Dense, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    const value_type data[] = {
        1.0, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on

    auto m = gko::matrix::Dense<TypeParam>::create_const(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::const_view(this->exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(2, 1), value_type{6.0});
}


TYPED_TEST(Dense, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx);
    ASSERT_EQ(this->mtx->get_stride(), 4);
}


TYPED_TEST(Dense, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(1, 0), value_type{2});
}


TYPED_TEST(Dense, CanBeListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(2, {1.0, 2.0},
                                                            this->exec);
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0), value_type{2.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 0), value_type{3.0});
    ASSERT_EQ(m->at(1, 1), value_type{4.0});
    EXPECT_EQ(m->at(2, 0), value_type{5.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 0), value_type{3.0});
    ASSERT_EQ(m->at(1, 1), value_type{4.0});
    EXPECT_EQ(m->at(2, 0), value_type{5.0});
}


TYPED_TEST(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx);
    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->at(0, 0) = 7;
    this->assert_equal_to_original_mtx(mtx_copy);
    ASSERT_EQ(this->mtx->get_stride(), 4);
    ASSERT_EQ(mtx_copy->get_stride(), 3);
}


TYPED_TEST(Dense, CanBeMoved)
{
    auto mtx_copy = gko::matrix::Dense<TypeParam>::create(this->exec);
    mtx_copy->move_from(this->mtx);
    this->assert_equal_to_original_mtx(mtx_copy);
    ASSERT_EQ(mtx_copy->get_stride(), 4);
}


TYPED_TEST(Dense, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(mtx_clone);
    ASSERT_EQ(mtx_clone->get_stride(), 3);
}


TYPED_TEST(Dense, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::Dense<TypeParam>::create(this->exec);
    m->read(gko::matrix_data<TypeParam>{{2, 3},
                                        {{0, 0, 1.0},
                                         {0, 1, 3.0},
                                         {0, 2, 2.0},
                                         {1, 0, 0.0},
                                         {1, 1, 5.0},
                                         {1, 2, 0.0}}});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 1), value_type{3.0});
    EXPECT_EQ(m->at(1, 1), value_type{5.0});
    EXPECT_EQ(m->at(0, 2), value_type{2.0});
    ASSERT_EQ(m->at(1, 2), value_type{0.0});
}


TYPED_TEST(Dense, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    gko::matrix_data<TypeParam> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 6);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 0, value_type{1.5}));
    EXPECT_EQ(data.nonzeros[4], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data.nonzeros[5], tpl(1, 2, value_type{3.5}));
}


TYPED_TEST(Dense, CanCreateDeviceView)
{
    auto view = this->mtx->get_device_view();

    EXPECT_EQ(view.size, this->mtx->get_size());
    EXPECT_EQ(view.stride, this->mtx->get_stride());
    EXPECT_EQ(view.values, this->mtx->get_values());
}


TYPED_TEST(Dense, CanCreateConstDeviceView)
{
    auto view = this->mtx->get_const_device_view();

    EXPECT_EQ(view.size, this->mtx->get_size());
    EXPECT_EQ(view.stride, this->mtx->get_stride());
    EXPECT_EQ(view.values, this->mtx->get_values());
}


TYPED_TEST(Dense, CanCreateSubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_subview(gko::span{0, 1}, gko::span{1, 3});

    EXPECT_EQ(submtx->get_size(), gko::dim<2>(1, 2));
    EXPECT_EQ(submtx->at(0, 0), value_type{2.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.0});
    EXPECT_LT(std::distance(this->mtx->get_values(), submtx->get_values()),
              this->mtx->get_num_stored_elements());
    EXPECT_EQ(&submtx->at(0, 0), &this->mtx->at(0, 1));
    EXPECT_EQ(&submtx->at(0, 1), &this->mtx->at(0, 2));
}


TYPED_TEST(Dense, CanCreateEmptySubmatrix)
{
    auto submtx = this->mtx->create_subview(gko::span{0, 0}, gko::span{1, 1});

    EXPECT_EQ(submtx->get_size(), gko::dim<2>{});
}


TYPED_TEST(Dense, RecognizesInfiniteValue)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    value_type data[] = {
        INFINITY, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on
    auto m = gko::matrix::Dense<TypeParam>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 9, data), 3);

    ASSERT_THROW(m->validate_data(), gko::InvalidData);
}


TYPED_TEST(Dense, AllowsInfinitePaddingValue)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    value_type data[] = {
        1.0, 2.0, INFINITY,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on
    auto m = gko::matrix::Dense<TypeParam>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 9, data), 3);

    ASSERT_NO_THROW(m->validate_data());
}


}  // namespace
