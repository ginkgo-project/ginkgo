// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


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


TYPED_TEST(Dense, CreateWithSameConfigKeepsStride)
{
    auto m =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 3}, 4);
    auto m2 = gko::matrix::Dense<TypeParam>::create_with_config_of(m);

    ASSERT_EQ(m2->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m2->get_stride(), 4);
    ASSERT_EQ(m2->get_num_stored_elements(), 8);
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
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});
}


TYPED_TEST(Dense, CanBeListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(2, {1.0, 2.0},
                                                            this->exec);
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(Dense, CanBeDoubleListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(Dense, CanBeCopied)
{
    auto mtx_copy = gko::matrix::Dense<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx);
    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->at(0) = 7;
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


TYPED_TEST(Dense, CanBeCleared)
{
    this->mtx->clear();
    this->assert_empty(this->mtx.get());
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


TYPED_TEST(Dense, CanBeReadFromMatrixAssemblyData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::Dense<TypeParam>::create(this->exec);
    gko::matrix_assembly_data<TypeParam> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 0, 0.0);
    data.set_value(1, 1, 5.0);
    data.set_value(1, 2, 0.0);

    m->read(data);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 1), value_type{3.0});
    EXPECT_EQ(m->at(1, 1), value_type{5.0});
    EXPECT_EQ(m->at(0, 2), value_type{2.0});
    ASSERT_EQ(m->at(1, 2), value_type{0.0});
}


TYPED_TEST(Dense, CanCreateSubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_submatrix(gko::span{0, 1}, gko::span{1, 2});

    EXPECT_EQ(submtx->at(0, 0), value_type{2.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.0});
    EXPECT_EQ(submtx->at(1, 0), value_type{2.5});
    EXPECT_EQ(submtx->at(1, 1), value_type{3.5});
}


TYPED_TEST(Dense, CanCreateEmptySubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_submatrix(gko::span{0, 0}, gko::span{1, 1});

    EXPECT_EQ(submtx->get_size(), gko::dim<2>{});
}


TYPED_TEST(Dense, CanCreateSubmatrixWithStride)
{
    using value_type = typename TestFixture::value_type;
    auto submtx =
        this->mtx->create_submatrix(gko::span{1, 2}, gko::span{1, 3}, 3);

    EXPECT_EQ(submtx->at(0, 0), value_type{2.5});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.5});
    EXPECT_EQ(submtx->get_num_stored_elements(), 2);
}


TYPED_TEST(Dense, CanCreateRealView)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    auto real_view = this->mtx->create_real_view();

    if (gko::is_complex<value_type>()) {
        EXPECT_EQ(real_view->get_size()[0], this->mtx->get_size()[0]);
        EXPECT_EQ(real_view->get_size()[1], 2 * this->mtx->get_size()[1]);
        EXPECT_EQ(real_view->get_stride(), 2 * this->mtx->get_stride());
        EXPECT_EQ(real_view->at(0, 0), real_type{1.0});
        EXPECT_EQ(real_view->at(0, 1), real_type{0.0});
        EXPECT_EQ(real_view->at(0, 2), real_type{2.0});
        EXPECT_EQ(real_view->at(0, 3), real_type{0.0});
        EXPECT_EQ(real_view->at(0, 4), real_type{3.0});
        EXPECT_EQ(real_view->at(0, 5), real_type{0.0});
        EXPECT_EQ(real_view->at(1, 0), real_type{1.5});
        EXPECT_EQ(real_view->at(1, 1), real_type{0.0});
        EXPECT_EQ(real_view->at(1, 2), real_type{2.5});
        EXPECT_EQ(real_view->at(1, 3), real_type{0.0});
        EXPECT_EQ(real_view->at(1, 4), real_type{3.5});
        EXPECT_EQ(real_view->at(1, 5), real_type{0.0});
    } else {
        EXPECT_EQ(real_view->get_size()[0], this->mtx->get_size()[0]);
        EXPECT_EQ(real_view->get_size()[1], this->mtx->get_size()[1]);
        EXPECT_EQ(real_view->get_stride(), this->mtx->get_stride());
        EXPECT_EQ(real_view->at(0, 0), real_type{1.0});
        EXPECT_EQ(real_view->at(0, 1), real_type{2.0});
        EXPECT_EQ(real_view->at(0, 2), real_type{3.0});
        EXPECT_EQ(real_view->at(1, 0), real_type{1.5});
        EXPECT_EQ(real_view->at(1, 1), real_type{2.5});
        EXPECT_EQ(real_view->at(1, 2), real_type{3.5});
    }
}


TYPED_TEST(Dense, CanMakeMutableView)
{
    auto view = gko::make_dense_view(this->mtx);

    ASSERT_EQ(view->get_values(), this->mtx->get_values());
    ASSERT_EQ(view->get_executor(), this->mtx->get_executor());
    GKO_ASSERT_MTX_NEAR(view, this->mtx, 0.0);
}


TYPED_TEST(Dense, CanMakeConstView)
{
    auto view = gko::make_const_dense_view(this->mtx);

    ASSERT_EQ(view->get_const_values(), this->mtx->get_const_values());
    ASSERT_EQ(view->get_executor(), this->mtx->get_executor());
    GKO_ASSERT_MTX_NEAR(view, this->mtx, 0.0);
}


class CustomDense : public gko::EnableLinOp<CustomDense, gko::matrix::Dense<>> {
    friend class gko::EnablePolymorphicObject<CustomDense,
                                              gko::matrix::Dense<>>;

public:
    static std::unique_ptr<CustomDense> create(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size, int data)
    {
        return std::unique_ptr<CustomDense>(
            new CustomDense(std::move(exec), size, data));
    }

    int get_data() const { return data_; }

private:
    explicit CustomDense(std::shared_ptr<const gko::Executor> exec,
                         gko::dim<2> size = {}, int data = 0)
        : gko::EnableLinOp<CustomDense, gko::matrix::Dense<>>(std::move(exec),
                                                              size),
          data_(data)
    {}

    std::unique_ptr<gko::matrix::Dense<>> create_view_of_impl() override
    {
        auto view = create(this->get_executor(), {}, this->get_data());
        gko::matrix::Dense<>::create_view_of_impl()->move_to(view);
        return view;
    }

    int data_;
};


TEST(DenseView, CustomViewKeepsRuntimeType)
{
    auto vector = CustomDense::create(gko::ReferenceExecutor::create(),
                                      gko::dim<2>{3, 4}, 2);

    auto view = gko::make_dense_view(vector);

    ASSERT_EQ(view->get_values(), vector->get_values());
    EXPECT_TRUE(dynamic_cast<CustomDense*>(view.get()));
    ASSERT_EQ(dynamic_cast<CustomDense*>(view.get())->get_data(), 2);
}


}  // namespace
