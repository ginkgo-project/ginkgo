// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class MultiVector : public ::testing::Test {
protected:
    using value_type = T;
    MultiVector()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<gko::matrix::MultiVector<value_type>>(
              4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}}, exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::ptr_param<gko::matrix::MultiVector<value_type>> m)
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

    static void assert_empty(
        gko::ptr_param<gko::matrix::MultiVector<value_type>> m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::MultiVector<value_type>> mtx;
};

TYPED_TEST_SUITE(MultiVector, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MultiVector, CanBeEmpty)
{
    auto empty = gko::matrix::MultiVector<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(MultiVector, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::MultiVector<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(MultiVector, CanBeConstructedWithSize)
{
    auto m = gko::matrix::MultiVector<TypeParam>::create(this->exec,
                                                         gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
}


TYPED_TEST(MultiVector, CanBeConstructedWithSizeAndStride)
{
    auto m = gko::matrix::MultiVector<TypeParam>::create(this->exec,
                                                         gko::dim<2>{2, 3}, 4);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TYPED_TEST(MultiVector, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    value_type data[] = {
        1.0, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on

    auto m = gko::matrix::MultiVector<TypeParam>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(2, 1), value_type{6.0});
}


TYPED_TEST(MultiVector, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    // clang-format off
    const value_type data[] = {
        1.0, 2.0, -1.0,
        3.0, 4.0, -1.0,
        5.0, 6.0, -1.0};
    // clang-format on

    auto m = gko::matrix::MultiVector<TypeParam>::create_const(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::const_view(this->exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(2, 1), value_type{6.0});
}


TYPED_TEST(MultiVector, CreateWithSameConfigKeepsStride)
{
    auto m = gko::matrix::MultiVector<TypeParam>::create(this->exec,
                                                         gko::dim<2>{2, 3}, 4);
    auto m2 = gko::matrix::MultiVector<TypeParam>::create_with_config_of(m);

    ASSERT_EQ(m2->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m2->get_stride(), 4);
    ASSERT_EQ(m2->get_num_stored_elements(), 8);
}


TYPED_TEST(MultiVector, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx);
    ASSERT_EQ(this->mtx->get_stride(), 4);
}


TYPED_TEST(MultiVector, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>({1.0, 2.0},
                                                                  this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 2);
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});
}


TYPED_TEST(MultiVector, CanBeListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>(2, {1.0, 2.0},
                                                                  this->exec);
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 1));
    ASSERT_EQ(m->get_num_stored_elements(), 4);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
}


TYPED_TEST(MultiVector, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>(
        {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 6);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(MultiVector, CanBeDoubleListConstructedWithstride)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::MultiVector<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    ASSERT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
}


TYPED_TEST(MultiVector, CanBeCopied)
{
    auto mtx_copy = gko::matrix::MultiVector<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx);
    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->at(0) = 7;
    this->assert_equal_to_original_mtx(mtx_copy);
    ASSERT_EQ(this->mtx->get_stride(), 4);
    ASSERT_EQ(mtx_copy->get_stride(), 3);
}


TYPED_TEST(MultiVector, CanBeMoved)
{
    auto mtx_copy = gko::matrix::MultiVector<TypeParam>::create(this->exec);
    mtx_copy->move_from(this->mtx);
    this->assert_equal_to_original_mtx(mtx_copy);
    ASSERT_EQ(mtx_copy->get_stride(), 4);
}


TYPED_TEST(MultiVector, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(mtx_clone);
    ASSERT_EQ(mtx_clone->get_stride(), 3);
}


TYPED_TEST(MultiVector, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::MultiVector<TypeParam>::create(this->exec);
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


TYPED_TEST(MultiVector, GeneratesCorrectMatrixData)
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


TYPED_TEST(MultiVector, CanBeReadFromMatrixAssemblyData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::MultiVector<TypeParam>::create(this->exec);
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


TYPED_TEST(MultiVector, CanCreateDeviceView)
{
    auto view = this->mtx->get_device_view();

    EXPECT_EQ(view.size, this->mtx->get_size());
    EXPECT_EQ(view.stride, this->mtx->get_stride());
    EXPECT_EQ(view.values, this->mtx->get_values());
}


TYPED_TEST(MultiVector, CanCreateConstDeviceView)
{
    auto view = this->mtx->get_const_device_view();

    EXPECT_EQ(view.size, this->mtx->get_size());
    EXPECT_EQ(view.stride, this->mtx->get_stride());
    EXPECT_EQ(view.values, this->mtx->get_values());
}


TYPED_TEST(MultiVector, CanCreateSubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_submatrix(gko::span{0, 1}, gko::span{1, 3});

    EXPECT_EQ(submtx->get_size(), gko::dim<2>(1, 2));
    EXPECT_EQ(submtx->at(0, 0), value_type{2.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{3.0});
    EXPECT_LT(std::distance(this->mtx->get_values(), submtx->get_values()),
              this->mtx->get_num_stored_elements());
    EXPECT_EQ(&submtx->at(0, 0), &this->mtx->at(0, 1));
    EXPECT_EQ(&submtx->at(0, 1), &this->mtx->at(0, 2));
}


TYPED_TEST(MultiVector, CanCreateSubmatrixWithGlobalSize)
{
    using value_type = typename TestFixture::value_type;
    auto submtx_orig =
        this->mtx->create_submatrix(gko::span{0, 1}, gko::span{1, 3});
    auto submtx = this->mtx->create_submatrix(
        gko::local_span{0, 1}, gko::local_span{1, 3}, gko::dim<2>{1, 2});

    GKO_ASSERT_MTX_NEAR(submtx_orig, submtx, 0.0);
    EXPECT_EQ(submtx->get_values(), submtx_orig->get_values());
}


TYPED_TEST(MultiVector, CreateSubmatrixWithGlobalSizeThrowsOnIncorrectSize)
{
    EXPECT_THROW(
        this->mtx->create_submatrix(gko::local_span{0, 1},
                                    gko::local_span{1, 3}, gko::dim<2>{1, 20}),
        gko::DimensionMismatch);
}


TYPED_TEST(MultiVector, CanCreateEmptySubmatrix)
{
    using value_type = typename TestFixture::value_type;
    auto submtx = this->mtx->create_submatrix(gko::span{0, 0}, gko::span{1, 1});

    EXPECT_EQ(submtx->get_size(), gko::dim<2>{});
}


TYPED_TEST(MultiVector, CanCreateSubmatrixWithStride)
{
    using value_type = typename TestFixture::value_type;
    auto submtx =
        this->mtx->create_submatrix(gko::span{0, 2}, gko::span{0, 2}, 3);

    EXPECT_EQ(submtx->get_size(), gko::dim<2>(2, 2));
    EXPECT_EQ(submtx->get_stride(), 3);
    // The entry submtx->at(1, 0) points to the strided data of this->mtx
    // which means that it is undefined. Thus it is skipped in the tests
    EXPECT_EQ(submtx->at(0, 0), value_type{1.0});
    EXPECT_EQ(submtx->at(0, 1), value_type{2.0});
    EXPECT_EQ(submtx->at(1, 1), value_type{1.5});
    EXPECT_EQ(submtx->get_num_stored_elements(), 6);
    EXPECT_LT(std::distance(this->mtx->get_values(), submtx->get_values()),
              this->mtx->get_num_stored_elements());
    EXPECT_EQ(&submtx->at(0, 0), &this->mtx->at(0, 0));
    EXPECT_EQ(&submtx->at(0, 1), &this->mtx->at(0, 1));
    EXPECT_EQ(&submtx->at(1, 1), &this->mtx->at(1, 0));
}


TYPED_TEST(MultiVector, CanCreateRealView)
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


TYPED_TEST(MultiVector, CanMakeMutableView)
{
    auto view = gko::make_dense_view(this->mtx);

    ASSERT_EQ(view->get_values(), this->mtx->get_values());
    ASSERT_EQ(view->get_executor(), this->mtx->get_executor());
    GKO_ASSERT_MTX_NEAR(view, this->mtx, 0.0);
}


TYPED_TEST(MultiVector, CanMakeConstView)
{
    auto view = gko::make_const_dense_view(this->mtx);

    ASSERT_EQ(view->get_const_values(), this->mtx->get_const_values());
    ASSERT_EQ(view->get_executor(), this->mtx->get_executor());
    GKO_ASSERT_MTX_NEAR(view, this->mtx, 0.0);
}


class CustomMultiVector : public gko::matrix::MultiVector<>,
                          public gko::ConvertibleTo<CustomMultiVector> {
    friend class gko::matrix::MultiVector<>;

public:
    static std::unique_ptr<CustomMultiVector> create(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size, int data)
    {
        return std::unique_ptr<CustomMultiVector>(
            new CustomMultiVector(std::move(exec), size, data));
    }

    int get_data() const { return data_; }

    void convert_to(CustomMultiVector* result) const override
    {
        *result = *this;
    }

    void move_to(CustomMultiVector* result) override
    {
        *result = std::move(*this);
    }

    CustomMultiVector& operator=(const CustomMultiVector& other)
    {
        if (&other != this) {
            gko::matrix::MultiVector<>::operator=(other);
            data_ = other.data_;
        }
        return *this;
    }

    CustomMultiVector& operator=(CustomMultiVector&& other)
    {
        if (&other != this) {
            gko::matrix::MultiVector<>::operator=(std::move(other));
            data_ = std::exchange(other.data_, 0);
        }
        return *this;
    }

protected:
    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl(
        std::shared_ptr<const gko::Executor> exec) const override
    {
        return create(exec, this->get_size(), this->data_);
    }

private:
    explicit CustomMultiVector(std::shared_ptr<const gko::Executor> exec,
                               gko::dim<2> size = {}, int data = 0)
        : gko::matrix::MultiVector<>(std::move(exec), size), data_(data)
    {}

    std::unique_ptr<gko::matrix::MultiVector<>> create_view_of_impl() override
    {
        auto view = create(this->get_executor(), {}, this->get_data());
        gko::matrix::MultiVector<>::create_view_of_impl()->move_to(view);
        return view;
    }

    int data_;
};


TEST(CustomMultiVector, Clone)
{
    auto vector = CustomMultiVector::create(gko::ReferenceExecutor::create(),
                                            gko::dim<2>{3, 4}, 2);

    auto v = gko::share(vector->clone());

    GKO_ASSERT_EQ(gko::as<CustomMultiVector>(v)->get_data(), 2);
}


TEST(CustomMultiVector, CustomViewKeepsRuntimeType)
{
    auto vector = CustomMultiVector::create(gko::ReferenceExecutor::create(),
                                            gko::dim<2>{3, 4}, 2);

    auto view = gko::make_dense_view(vector);

    ASSERT_EQ(view->get_values(), vector->get_values());
    EXPECT_TRUE(dynamic_cast<CustomMultiVector*>(view.get()));
    ASSERT_EQ(dynamic_cast<CustomMultiVector*>(view.get())->get_data(), 2);
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
