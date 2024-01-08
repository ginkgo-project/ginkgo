// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/diagonal.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class Diagonal : public ::testing::Test {
protected:
    using value_type = ValueType;
    using Diag = gko::matrix::Diagonal<value_type>;

    Diagonal()
        : exec(gko::ReferenceExecutor::create()),
          diag(gko::matrix::Diagonal<value_type>::create(exec, 3u))
    {
        value_type* v = diag->get_values();
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Diag> diag;

    void assert_equal_to_original_mtx(gko::ptr_param<const Diag> m)
    {
        auto v = m->get_const_values();
        ASSERT_EQ(m->get_size(), gko::dim<2>(3, 3));
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
    }

    void assert_empty(const Diag* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_const_values(), nullptr);
    }
};

TYPED_TEST_SUITE(Diagonal, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Diagonal, KnowsItsSize)
{
    ASSERT_EQ(this->diag->get_size(), gko::dim<2>(3, 3));
}


TYPED_TEST(Diagonal, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->diag);
}


TYPED_TEST(Diagonal, CanBeEmpty)
{
    using Diag = typename TestFixture::Diag;
    auto diag = Diag::create(this->exec);

    this->assert_empty(diag.get());
}


TYPED_TEST(Diagonal, CanBeCreatedFromExistingData)
{
    using Diag = typename TestFixture::Diag;
    using value_type = typename TestFixture::value_type;
    value_type values[] = {1.0, 2.0, 3.0};

    auto diag = gko::matrix::Diagonal<value_type>::create(
        this->exec, 3, gko::make_array_view(this->exec, 3, values));

    ASSERT_EQ(diag->get_const_values(), values);
}


TYPED_TEST(Diagonal, CanBeCreatedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    const value_type values[] = {1.0, 2.0, 3.0};

    auto diag = gko::matrix::Diagonal<value_type>::create_const(
        this->exec, 3,
        gko::array<value_type>::const_view(this->exec, 3, values));

    ASSERT_EQ(diag->get_const_values(), values);
}


TYPED_TEST(Diagonal, CanBeCopied)
{
    using Diag = typename TestFixture::Diag;
    auto copy = Diag::create(this->exec);

    copy->copy_from(this->diag);

    this->assert_equal_to_original_mtx(this->diag);
    this->diag->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Diagonal, CanBeMoved)
{
    using Diag = typename TestFixture::Diag;
    auto copy = Diag::create(this->exec);

    copy->move_from(this->diag);

    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Diagonal, CanBeCloned)
{
    using Diag = typename TestFixture::Diag;

    auto clone = this->diag->clone();

    this->assert_equal_to_original_mtx(this->diag);
    this->diag->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Diag*>(clone.get()));
}


TYPED_TEST(Diagonal, CanBeCleared)
{
    this->diag->clear();

    this->assert_empty(this->diag.get());
}


TYPED_TEST(Diagonal, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::Diagonal<TypeParam>::create(this->exec);
    m->read(gko::matrix_data<TypeParam>{{3, 3}, {{2, 2, 2.0}, {0, 0, 1.0}}});

    const auto values = m->get_const_values();

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 3));
    EXPECT_EQ(values[0], value_type{1.0});
    EXPECT_EQ(values[1], value_type{0.0});
    EXPECT_EQ(values[2], value_type{2.0});
}


TYPED_TEST(Diagonal, CannotBeReadFromNonSquareMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::Diagonal<TypeParam>::create(this->exec);

    ASSERT_THROW(m->read(gko::matrix_data<TypeParam>{
                     {3, 4}, {{0, 0, 1.0}, {1, 1, 3.0}, {2, 2, 2.0}}}),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<value_type>::nonzero_type;
    gko::matrix_data<value_type> data;

    this->diag->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
    ASSERT_EQ(data.nonzeros.size(), 3);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(1, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(2, 2, value_type{2.0}));
}


}  // namespace
