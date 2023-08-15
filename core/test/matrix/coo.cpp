// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/coo.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Coo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Coo<value_type, index_type>;

    Coo()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Coo<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 4))
    {
        value_type* v = mtx->get_values();
        index_type* c = mtx->get_col_idxs();
        index_type* r = mtx->get_row_idxs();
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
        r[3] = 1;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(gko::ptr_param<const Mtx> m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_idxs();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
        EXPECT_EQ(v[3], value_type{5.0});
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_row_idxs(), nullptr);
    }
};

TYPED_TEST_SUITE(Coo, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Coo, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 4);
}


TYPED_TEST(Coo, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Coo, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Coo, CanBeCreatedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 2.0, 3.0, 4.0};
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_idxs[] = {0, 0, 1, 2};

    auto mtx = gko::matrix::Coo<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 4, values),
        gko::make_array_view(this->exec, 4, col_idxs),
        gko::make_array_view(this->exec, 4, row_idxs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_idxs(), row_idxs);
}


TYPED_TEST(Coo, CanBeCreatedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const value_type values[] = {1.0, 2.0, 3.0, 4.0};
    const index_type col_idxs[] = {0, 1, 1, 0};
    const index_type row_idxs[] = {0, 0, 1, 2};

    auto mtx = gko::matrix::Coo<value_type, index_type>::create_const(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::const_view(this->exec, 4, values),
        gko::array<index_type>::const_view(this->exec, 4, col_idxs),
        gko::array<index_type>::const_view(this->exec, 4, row_idxs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_idxs(), row_idxs);
}


TYPED_TEST(Coo, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Coo, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Coo, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Coo, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Coo, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 3.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Coo, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Coo, GeneratesCorrectMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


}  // namespace
