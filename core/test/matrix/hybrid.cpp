// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/hybrid.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


template <typename T>
struct change_index_s {
    using type = gko::int32;
};

template <>
struct change_index_s<gko::int32> {
    using type = gko::int64;
};


template <typename T>
using change_index = typename change_index_s<T>::type;


template <typename ValueIndexType>
class Hybrid : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Hybrid<value_type, index_type>;

    index_type invalid_index = gko::invalid_index<index_type>();

    Hybrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Hybrid<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 2, 2, 1))
    {
        value_type* v = mtx->get_ell_values();
        index_type* c = mtx->get_ell_col_idxs();
        c[0] = 0;
        c[1] = 1;
        c[2] = 1;
        c[3] = invalid_index;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 0.0;
        v[3] = 0.0;
        mtx->get_coo_values()[0] = 2.0;
        mtx->get_coo_col_idxs()[0] = 2;
        mtx->get_coo_row_idxs()[0] = 0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(gko::ptr_param<const Mtx> m)
    {
        auto v = m->get_const_ell_values();
        auto c = m->get_const_ell_col_idxs();
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 4);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 1);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{5.0});
        EXPECT_EQ(v[2], value_type{0.0});
        EXPECT_EQ(v[3], value_type{0.0});
        EXPECT_EQ(m->get_const_coo_values()[0], value_type{2.0});
        EXPECT_EQ(m->get_const_coo_col_idxs()[0], 2);
        EXPECT_EQ(m->get_const_coo_row_idxs()[0], 0);
    }

    void assert_empty(gko::ptr_param<const Mtx> m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_ell_values(), nullptr);
        ASSERT_EQ(m->get_const_ell_col_idxs(), nullptr);
        ASSERT_EQ(m->get_ell_num_stored_elements_per_row(), 0);
        ASSERT_EQ(m->get_ell_stride(), 0);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_coo_values(), nullptr);
        ASSERT_EQ(m->get_const_coo_col_idxs(), nullptr);
    }
};

TYPED_TEST_SUITE(Hybrid, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Hybrid, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_ell_num_stored_elements(), 4);
    ASSERT_EQ(this->mtx->get_ell_num_stored_elements_per_row(), 2);
    ASSERT_EQ(this->mtx->get_ell_stride(), 2);
    ASSERT_EQ(this->mtx->get_coo_num_stored_elements(), 1);
}


TYPED_TEST(Hybrid, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Hybrid, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Hybrid, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_ell_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Hybrid, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Hybrid, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_ell_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(static_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Hybrid, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataAutomatically)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto m =
        Mtx::create(this->exec, std::make_shared<typename Mtx::automatic>());
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    auto v = m->get_const_coo_values();
    auto c = m->get_const_coo_col_idxs();
    auto r = m->get_const_coo_row_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(m->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{0.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataByColumns2)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::column_limit>(2));
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixDataByPercent40)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::imbalance_limit>(0.4));
    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 0.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    auto v = m->get_const_ell_values();
    auto c = m->get_const_ell_col_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    auto coo_v = m->get_const_coo_values();
    auto coo_c = m->get_const_coo_col_idxs();
    auto coo_r = m->get_const_coo_row_idxs();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 2);
    ASSERT_EQ(m->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{5.0});
    EXPECT_EQ(coo_v[0], value_type{0.0});
    EXPECT_EQ(coo_v[1], value_type{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixAssemblyDataAutomatically)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m =
        Mtx::create(this->exec, std::make_shared<typename Mtx::automatic>());
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 0.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    auto v = m->get_const_coo_values();
    auto c = m->get_const_coo_col_idxs();
    auto r = m->get_const_coo_row_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(m->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{0.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixAssemblyDataByColumns2)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::column_limit>(2));
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 0.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Hybrid, CanBeReadFromMatrixAssemblyDataByPercent40)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::imbalance_limit>(0.4));
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 0.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    auto v = m->get_const_ell_values();
    auto c = m->get_const_ell_col_idxs();
    auto n = m->get_ell_num_stored_elements_per_row();
    auto p = m->get_ell_stride();
    auto coo_v = m->get_const_coo_values();
    auto coo_c = m->get_const_coo_col_idxs();
    auto coo_r = m->get_const_coo_row_idxs();
    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(m->get_ell_num_stored_elements(), 2);
    ASSERT_EQ(m->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{5.0});
    EXPECT_EQ(coo_v[0], value_type{0.0});
    EXPECT_EQ(coo_v[1], value_type{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Hybrid, GeneratesCorrectMatrixData)
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


TYPED_TEST(Hybrid, GetCorrectColumnLimit)
{
    using Mtx = typename TestFixture::Mtx;
    using Mtx2 = gko::remove_complex<Mtx>;
    using strategy = typename Mtx::column_limit;
    using strategy2 = typename Mtx2::column_limit;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>(2));
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());
    auto mtx2_stra = gko::as<strategy2>(mtx->template get_strategy<Mtx2>());

    EXPECT_EQ(mtx_stra->get_num_columns(), 2);
    EXPECT_EQ(mtx2_stra->get_num_columns(), 2);
}


TYPED_TEST(Hybrid, GetCorrectImbalanceLimit)
{
    using Mtx = typename TestFixture::Mtx;
    using Mtx2 = gko::remove_complex<Mtx>;
    using strategy = typename Mtx::imbalance_limit;
    using strategy2 = typename Mtx2::imbalance_limit;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>(0.4));
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());
    auto mtx2_stra = gko::as<strategy2>(mtx->template get_strategy<Mtx2>());

    EXPECT_EQ(mtx_stra->get_percentage(), 0.4);
    EXPECT_EQ(mtx2_stra->get_percentage(), 0.4);
}


TYPED_TEST(Hybrid, GetCorrectImbalanceBoundedLimit)
{
    using Mtx = typename TestFixture::Mtx;
    using Mtx2 = gko::remove_complex<Mtx>;
    using strategy = typename Mtx::imbalance_bounded_limit;
    using strategy2 = typename Mtx2::imbalance_bounded_limit;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>(0.4, 0.1));
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());
    auto mtx2_stra = gko::as<strategy2>(mtx->template get_strategy<Mtx2>());

    EXPECT_EQ(mtx_stra->get_percentage(), 0.4);
    EXPECT_EQ(mtx_stra->get_ratio(), 0.1);
    EXPECT_EQ(mtx2_stra->get_percentage(), 0.4);
    EXPECT_EQ(mtx2_stra->get_ratio(), 0.1);
}


TYPED_TEST(Hybrid, GetCorrectMinimalStorageLimitWithDifferentHybType)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx2 = gko::matrix::Hybrid<value_type, change_index<index_type>>;
    using strategy = typename Mtx::minimal_storage_limit;
    using strategy2 = typename Mtx2::imbalance_limit;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>());
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());
    auto mtx2_stra = gko::as<strategy2>(mtx->template get_strategy<Mtx2>());

    EXPECT_EQ(mtx2_stra->get_percentage(), mtx_stra->get_percentage());
}


TYPED_TEST(Hybrid, GetCorrectMinimalStorageLimitWithSameHybType)
{
    using Mtx = typename TestFixture::Mtx;
    using Mtx2 = Mtx;
    using strategy = typename Mtx::minimal_storage_limit;
    using strategy2 = typename Mtx2::minimal_storage_limit;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>());
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());
    auto mtx2_stra = gko::as<strategy2>(mtx->template get_strategy<Mtx2>());

    EXPECT_EQ(mtx2_stra->get_percentage(), mtx_stra->get_percentage());
}


TYPED_TEST(Hybrid, GetCorrectAutomatic)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx2 = Mtx;
    using strategy = typename Mtx::automatic;
    using strategy2 = typename Mtx2::automatic;

    auto mtx = Mtx::create(this->exec, std::make_shared<strategy>());
    auto mtx_stra = gko::as<strategy>(mtx->get_strategy());

    ASSERT_NO_THROW(gko::as<strategy2>(mtx->template get_strategy<Mtx2>()));
}
