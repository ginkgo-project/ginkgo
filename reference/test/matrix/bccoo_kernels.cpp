/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/bccoo.hpp>


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/bccoo_kernels.hpp"
#include "core/test/utils.hpp"


#define BCCOO_BLOCK_SIZE_TESTED 1
#define BCCOO_BLOCK_SIZE_COPIED 5


namespace {


template <typename ValueIndexType>
class Bccoo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::Bccoo<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    Bccoo()
        : exec(gko::ReferenceExecutor::create()),
          mtx_elm(Mtx::create(exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                              gko::matrix::bccoo::compression::element)),
          mtx_blk(Mtx::create(exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                              gko::matrix::bccoo::compression::block)),
          mtx(Mtx::create(exec))
    {
        mtx = gko::initialize<Mtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);
        mtx_elm =
            gko::initialize<Mtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec,
                                 index_type{BCCOO_BLOCK_SIZE_TESTED},
                                 gko::matrix::bccoo::compression::element);
        mtx_blk = gko::initialize<Mtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec,
                                       index_type{BCCOO_BLOCK_SIZE_TESTED},
                                       gko::matrix::bccoo::compression::block);
        //				if (mtx_elm->use_block_compression()) {
        //						std::cout << "MTX_ELM
        // BLOCK"
        //<< std::endl;
        //				}
        //				if (mtx_blk->use_element_compression())
        //{ 						std::cout << "MTX_BLK
        // ELEMENT"
        //<< std::endl;
        //				}
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx_elm;
    std::unique_ptr<Mtx> mtx_blk;
    std::unique_ptr<Mtx> uns_mtx;
    std::unique_ptr<Mtx> uns_mtx_elm;
    std::unique_ptr<Mtx> uns_mtx_blk;
};

TYPED_TEST_SUITE(Bccoo, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Bccoo, ConvertsToPrecisionElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx_elm->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_elm, res, residual);
}


TYPED_TEST(Bccoo, ConvertsToPrecisionBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    //		std::cout << "BEFORE NEXT PRECISION A" << std::endl;
    this->mtx_blk->convert_to(tmp.get());
    //		std::cout << "BEFORE NEXT PRECISION B" << std::endl;
    tmp->convert_to(res.get());
    //		std::cout << "AFTER  NEXT PRECISION A+B" << std::endl;

    //		std::cout << "BEFORE TESTING" << std::endl;
    GKO_ASSERT_MTX_NEAR(this->mtx_blk, res, residual);
    //		std::cout << "AFTER  TESTING" << std::endl;
}


TYPED_TEST(Bccoo, MovesToPrecisionElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx_elm->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_elm, res, residual);
}


TYPED_TEST(Bccoo, MovesToPrecisionBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx_blk->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_blk, res, residual);
}


TYPED_TEST(Bccoo, ConvertsToCooElm)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_elm = Coo::create(this->mtx_elm->get_executor());
    this->mtx_elm->convert_to(coo_mtx_elm.get());

    auto v = coo_mtx_elm->get_const_values();
    auto c = coo_mtx_elm->get_const_col_idxs();
    auto r = coo_mtx_elm->get_const_row_idxs();

    ASSERT_EQ(coo_mtx_elm->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_elm->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, ConvertsToCooBlk)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_blk = Coo::create(this->mtx_blk->get_executor());
    this->mtx_blk->convert_to(coo_mtx_blk.get());

    auto v = coo_mtx_blk->get_const_values();
    auto c = coo_mtx_blk->get_const_col_idxs();
    auto r = coo_mtx_blk->get_const_row_idxs();

    ASSERT_EQ(coo_mtx_blk->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_blk->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, MovesToCooElm)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_elm = Coo::create(this->mtx_elm->get_executor());
    this->mtx_elm->move_to(coo_mtx_elm.get());

    auto v = coo_mtx_elm->get_const_values();
    auto c = coo_mtx_elm->get_const_col_idxs();
    auto r = coo_mtx_elm->get_const_row_idxs();

    ASSERT_EQ(coo_mtx_elm->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_elm->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, MovesToCooBlk)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_blk = Coo::create(this->mtx_blk->get_executor());
    this->mtx_blk->move_to(coo_mtx_blk.get());

    auto v = coo_mtx_blk->get_const_values();
    auto c = coo_mtx_blk->get_const_col_idxs();
    auto r = coo_mtx_blk->get_const_row_idxs();

    ASSERT_EQ(coo_mtx_blk->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_blk->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, ConvertsToCsrElm)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_elm_c =
        Csr::create(this->mtx_elm->get_executor(), csr_s_classical);
    auto csr_mtx_elm_m =
        Csr::create(this->mtx_elm->get_executor(), csr_s_merge);

    this->mtx_elm->convert_to(csr_mtx_elm_c.get());
    this->mtx_elm->convert_to(csr_mtx_elm_m.get());

    auto v = csr_mtx_elm_c->get_const_values();
    auto c = csr_mtx_elm_c->get_const_col_idxs();
    auto r = csr_mtx_elm_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_elm_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_elm_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{3.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
    ASSERT_EQ(csr_mtx_elm_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_elm_c.get(), csr_mtx_elm_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_elm_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, ConvertsToCsrBlk)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_blk_c =
        Csr::create(this->mtx_blk->get_executor(), csr_s_classical);
    auto csr_mtx_blk_m =
        Csr::create(this->mtx_blk->get_executor(), csr_s_merge);

    this->mtx_blk->convert_to(csr_mtx_blk_c.get());
    this->mtx_blk->convert_to(csr_mtx_blk_m.get());

    auto v = csr_mtx_blk_c->get_const_values();
    auto c = csr_mtx_blk_c->get_const_col_idxs();
    auto r = csr_mtx_blk_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_blk_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_blk_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{3.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
    ASSERT_EQ(csr_mtx_blk_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_blk_c.get(), csr_mtx_blk_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_blk_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, MovesToCsrElm)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_elm_c =
        Csr::create(this->mtx_elm->get_executor(), csr_s_classical);
    auto csr_mtx_elm_m =
        Csr::create(this->mtx_elm->get_executor(), csr_s_merge);

    this->mtx_elm->move_to(csr_mtx_elm_c.get());
    this->mtx_elm->move_to(csr_mtx_elm_m.get());

    auto v = csr_mtx_elm_c->get_const_values();
    auto c = csr_mtx_elm_c->get_const_col_idxs();
    auto r = csr_mtx_elm_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_elm_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_elm_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{3.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
    ASSERT_EQ(csr_mtx_elm_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_elm_c.get(), csr_mtx_elm_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_elm_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, MovesToCsrBlk)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_blk_c =
        Csr::create(this->mtx_blk->get_executor(), csr_s_classical);
    auto csr_mtx_blk_m =
        Csr::create(this->mtx_blk->get_executor(), csr_s_merge);

    this->mtx_blk->move_to(csr_mtx_blk_c.get());
    this->mtx_blk->move_to(csr_mtx_blk_m.get());

    auto v = csr_mtx_blk_c->get_const_values();
    auto c = csr_mtx_blk_c->get_const_col_idxs();
    auto r = csr_mtx_blk_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_blk_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_blk_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], value_type{1.0});
    EXPECT_EQ(v[1], value_type{3.0});
    EXPECT_EQ(v[2], value_type{2.0});
    EXPECT_EQ(v[3], value_type{5.0});
    ASSERT_EQ(csr_mtx_blk_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_blk_c.get(), csr_mtx_blk_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_blk_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, ConvertsToDenseElm)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_elm = Dense::create(this->mtx_elm->get_executor());

    this->mtx_elm->convert_to(dense_mtx_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_elm,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ConvertsToDenseBlk)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_blk = Dense::create(this->mtx_blk->get_executor());

    this->mtx_blk->convert_to(dense_mtx_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_blk,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, MovesToDenseElm)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_elm = Dense::create(this->mtx_elm->get_executor());

    this->mtx_elm->move_to(dense_mtx_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_elm,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, MovesToDenseBlk)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_blk = Dense::create(this->mtx_blk->get_executor());

    this->mtx_blk->move_to(dense_mtx_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_blk,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);

    //		std::cout << "BEFORE EMPTY " << std::endl;
    empty->convert_to(res.get());
    //		std::cout << "AFTER  EMPTY " << std::endl;

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec);
    auto res = Bccoo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToPrecisionElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0,
                                    gko::matrix::bccoo::compression::element);
    auto res = Bccoo::create(this->exec);

    //		std::cout << "BEFORE EMPTY " << std::endl;
    empty->convert_to(res.get());
    //		std::cout << "AFTER  EMPTY " << std::endl;

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToPrecisionElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0,
                                    gko::matrix::bccoo::compression::element);
    auto res = Bccoo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToPrecisionBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0,
                                    gko::matrix::bccoo::compression::block);
    auto res = Bccoo::create(this->exec);

    //		std::cout << "BEFORE EMPTY " << std::endl;
    empty->convert_to(res.get());
    //		std::cout << "AFTER  EMPTY " << std::endl;

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToPrecisionBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0,
                                    gko::matrix::bccoo::compression::block);
    auto res = Bccoo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCoo)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty = Bccoo::create(this->exec);
    auto res = Coo::create(this->exec);
    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCoo)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty = Bccoo::create(this->exec);
    auto res = Coo::create(this->exec);
    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCooElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Coo::create(this->exec);
    empty_elm->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCooElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Coo::create(this->exec);
    empty_elm->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCooBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Coo::create(this->exec);
    empty_blk->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCooBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Coo::create(this->exec);
    empty_blk->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Bccoo::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Bccoo::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCsrElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Csr::create(this->exec);

    empty_elm->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCsrElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Csr::create(this->exec);

    empty_elm->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCsrBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Csr::create(this->exec);

    empty_blk->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCsrBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Csr::create(this->exec);

    empty_blk->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Bccoo::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Bccoo::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToDenseElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Dense::create(this->exec);

    empty_elm->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToDenseElm)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_elm =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::element);
    auto res = Dense::create(this->exec);

    empty_elm->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToDenseBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Dense::create(this->exec);

    empty_blk->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToDenseBlk)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_blk =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      gko::matrix::bccoo::compression::block);
    auto res = Dense::create(this->exec);

    empty_blk->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}

/*
TYPED_TEST(Bccoo, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = Vec::create(this->exec, gko::dim<2>{2, 1});
    auto y_blk = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_elm->apply(x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({13.0, 5.0}), 0.0);

    this->mtx_blk->apply(x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({13.0, 5.0}), 0.0);
}
*/

TYPED_TEST(Bccoo, AppliesToDenseVectorElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_elm->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToDenseVectorBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_blk->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedDenseVectorElm)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = MixedVec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_elm->apply(x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedDenseVectorBlk)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = MixedVec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_blk->apply(x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToDenseMatrixElm)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y_elm = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->mtx_elm->apply(x.get(), y_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_elm,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesToDenseMatrixBlk)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y_blk = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->mtx_blk->apply(x.get(), y_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_blk,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseVectorElm)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_elm->apply(alpha.get(), x.get(), beta.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseVectorBlk)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_blk->apply(alpha.get(), x.get(), beta.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToMixedDenseVectorElm)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_elm->apply(alpha.get(), x.get(), beta.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToMixedDenseVectorBlk)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_blk->apply(alpha.get(), x.get(), beta.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseMatrixElm)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_elm = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_elm->apply(alpha.get(), x.get(), beta.get(), y_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_elm,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseMatrixBlk)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_blk = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_blk->apply(alpha.get(), x.get(), beta.get(), y_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_blk,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongInnerDimensionElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_elm->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongInnerDimensionBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_blk->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfRowsElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_elm->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfRowsBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_blk->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfColsElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_elm->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfColsBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_blk->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, AppliesAddToDenseVectorElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<Vec>({2.0, 1.0}, this->exec);

    this->mtx_elm->apply2(x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToDenseVectorBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<Vec>({2.0, 1.0}, this->exec);

    this->mtx_blk->apply2(x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToMixedDenseVectorElm)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<MixedVec>({2.0, 1.0}, this->exec);

    this->mtx_elm->apply2(x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToMixedDenseVectorBlk)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<MixedVec>({2.0, 1.0}, this->exec);

    this->mtx_blk->apply2(x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToDenseMatrixElm)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_elm = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_elm->apply2(x.get(), y_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_elm,
                        l({{14.0,  4.0},
                           { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesAddToDenseMatrixBlk)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_blk = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_blk->apply2(x.get(), y_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_blk,
                        l({{14.0,  4.0},
                           { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseVectorElm)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_elm->apply2(alpha.get(), x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseVectorBlk)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_blk->apply2(alpha.get(), x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToMixedDenseVectorElm)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_elm = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_elm->apply2(alpha.get(), x.get(), y_elm.get());

    GKO_ASSERT_MTX_NEAR(y_elm, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToMixedDenseVectorBlk)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_blk = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_blk->apply2(alpha.get(), x.get(), y_blk.get());

    GKO_ASSERT_MTX_NEAR(y_blk, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseMatrixElm)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_elm = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_elm->apply2(alpha.get(), x.get(), y_elm.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_elm,
                        l({{-12.0, -3.0},
                           { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseMatrixBlk)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_blk = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_blk->apply2(alpha.get(), x.get(), y_blk.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_blk,
                        l({{-12.0, -3.0},
                           { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongInnerDimensionElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_elm->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongInnerDimensionBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_blk->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfRowsElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_elm->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfRowsBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_blk->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfColsElm)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_elm->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfColsBlk)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_blk->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ExtractsDiagonalElm)
{
    using T = typename TestFixture::value_type;
    auto matrix_elm = this->mtx_elm->clone();
    auto diag_elm = matrix_elm->extract_diagonal();

    ASSERT_EQ(diag_elm->get_size()[0], 2);
    ASSERT_EQ(diag_elm->get_size()[1], 2);
    ASSERT_EQ(diag_elm->get_values()[0], T{1.});
    ASSERT_EQ(diag_elm->get_values()[1], T{5.});
}


TYPED_TEST(Bccoo, ExtractsDiagonalBlk)
{
    using T = typename TestFixture::value_type;
    auto matrix_blk = this->mtx_blk->clone();
    auto diag_blk = matrix_blk->extract_diagonal();

    ASSERT_EQ(diag_blk->get_size()[0], 2);
    ASSERT_EQ(diag_blk->get_size()[1], 2);
    ASSERT_EQ(diag_blk->get_values()[0], T{1.});
    ASSERT_EQ(diag_blk->get_values()[1], T{5.});
}


TYPED_TEST(Bccoo, InplaceAbsoluteElm)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_elm = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED},
        gko::matrix::bccoo::compression::element);

    mtx_elm->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_elm, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Bccoo, InplaceAbsoluteBlk)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_blk = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED},
        gko::matrix::bccoo::compression::block);

    mtx_blk->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_blk, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Bccoo, OutplaceAbsoluteElm)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_elm = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED},
        gko::matrix::bccoo::compression::element);

    auto abs_mtx_elm = mtx_elm->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_elm,
                        l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}),
                        0.0);
}


TYPED_TEST(Bccoo, OutplaceAbsoluteBlk)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_blk = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED},
        gko::matrix::bccoo::compression::block);

    auto abs_mtx_blk = mtx_blk->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_blk,
                        l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}),
                        0.0);
}


TYPED_TEST(Bccoo, AppliesToComplexElm)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_elm = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_elm->apply(b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToComplexBlk)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_blk = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_blk->apply(b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedComplexElm)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_elm = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_elm->apply(b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedComplexBlk)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_blk = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_blk->apply(b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToComplexElm)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx_elm->apply(alpha.get(), b.get(), beta.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToComplexBlk)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx_blk->apply(alpha.get(), b.get(), beta.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToMixedComplexElm)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx_elm->apply(alpha.get(), b.get(), beta.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx_blk->apply(alpha.get(), b.get(), beta.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToComplexElm)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_elm->apply2(b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{complex_type{14.0, 14.0}, complex_type{21.0, 21.0}},
           {complex_type{12.0, 12.0}, complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToComplexBlk)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_blk->apply2(b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{complex_type{14.0, 14.0}, complex_type{21.0, 21.0}},
           {complex_type{12.0, 12.0}, complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToMixedComplexElm)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedVec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_elm->apply2(b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{mixed_complex_type{14.0, 14.0}, mixed_complex_type{21.0, 21.0}},
           {mixed_complex_type{12.0, 12.0}, mixed_complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToMixedComplexBlk)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedVec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_blk->apply2(b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{mixed_complex_type{14.0, 14.0}, mixed_complex_type{21.0, 21.0}},
           {mixed_complex_type{12.0, 12.0}, mixed_complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToComplexElm)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_elm->apply2(alpha.get(), b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{complex_type{-12.0, -14.0}, complex_type{-17.0, -19.0}},
           {complex_type{-8.0, -8.0}, complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToComplexBlk)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_blk->apply2(alpha.get(), b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{complex_type{-12.0, -14.0}, complex_type{-17.0, -19.0}},
           {complex_type{-8.0, -8.0}, complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToMixedComplexElm)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_elm = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_elm->apply2(alpha.get(), b.get(), x_elm.get());

    GKO_ASSERT_MTX_NEAR(
        x_elm,
        l({{mixed_complex_type{-12.0, -14.0}, mixed_complex_type{-17.0, -19.0}},
           {mixed_complex_type{-8.0, -8.0}, mixed_complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToMixedComplexBlk)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x_blk = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_blk->apply2(alpha.get(), b.get(), x_blk.get());

    GKO_ASSERT_MTX_NEAR(
        x_blk,
        l({{mixed_complex_type{-12.0, -14.0}, mixed_complex_type{-17.0, -19.0}},
           {mixed_complex_type{-8.0, -8.0}, mixed_complex_type{-12.0, -12.0}}}),
        0.0);
}


template <typename ValueIndexType>
class BccooComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Bccoo<value_type, index_type>;
};

TYPED_TEST_SUITE(BccooComplex, gko::test::ComplexValueIndexTypes);


TYPED_TEST(BccooComplex, OutplaceAbsoluteElm)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_elm = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
						index_type{BCCOO_BLOCK_SIZE_TESTED},
            gko::matrix::bccoo::compression::element);
    // clang-format on

    auto abs_mtx_elm = mtx_elm->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_elm,
                        l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
                        0.0);
}


TYPED_TEST(BccooComplex, OutplaceAbsoluteBlk)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_blk = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
						index_type{BCCOO_BLOCK_SIZE_TESTED},
            gko::matrix::bccoo::compression::block);
    // clang-format on

    auto abs_mtx_blk = mtx_blk->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_blk,
                        l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
                        0.0);
}


TYPED_TEST(BccooComplex, InplaceAbsoluteElm)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_elm = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
            index_type{BCCOO_BLOCK_SIZE_TESTED},
            gko::matrix::bccoo::compression::element);
    // clang-format on

    mtx_elm->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_elm, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(BccooComplex, InplaceAbsoluteBlk)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_blk = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
            index_type{BCCOO_BLOCK_SIZE_TESTED},
            gko::matrix::bccoo::compression::block);
    // clang-format on

    mtx_blk->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_blk, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


}  // namespace
