/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


using namespace gko::matrix::bccoo;


namespace {


constexpr static int BCCOO_BLOCK_SIZE_TESTED = 1;


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
          mtx_ind(Mtx::create(exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                              compression::individual)),
          mtx_grp(Mtx::create(exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                              compression::group))
    {
        mtx_ind = gko::initialize<Mtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec,
                                       index_type{BCCOO_BLOCK_SIZE_TESTED},
                                       compression::individual);
        mtx_grp = gko::initialize<Mtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec,
                                       index_type{BCCOO_BLOCK_SIZE_TESTED},
                                       compression::group);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx_ind;
    std::unique_ptr<Mtx> mtx_grp;
    std::unique_ptr<Mtx> uns_mtx;
    std::unique_ptr<Mtx> uns_mtx_ind;
    std::unique_ptr<Mtx> uns_mtx_grp;
};

TYPED_TEST_SUITE(Bccoo, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Bccoo, ConvertsToPrecisionInd)
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

    this->mtx_ind->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_ind, res, residual);
}


TYPED_TEST(Bccoo, ConvertsToPrecisionGrp)
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

    this->mtx_grp->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_grp, res, residual);
}


TYPED_TEST(Bccoo, MovesToPrecisionInd)
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

    this->mtx_ind->convert_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_ind, res, residual);
}


TYPED_TEST(Bccoo, MovesToPrecisionGrp)
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

    this->mtx_grp->convert_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_grp, res, residual);
}


TYPED_TEST(Bccoo, ConvertsToCooInd)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_ind = Coo::create(this->mtx_ind->get_executor());
    this->mtx_ind->convert_to(coo_mtx_ind.get());

    auto v = coo_mtx_ind->get_const_values();
    auto c = coo_mtx_ind->get_const_col_idxs();
    auto r = coo_mtx_ind->get_const_row_idxs();
    ASSERT_EQ(coo_mtx_ind->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_ind->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, ConvertsToCooGrp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_grp = Coo::create(this->mtx_grp->get_executor());
    this->mtx_grp->convert_to(coo_mtx_grp.get());

    auto v = coo_mtx_grp->get_const_values();
    auto c = coo_mtx_grp->get_const_col_idxs();
    auto r = coo_mtx_grp->get_const_row_idxs();
    ASSERT_EQ(coo_mtx_grp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_grp->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, MovesToCooInd)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_ind = Coo::create(this->mtx_ind->get_executor());
    this->mtx_ind->move_to(coo_mtx_ind.get());

    auto v = coo_mtx_ind->get_const_values();
    auto c = coo_mtx_ind->get_const_col_idxs();
    auto r = coo_mtx_ind->get_const_row_idxs();
    ASSERT_EQ(coo_mtx_ind->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_ind->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, MovesToCooGrp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Coo = typename TestFixture::Coo;

    auto coo_mtx_grp = Coo::create(this->mtx_grp->get_executor());
    this->mtx_grp->move_to(coo_mtx_grp.get());

    auto v = coo_mtx_grp->get_const_values();
    auto c = coo_mtx_grp->get_const_col_idxs();
    auto r = coo_mtx_grp->get_const_row_idxs();
    ASSERT_EQ(coo_mtx_grp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx_grp->get_num_stored_elements(), 4);
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


TYPED_TEST(Bccoo, ConvertsToCsrInd)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_ind_c =
        Csr::create(this->mtx_ind->get_executor(), csr_s_classical);
    auto csr_mtx_ind_m =
        Csr::create(this->mtx_ind->get_executor(), csr_s_merge);

    this->mtx_ind->convert_to(csr_mtx_ind_c.get());
    this->mtx_ind->convert_to(csr_mtx_ind_m.get());

    auto v = csr_mtx_ind_c->get_const_values();
    auto c = csr_mtx_ind_c->get_const_col_idxs();
    auto r = csr_mtx_ind_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_ind_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_ind_c->get_num_stored_elements(), 4);
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
    ASSERT_EQ(csr_mtx_ind_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_ind_c.get(), csr_mtx_ind_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_ind_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, ConvertsToCsrGrp)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_grp_c =
        Csr::create(this->mtx_grp->get_executor(), csr_s_classical);
    auto csr_mtx_grp_m =
        Csr::create(this->mtx_grp->get_executor(), csr_s_merge);

    this->mtx_grp->convert_to(csr_mtx_grp_c.get());
    this->mtx_grp->convert_to(csr_mtx_grp_m.get());

    auto v = csr_mtx_grp_c->get_const_values();
    auto c = csr_mtx_grp_c->get_const_col_idxs();
    auto r = csr_mtx_grp_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_grp_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_grp_c->get_num_stored_elements(), 4);
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
    ASSERT_EQ(csr_mtx_grp_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_grp_c.get(), csr_mtx_grp_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_grp_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, MovesToCsrInd)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_ind_c =
        Csr::create(this->mtx_ind->get_executor(), csr_s_classical);
    auto csr_mtx_ind_m =
        Csr::create(this->mtx_ind->get_executor(), csr_s_merge);

    this->mtx_ind->clone()->move_to(csr_mtx_ind_c.get());
    this->mtx_ind->move_to(csr_mtx_ind_m.get());

    auto v = csr_mtx_ind_c->get_const_values();
    auto c = csr_mtx_ind_c->get_const_col_idxs();
    auto r = csr_mtx_ind_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_ind_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_ind_c->get_num_stored_elements(), 4);
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
    ASSERT_EQ(csr_mtx_ind_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_ind_c.get(), csr_mtx_ind_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_ind_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, MovesToCsrGrp)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_grp_c =
        Csr::create(this->mtx_grp->get_executor(), csr_s_classical);
    auto csr_mtx_grp_m =
        Csr::create(this->mtx_grp->get_executor(), csr_s_merge);

    this->mtx_grp->clone()->move_to(csr_mtx_grp_c.get());
    this->mtx_grp->move_to(csr_mtx_grp_m.get());

    auto v = csr_mtx_grp_c->get_const_values();
    auto c = csr_mtx_grp_c->get_const_col_idxs();
    auto r = csr_mtx_grp_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_grp_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_grp_c->get_num_stored_elements(), 4);
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
    ASSERT_EQ(csr_mtx_grp_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_grp_c.get(), csr_mtx_grp_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_grp_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Bccoo, ConvertsToDenseInd)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_ind = Dense::create(this->mtx_ind->get_executor());

    this->mtx_ind->convert_to(dense_mtx_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_ind,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ConvertsToDenseGrp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_grp = Dense::create(this->mtx_grp->get_executor());

    this->mtx_grp->convert_to(dense_mtx_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_grp,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, MovesToDenseInd)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_ind = Dense::create(this->mtx_ind->get_executor());

    this->mtx_ind->move_to(dense_mtx_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_ind,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, MovesToDenseGrp)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx_grp = Dense::create(this->mtx_grp->get_executor());

    this->mtx_grp->move_to(dense_mtx_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx_grp,
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

    empty->convert_to(res.get());

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


TYPED_TEST(Bccoo, ConvertsEmptyToPrecisionInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0, compression::individual);
    auto res = Bccoo::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToPrecisionInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0, compression::individual);
    auto res = Bccoo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToPrecisionGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0, compression::group);
    auto res = Bccoo::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToPrecisionGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = typename TestFixture::Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto empty = OtherBccoo::create(this->exec, 0, compression::group);
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


TYPED_TEST(Bccoo, ConvertsEmptyToCooInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Coo::create(this->exec);
    empty_ind->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCooInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Coo::create(this->exec);
    empty_ind->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCooGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Coo::create(this->exec);
    empty_grp->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCooGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Coo = gko::matrix::Coo<ValueType, IndexType>;

    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Coo::create(this->exec);
    empty_grp->move_to(res.get());

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


TYPED_TEST(Bccoo, ConvertsEmptyToCsrInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Csr::create(this->exec);

    empty_ind->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCsrInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Csr::create(this->exec);

    empty_ind->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToCsrGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Csr::create(this->exec);

    empty_grp->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToCsrGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Csr::create(this->exec);

    empty_grp->move_to(res.get());

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


TYPED_TEST(Bccoo, ConvertsEmptyToDenseInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Dense::create(this->exec);

    empty_ind->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToDenseInd)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_ind =
        Bccoo::create(this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED},
                      compression::individual);
    auto res = Dense::create(this->exec);

    empty_ind->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, ConvertsEmptyToDenseGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Dense::create(this->exec);

    empty_grp->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, MovesEmptyToDenseGrp)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Bccoo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty_grp = Bccoo::create(
        this->exec, IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);
    auto res = Dense::create(this->exec);

    empty_grp->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Bccoo, AppliesToDenseVectorInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_ind->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToDenseVectorGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_grp->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedDenseVectorInd)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = MixedVec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_ind->apply(x.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedDenseVectorGrp)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = MixedVec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx_grp->apply(x.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesToDenseMatrixInd)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y_ind = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->mtx_ind->apply(x.get(), y_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_ind,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesToDenseMatrixGrp)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y_grp = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->mtx_grp->apply(x.get(), y_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_grp,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseVectorInd)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_ind->apply(alpha.get(), x.get(), beta.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseVectorGrp)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_grp->apply(alpha.get(), x.get(), beta.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToMixedDenseVectorInd)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_ind->apply(alpha.get(), x.get(), beta.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToMixedDenseVectorGrp)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_grp->apply(alpha.get(), x.get(), beta.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseMatrixInd)
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
    auto y_ind = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_ind->apply(alpha.get(), x.get(), beta.get(), y_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_ind,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationToDenseMatrixGrp)
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
    auto y_grp = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_grp->apply(alpha.get(), x.get(), beta.get(), y_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_grp,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongInnerDimensionInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_ind->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongInnerDimensionGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_grp->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfRowsInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_ind->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfRowsGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_grp->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfColsInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_ind->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyFailsOnWrongNumberOfColsGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_grp->apply(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, AppliesAddToDenseVectorInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<Vec>({2.0, 1.0}, this->exec);

    this->mtx_ind->apply2(x.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToDenseVectorGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<Vec>({2.0, 1.0}, this->exec);

    this->mtx_grp->apply2(x.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToMixedDenseVectorInd)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<MixedVec>({2.0, 1.0}, this->exec);

    this->mtx_ind->apply2(x.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToMixedDenseVectorGrp)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<MixedVec>({2.0, 1.0}, this->exec);

    this->mtx_grp->apply2(x.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesAddToDenseMatrixInd)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_ind = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_ind->apply2(x.get(), y_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_ind,
                        l({{14.0,  4.0},
                           { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesAddToDenseMatrixGrp)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_grp = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_grp->apply2(x.get(), y_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_grp,
                        l({{14.0,  4.0},
                           { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseVectorInd)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_ind->apply2(alpha.get(), x.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseVectorGrp)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx_grp->apply2(alpha.get(), x.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToMixedDenseVectorInd)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_ind = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_ind->apply2(alpha.get(), x.get(), y_ind.get());

    GKO_ASSERT_MTX_NEAR(y_ind, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToMixedDenseVectorGrp)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y_grp = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx_grp->apply2(alpha.get(), x.get(), y_grp.get());

    GKO_ASSERT_MTX_NEAR(y_grp, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseMatrixInd)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_ind = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_ind->apply2(alpha.get(), x.get(), y_ind.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_ind,
                        l({{-12.0, -3.0},
                           { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, AppliesLinearCombinationAddToDenseMatrixGrp)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y_grp = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx_grp->apply2(alpha.get(), x.get(), y_grp.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y_grp,
                        l({{-12.0, -3.0},
                           { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongInnerDimensionInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_ind->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongInnerDimensionGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_grp->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfRowsInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_ind->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfRowsGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx_grp->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfColsInd)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_ind->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ApplyAddFailsOnWrongNumberOfColsGrp)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx_grp->apply2(x.get(), y.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Bccoo, ExtractsDiagonalInd)
{
    using T = typename TestFixture::value_type;
    auto matrix_ind = this->mtx_ind->clone();
    auto diag_ind = matrix_ind->extract_diagonal();

    ASSERT_EQ(diag_ind->get_size()[0], 2);
    ASSERT_EQ(diag_ind->get_size()[1], 2);
    ASSERT_EQ(diag_ind->get_values()[0], T{1.});
    ASSERT_EQ(diag_ind->get_values()[1], T{5.});
}


TYPED_TEST(Bccoo, ExtractsDiagonalGrp)
{
    using T = typename TestFixture::value_type;
    auto matrix_grp = this->mtx_grp->clone();
    auto diag_grp = matrix_grp->extract_diagonal();

    ASSERT_EQ(diag_grp->get_size()[0], 2);
    ASSERT_EQ(diag_grp->get_size()[1], 2);
    ASSERT_EQ(diag_grp->get_values()[0], T{1.});
    ASSERT_EQ(diag_grp->get_values()[1], T{5.});
}


TYPED_TEST(Bccoo, InplaceAbsoluteInd)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_ind = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::individual);

    mtx_ind->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_ind, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Bccoo, InplaceAbsoluteGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_grp = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);

    mtx_grp->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_grp, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Bccoo, OutplaceAbsoluteInd)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_ind = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::individual);

    auto abs_mtx_ind = mtx_ind->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_ind,
                        l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}),
                        0.0);
}


TYPED_TEST(Bccoo, OutplaceAbsoluteGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using IndexType = typename TestFixture::index_type;
    auto mtx_grp = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        IndexType{BCCOO_BLOCK_SIZE_TESTED}, compression::group);

    auto abs_mtx_grp = mtx_grp->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_grp,
                        l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}),
                        0.0);
}


TYPED_TEST(Bccoo, AppliesToComplexInd)
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
    auto x_ind = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_ind->apply(b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToComplexGrp)
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
    auto x_grp = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_grp->apply(b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedComplexInd)
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
    auto x_ind = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_ind->apply(b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AppliesToMixedComplexGrp)
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
    auto x_grp = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx_grp->apply(b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToComplexInd)
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
    auto x_ind = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx_ind->apply(alpha.get(), b.get(), beta.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToComplexGrp)
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
    auto x_grp = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx_grp->apply(alpha.get(), b.get(), beta.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, AdvancedAppliesToMixedComplexInd)
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
    auto x_ind = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx_ind->apply(alpha.get(), b.get(), beta.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
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
    auto x_grp = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx_grp->apply(alpha.get(), b.get(), beta.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToComplexInd)
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
    auto x_ind = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_ind->apply2(b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{complex_type{14.0, 14.0}, complex_type{21.0, 21.0}},
           {complex_type{12.0, 12.0}, complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToComplexGrp)
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
    auto x_grp = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_grp->apply2(b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{complex_type{14.0, 14.0}, complex_type{21.0, 21.0}},
           {complex_type{12.0, 12.0}, complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToMixedComplexInd)
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
    auto x_ind = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_ind->apply2(b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{mixed_complex_type{14.0, 14.0}, mixed_complex_type{21.0, 21.0}},
           {mixed_complex_type{12.0, 12.0}, mixed_complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsToMixedComplexGrp)
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
    auto x_grp = gko::initialize<MixedVec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    // clang-format on

    this->mtx_grp->apply2(b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{mixed_complex_type{14.0, 14.0}, mixed_complex_type{21.0, 21.0}},
           {mixed_complex_type{12.0, 12.0}, mixed_complex_type{18.0, 18.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToComplexInd)
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
    auto x_ind = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_ind->apply2(alpha.get(), b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{complex_type{-12.0, -14.0}, complex_type{-17.0, -19.0}},
           {complex_type{-8.0, -8.0}, complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToComplexGrp)
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
    auto x_grp = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_grp->apply2(alpha.get(), b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
        l({{complex_type{-12.0, -14.0}, complex_type{-17.0, -19.0}},
           {complex_type{-8.0, -8.0}, complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToMixedComplexInd)
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
    auto x_ind = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_ind->apply2(alpha.get(), b.get(), x_ind.get());

    GKO_ASSERT_MTX_NEAR(
        x_ind,
        l({{mixed_complex_type{-12.0, -14.0}, mixed_complex_type{-17.0, -19.0}},
           {mixed_complex_type{-8.0, -8.0}, mixed_complex_type{-12.0, -12.0}}}),
        0.0);
}


TYPED_TEST(Bccoo, ApplyAddsScaledToMixedComplexGrp)
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
    auto x_grp = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    // clang-format on

    this->mtx_grp->apply2(alpha.get(), b.get(), x_grp.get());

    GKO_ASSERT_MTX_NEAR(
        x_grp,
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


TYPED_TEST(BccooComplex, OutplaceAbsoluteInd)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_ind = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
						index_type{BCCOO_BLOCK_SIZE_TESTED},
            compression::individual);
    // clang-format on

    auto abs_mtx_ind = mtx_ind->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_ind,
                        l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
                        0.0);
}


TYPED_TEST(BccooComplex, OutplaceAbsoluteGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_grp = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
						index_type{BCCOO_BLOCK_SIZE_TESTED},
            compression::group);
    // clang-format on

    auto abs_mtx_grp = mtx_grp->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_grp,
                        l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
                        0.0);
}


TYPED_TEST(BccooComplex, InplaceAbsoluteInd)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_ind = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
            index_type{BCCOO_BLOCK_SIZE_TESTED},
            compression::individual);
    // clang-format on

    mtx_ind->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_ind, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(BccooComplex, InplaceAbsoluteGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx_grp = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec,
            index_type{BCCOO_BLOCK_SIZE_TESTED},
            compression::group);
    // clang-format on

    mtx_grp->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx_grp, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


}  // namespace
