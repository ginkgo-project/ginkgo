/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/dense.hpp>


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
                                    exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}},
                                    exec)),
          mtx5(gko::initialize<Mtx>(
              {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec)),
          mtx6(gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx7(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
    std::unique_ptr<Mtx> mtx5;
    std::unique_ptr<Mtx> mtx6;
    std::unique_ptr<Mtx> mtx7;

    std::ranlux48 rand_engine;

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, exec);
    }
};


TYPED_TEST_SUITE(Dense, gko::test::ValueTypes);


TYPED_TEST(Dense, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    this->mtx2->apply(this->mtx1.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{-0.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{1.0});
    ASSERT_EQ(this->mtx3->at(1, 2), T{1.0});
}


TYPED_TEST(Dense, AppliesLinearCombinationToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);

    this->mtx2->apply(alpha.get(), this->mtx1.get(), beta.get(),
                      this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{2.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{4.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{6.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{0.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{2.0});
    ASSERT_EQ(this->mtx3->at(1, 2), T{4.0});
}


TYPED_TEST(Dense, ApplyFailsOnWrongInnerDimension)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(this->mtx1.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfRows)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{3});

    ASSERT_THROW(this->mtx1->apply(this->mtx2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfCols)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2}, 3);

    ASSERT_THROW(this->mtx1->apply(this->mtx2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{2.0, -2.0}}, this->exec);

    this->mtx2->scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(Dense, ScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);

    this->mtx2->scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{-2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{4.0});
}


TYPED_TEST(Dense, ScalesDataWithStride)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{-1.0, 1.0, 2.0}}, this->exec);

    this->mtx1->scale(alpha.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{-1.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{6.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{-1.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{2.5});
    ASSERT_EQ(this->mtx1->at(1, 2), T{7.0});
}


TYPED_TEST(Dense, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{2.0, 1.0, -2.0}}, this->exec);

    this->mtx1->add_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    ASSERT_EQ(this->mtx1->at(1, 2), T{-1.5});
}


TYPED_TEST(Dense, AddsScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);

    this->mtx1->add_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{6.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{9.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{5.5});
    ASSERT_EQ(this->mtx1->at(1, 2), T{8.5});
}


TYPED_TEST(Dense, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->add_scaled(alpha.get(), this->mtx2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, AddsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2, I<T>{3.0, 2.0});

    this->mtx2->add_scaled(alpha.get(), diag.get());

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(Dense, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(Dense, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 1), T_nc{5.0});
}


TYPED_TEST(Dense, ComputDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ConvertsToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, MovesToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, ConvertsToCoo32)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->convert_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToCoo32)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->move_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToCoo64)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int64>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->convert_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToCoo64)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int64>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->move_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToCsr32)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);

    this->mtx4->convert_to(csr_mtx_c.get());
    this->mtx4->convert_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, MovesToCsr32)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx4->clone();

    this->mtx4->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, ConvertsToCsr64)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int64>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);

    this->mtx4->convert_to(csr_mtx_c.get());
    this->mtx4->convert_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, MovesToCsr64)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int64>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx4->clone();

    this->mtx4->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, ConvertsToSparsityCsr32)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->convert_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, MovesToSparsityCsr32)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->move_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, ConvertsToSparsityCsr64)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int64>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->convert_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, MovesToSparsityCsr64)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int64>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->move_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, ConvertsToEll32)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, MovesToEll32)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, ConvertsToEll64)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int64>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, MovesToEll64)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int64>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, ConvertsToEllWithStride)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor(), gko::dim<2>{}, 0, 3);

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{2.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
}


TYPED_TEST(Dense, MovesToEllWithStride)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor(), gko::dim<2>{}, 0, 3);

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{2.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
}


TYPED_TEST(Dense, MovesToHybridAutomatically32)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
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
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridAutomatically32)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
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
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridAutomatically64)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int64>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
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
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridAutomatically64)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int64>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
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
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridWithStrideAutomatically)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3);

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideAutomatically)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3);

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridWithStrideAndCooLengthByColumns2)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 3);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{3.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], T{2.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[1], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[2], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[2], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[2], 0);
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideAndCooLengthByColumns2)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 3);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{3.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], T{2.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[1], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[2], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[2], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[2], 0);
}


TYPED_TEST(Dense, MovesToHybridWithStrideByPercent40)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], T{3.0});
    EXPECT_EQ(coo_v[1], T{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideByPercent40)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{}, 0, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], T{3.0});
    EXPECT_EQ(coo_v[1], T{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Dense, ConvertsToSellp32)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, MovesToSellp32)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, ConvertsToSellp64)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int64>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, MovesToSellp64)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int64>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, ConvertsToSellpWithSliceSizeAndStrideFactor)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 0);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
    EXPECT_EQ(v[4], T{3.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(v[6], T{0.0});
    EXPECT_EQ(v[7], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TYPED_TEST(Dense, MovesToSellpWithSliceSizeAndStrideFactor)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 0);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
    EXPECT_EQ(v[4], T{3.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(v[6], T{0.0});
    EXPECT_EQ(v[7], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TYPED_TEST(Dense, ConvertsToAndFromSellpWithMoreThanOneSlice)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto x = this->template gen_mtx<Mtx>(65, 25);

    auto sellp_mtx = Sellp::create(this->exec);
    auto dense_mtx = Mtx::create(this->exec);
    x->convert_to(sellp_mtx.get());
    sellp_mtx->convert_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), x.get(), r<TypeParam>::value);
}


TYPED_TEST(Dense, ConvertsEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx5->transpose();
    auto trans_as_dense = static_cast<Mtx *>(trans.get());

    GKO_ASSERT_MTX_NEAR(
        trans_as_dense,
        l({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        r<TypeParam>::value);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx4->transpose();
    auto trans_as_dense = static_cast<Mtx *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_dense, l({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Dense, SquareMatrixCanGatherRows)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 0}};

    auto row_gathered = this->mtx5->row_gather(&permute_idxs);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_gathered,
                        l({{-2.0, 2.0, 4.5},
                           {1.0, -1.0, -0.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixCanGatherRowsIntoDense)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 0}};
    auto row_gathered = Mtx::create(exec, gko::dim<2>{2, 3});

    this->mtx5->row_gather(&permute_idxs, row_gathered.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_gathered,
                        l({{-2.0, 2.0, 4.5},
                           {1.0, -1.0, -0.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixCanGatherRows64)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int64> permute_idxs{exec, {1, 0}};

    auto row_gathered = this->mtx5->row_gather(&permute_idxs);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_gathered,
                        l({{-2.0, 2.0, 4.5},
                           {1.0, -1.0, -0.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixCanGatherRowsIntoDense64)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int64> permute_idxs{exec, {1, 0}};
    auto row_gathered = Mtx::create(exec, gko::dim<2>{2, 3});

    this->mtx5->row_gather(&permute_idxs, row_gathered.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_gathered,
                        l({{-2.0, 2.0, 4.5},
                           {1.0, -1.0, -0.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixIsRowPermutable)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto row_permute = this->mtx5->row_permute(&permute_idxs);
    auto row_permute_dense = static_cast<Mtx *>(row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_dense,
                        l({{-2.0, 2.0, 4.5},
                           {2.1, 3.4, 1.2},
                           {1.0, -1.0, -0.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsRowPermutable)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 0}};

    auto row_permute = this->mtx4->row_permute(&permute_idxs);
    auto row_permute_dense = static_cast<Mtx *>(row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_dense,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixIsColPermutable)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = this->mtx5->column_permute(&permute_idxs);
    auto c_permute_dense = static_cast<Mtx *>(c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_dense,
                        l({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsColPermutable)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = this->mtx4->column_permute(&permute_idxs);
    auto c_permute_dense = static_cast<Mtx *>(c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_dense,
                        l({{3.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0}}),
                        r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixIsInverseRowPermutable)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_row_permute =
        this->mtx5->inverse_row_permute(&inverse_permute_idxs);
    auto inverse_row_permute_dense =
        static_cast<Mtx *>(inverse_row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_dense,
                        l({{2.1, 3.4, 1.2},
                           {1.0, -1.0, -0.5},
                           {-2.0, 2.0, 4.5}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseRowPermutable)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 0}};

    auto inverse_row_permute =
        this->mtx4->inverse_row_permute(&inverse_permute_idxs);
    auto inverse_row_permute_dense =
        static_cast<Mtx *>(inverse_row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_dense,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, SquareMatrixIsInverseColPermutable)
{
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        this->mtx5->inverse_column_permute(&inverse_permute_idxs);
    auto inverse_c_permute_dense = static_cast<Mtx *>(inverse_c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_dense,
                        l({{-0.5, 1.0, -1.0},
                           {4.5, -2.0, 2.0},
                           {1.2, 2.1, 3.4}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseColPermutable)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        this->mtx4->inverse_column_permute(&inverse_permute_idxs);
    auto inverse_c_permute_dense = static_cast<Mtx *>(inverse_c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_dense,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0}}),
                        r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsRowPermutable64)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int64> permute_idxs{exec, {1, 0}};

    auto row_permute = this->mtx4->row_permute(&permute_idxs);
    auto row_permute_dense = static_cast<Mtx *>(row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_dense,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsColPermutable64)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = this->mtx4->column_permute(&permute_idxs);
    auto c_permute_dense = static_cast<Mtx *>(c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_dense,
                        l({{3.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0}}),
                        r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseRowPermutable64)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int64> inverse_permute_idxs{exec, {1, 0}};

    auto inverse_row_permute =
        this->mtx4->inverse_row_permute(&inverse_permute_idxs);
    auto inverse_row_permute_dense =
        static_cast<Mtx *>(inverse_row_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_dense,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}), r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseColPermutable64)
{
    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::Array<gko::int64> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        this->mtx4->inverse_column_permute(&inverse_permute_idxs);
    auto inverse_c_permute_dense = static_cast<Mtx *>(inverse_c_permute.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_dense,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0}}),
                        r<TypeParam>::value);
    // clang-format on
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrix)
{
    using T = typename TestFixture::value_type;

    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    auto diag = this->mtx5->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(Dense, ExtractsDiagonalFromNonSquareMatrix)
{
    using T = typename TestFixture::value_type;

    // clang-format off
    // {1.0, 3.0, 2.0},
    // {0.0, 5.0, 0.0}
    // clang-format on
    auto diag = this->mtx4->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, InplaceAbsolute)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    this->mtx5->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(this->mtx5,
                        l({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Dense, InplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    // mtx takes the left top corner 2 x 2 matrix
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(this->mtx5,
                        l({{1.0, 1.0, -0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Dense, OutplaceAbsolute)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto abs_mtx = this->mtx5->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx,
                        l({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Dense, OutplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on
    // mtx takes the left top corner 2 x 2 matrix
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, l({{1.0, 1.0}, {2.0, 2.0}}),
                        r<TypeParam>::value);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 2);
}


TYPED_TEST(Dense, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{14.0, 16.0}, complex_type{20.0, 22.0}},
           {complex_type{17.0, 19.0}, complex_type{24.5, 26.5}}}),
        0.0);
}


TYPED_TEST(Dense, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-12.0, -16.0}, complex_type{-16.0, -20.0}},
           {complex_type{-13.0, -15.0}, complex_type{-18.5, -20.5}}}),
        0.0);
}


TYPED_TEST(Dense, MakeComplex)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto complex_mtx = this->mtx5->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0);
}


TYPED_TEST(Dense, MakeComplexWithGivenResult)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto complex_mtx = ComplexMtx::create(exec, this->mtx5->get_size());
    this->mtx5->make_complex(complex_mtx.get());

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0);
}


TYPED_TEST(Dense, MakeComplexWithGivenResultFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto complex_mtx = ComplexMtx::create(exec);

    ASSERT_THROW(this->mtx5->make_complex(complex_mtx.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetReal)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto real_mtx = this->mtx5->get_real();

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0);
}


TYPED_TEST(Dense, GetRealWithGivenResult)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto real_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_real(real_mtx.get());

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0);
}


TYPED_TEST(Dense, GetRealWithGivenResultFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto real_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_real(real_mtx.get()), gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetImag)
{
    using T = typename TestFixture::value_type;
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto imag_mtx = this->mtx5->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}), 0);
}


TYPED_TEST(Dense, GetImagWithGivenResult)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto imag_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_imag(imag_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}), 0);
}


TYPED_TEST(Dense, GetImagWithGivenResultFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();
    // clang-format off
    // {1.0, -1.0, -0.5},
    // {-2.0, 2.0, 4.5},
    // {2.1, 3.4, 1.2}
    // clang-format on

    auto imag_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_imag(imag_mtx.get()), gko::DimensionMismatch);
}


template <typename T>
class DenseComplex : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using RealMtx = gko::matrix::Dense<gko::remove_complex<value_type>>;
};


TYPED_TEST_SUITE(DenseComplex, gko::test::ComplexValueTypes);


TYPED_TEST(DenseComplex, NonSquareMatrixIsConjugateTransposable)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);

    auto trans = mtx->conj_transpose();
    auto trans_as_dense = static_cast<Dense *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_dense,
                        l({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                           {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(DenseComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(DenseComplex, MakeComplex)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto complex_mtx = mtx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(DenseComplex, MakeComplexWithGivenResult)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto complex_mtx = Mtx::create(exec, mtx->get_size());
    mtx->make_complex(complex_mtx.get());

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(DenseComplex, GetReal)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto real_mtx = mtx->get_real();

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetRealWithGivenResult)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto real_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_real(real_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetImag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto imag_mtx = mtx->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetImagWithGivenResult)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto imag_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_imag(imag_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


}  // namespace
