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

#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include <complex>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/batch_diagonal_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchDiagonal : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using Mtx = gko::matrix::BatchDiagonal<value_type>;
    using DiagMtx = gko::matrix::Diagonal<value_type>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using RealMtx = gko::remove_complex<Mtx>;

    BatchDiagonal()
        : exec(gko::ReferenceExecutor::create()),
          mtx_1(gko::batch_diagonal_initialize(
              {I<T>({1.0, -1.0, 2.2}), I<T>({-2.0, 2.0, -0.5})}, exec)),
          mtx_10(generate_diag_matrix(I<T>({1.0, -1.0, 2.2}))),
          mtx_11(generate_diag_matrix(I<T>({-2.0, 2.0, -0.5}))),
          mtx_4(gko::batch_diagonal_initialize(
              {I<T>({1.0, 1.5, 3.0, -2.0, 0.5}), I<T>({6.0, 1.0, 5.0, -2.0})},
              exec)),
          mtx_5(generate_batch_diag_matrix())
    {}

    std::unique_ptr<Mtx> generate_batch_diag_matrix()
    {
        const gko::dim<2> size{3, 5};
        const size_type num_entries = 2;
        const size_type stored = std::min(size[0], size[1]);
        gko::array<value_type> vals(exec, stored * num_entries);
        auto valarr = vals.get_data();
        valarr[0] = 2.5;
        valarr[1] = -1.5;
        valarr[2] = 3.0;
        valarr[3] = 1.5;
        valarr[4] = 5.5;
        valarr[5] = -2.0;
        const gko::batch_dim<2> bsize{num_entries, size};
        return Mtx::create(exec, bsize, std::move(vals));
    }

    std::unique_ptr<DiagMtx> generate_diag_matrix(
        std::initializer_list<value_type> vals)
    {
        const size_type num_entries = vals.size();
        auto mtx = DiagMtx::create(exec, num_entries);
        auto valarr = mtx->get_values();
        int idx = 0;
        for (auto x : vals) {
            valarr[idx] = x;
            idx++;
        }
        return mtx;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx_1;
    std::unique_ptr<DiagMtx> mtx_10;
    std::unique_ptr<DiagMtx> mtx_11;
    std::unique_ptr<Mtx> mtx_4;
    std::unique_ptr<Mtx> mtx_5;
};


TYPED_TEST_SUITE(BatchDiagonal, gko::test::ValueTypes);


TYPED_TEST(BatchDiagonal, SquareAppliesToDenseInPlace)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::BatchDense<T>;
    auto mtx(gko::batch_initialize<Dense>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));
    auto diag =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(3, 3)));
    diag->at(0, 0) = 1.0;
    diag->at(0, 1) = 2.0;
    diag->at(0, 2) = 3.0;
    diag->at(1, 0) = -1.0;
    diag->at(1, 1) = -2.0;
    diag->at(1, 2) = -3.0;

    gko::kernels::reference::batch_diagonal::apply_in_place(
        this->exec, diag.get(), mtx.get());

    EXPECT_EQ(mtx->at(0, 0, 0), T{1.0});
    EXPECT_EQ(mtx->at(0, 1, 0), T{4.0});
    EXPECT_EQ(mtx->at(0, 2, 0), T{6.0});
    EXPECT_EQ(mtx->at(0, 0, 1), T{0.0});
    EXPECT_EQ(mtx->at(0, 1, 1), T{6.0});
    EXPECT_EQ(mtx->at(0, 2, 1), T{12.0});

    EXPECT_EQ(mtx->at(1, 0, 0), T{4.0});
    EXPECT_EQ(mtx->at(1, 1, 0), T{6.0});
    EXPECT_EQ(mtx->at(1, 2, 0), T{0.0});
    EXPECT_EQ(mtx->at(1, 0, 1), T{-2.0});
    EXPECT_EQ(mtx->at(1, 1, 1), T{4.0});
    EXPECT_EQ(mtx->at(1, 2, 1), T{-3.0});
}


TYPED_TEST(BatchDiagonal, ThickAppliesToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::BatchDense<T>;
    auto mtx(gko::batch_initialize<Dense>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}, I<T>{20.0, -15.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}, I<T>{2.0, 1.5}}},
        this->exec));
    auto diag =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(3, 4)));
    diag->at(0, 0) = 1.0;
    diag->at(0, 1) = 2.0;
    diag->at(0, 2) = 3.0;
    diag->at(1, 0) = -1.0;
    diag->at(1, 1) = -2.0;
    diag->at(1, 2) = -3.0;
    auto rmtx =
        Dense::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(3, 2)));

    diag->apply(mtx.get(), rmtx.get());

    EXPECT_EQ(rmtx->at(0, 0, 0), T{1.0});
    EXPECT_EQ(rmtx->at(0, 1, 0), T{4.0});
    EXPECT_EQ(rmtx->at(0, 2, 0), T{6.0});
    EXPECT_EQ(rmtx->at(0, 0, 1), T{0.0});
    EXPECT_EQ(rmtx->at(0, 1, 1), T{6.0});
    EXPECT_EQ(rmtx->at(0, 2, 1), T{12.0});

    EXPECT_EQ(rmtx->at(1, 0, 0), T{4.0});
    EXPECT_EQ(rmtx->at(1, 1, 0), T{6.0});
    EXPECT_EQ(rmtx->at(1, 2, 0), T{0.0});
    EXPECT_EQ(rmtx->at(1, 0, 1), T{-2.0});
    EXPECT_EQ(rmtx->at(1, 1, 1), T{4.0});
    EXPECT_EQ(rmtx->at(1, 2, 1), T{-3.0});
}


TYPED_TEST(BatchDiagonal, SkinnyAppliesToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::BatchDense<T>;
    auto mtx(gko::batch_initialize<Dense>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}}},
        this->exec));
    auto diag =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(4, 3)));
    diag->at(0, 0) = 1.0;
    diag->at(0, 1) = 2.0;
    diag->at(0, 2) = 3.0;
    diag->at(1, 0) = -1.0;
    diag->at(1, 1) = -2.0;
    diag->at(1, 2) = -3.0;
    auto rmtx =
        Dense::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(4, 2)));

    diag->apply(mtx.get(), rmtx.get());

    EXPECT_EQ(rmtx->at(0, 0, 0), T{1.0});
    EXPECT_EQ(rmtx->at(0, 1, 0), T{4.0});
    EXPECT_EQ(rmtx->at(0, 2, 0), T{6.0});
    EXPECT_EQ(rmtx->at(0, 3, 0), T{0.0});
    EXPECT_EQ(rmtx->at(0, 0, 1), T{0.0});
    EXPECT_EQ(rmtx->at(0, 1, 1), T{6.0});
    EXPECT_EQ(rmtx->at(0, 2, 1), T{12.0});
    EXPECT_EQ(rmtx->at(0, 3, 1), T{0.0});

    EXPECT_EQ(rmtx->at(1, 0, 0), T{4.0});
    EXPECT_EQ(rmtx->at(1, 1, 0), T{6.0});
    EXPECT_EQ(rmtx->at(1, 2, 0), T{0.0});
    EXPECT_EQ(rmtx->at(1, 3, 0), T{0.0});
    EXPECT_EQ(rmtx->at(1, 0, 1), T{-2.0});
    EXPECT_EQ(rmtx->at(1, 1, 1), T{4.0});
    EXPECT_EQ(rmtx->at(1, 2, 1), T{-3.0});
    EXPECT_EQ(rmtx->at(1, 3, 1), T{0.0});
}


TYPED_TEST(BatchDiagonal, AdvancedAppliesToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Dense = gko::matrix::BatchDense<T>;
    auto x(gko::batch_initialize<Dense>(
        {{I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}, I<T>{0.12, 1.21}},
         {I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}, I<T>{1.21, 0.12}}},
        this->exec));
    auto b(gko::batch_initialize<Dense>(
        {{I<T>{-4.0, 2.0}, I<T>{-3.0, -2.0}, I<T>{0.0, 1.0}},
         {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}},
        this->exec));
    auto alphat = gko::batch_initialize<Dense>(
        {I<T>{-1.0, 0.5}, I<T>{2.0, 0.25}}, this->exec);
    auto betat = gko::batch_initialize<Dense>(
        {I<T>{2.0, -3.0}, I<T>{-0.5, 4.0}}, this->exec);
    auto alpha = gko::as<Dense>(alphat->transpose());
    auto beta = gko::as<Dense>(betat->transpose());
    auto diag =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>(4, 3)));
    diag->at(0, 0) = 1.0;
    diag->at(0, 1) = 2.0;
    diag->at(0, 2) = 3.0;
    diag->at(1, 0) = -1.0;
    diag->at(1, 1) = -2.0;
    diag->at(1, 2) = -3.0;

    diag->apply(alpha.get(), b.get(), beta.get(), x.get());

    EXPECT_EQ(x->at(0, 0, 0), T{6.0});
    EXPECT_EQ(x->at(0, 1, 0), T{10.0});
    EXPECT_EQ(x->at(0, 2, 0), T{4.0});
    EXPECT_EQ(x->at(0, 3, 0), T{0.24});
    EXPECT_EQ(x->at(0, 0, 1), T{1.0});
    EXPECT_EQ(x->at(0, 1, 1), T{-11.0});
    EXPECT_EQ(x->at(0, 2, 1), T{-10.5});
    EXPECT_EQ(x->at(0, 3, 1), T{-3.63});

    EXPECT_EQ(x->at(1, 0, 0), T{0.0});
    EXPECT_EQ(x->at(1, 1, 0), T{-6.5});
    EXPECT_EQ(x->at(1, 2, 0), T{-12.0});
    EXPECT_EQ(x->at(1, 3, 0), T{-0.605});
    EXPECT_EQ(x->at(1, 0, 1), T{8.0});
    EXPECT_EQ(x->at(1, 1, 1), T{-9.5});
    EXPECT_EQ(x->at(1, 2, 1), T{1.0});
    EXPECT_EQ(x->at(1, 3, 1), T{0.48});
}


TYPED_TEST(BatchDiagonal, ConvertsToPrecision)
{
    using BatchDiagonal = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDiagonal = typename gko::matrix::BatchDiagonal<OtherT>;
    auto othertmp = OtherBatchDiagonal::create(this->exec);
    auto res = BatchDiagonal::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->convert_to(othertmp.get());
    othertmp->convert_to(res.get());

    auto ures = res->unbatch();
    auto umtx = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchDiagonal, MovesToPrecision)
{
    using BatchDiagonal = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDiagonal = typename gko::matrix::BatchDiagonal<OtherT>;
    auto tmp = OtherBatchDiagonal::create(this->exec);
    auto res = BatchDiagonal::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx_1->move_to(tmp.get());
    tmp->move_to(res.get());

    auto ures = res->unbatch();
    auto umtx = this->mtx_1->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchDiagonal, ConvertsEmptyToPrecision)
{
    using BatchDiagonal = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDiagonal = typename gko::matrix::BatchDiagonal<OtherT>;
    auto empty = OtherBatchDiagonal::create(this->exec);
    auto res = BatchDiagonal::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_num_batch_entries());
}


TYPED_TEST(BatchDiagonal, MovesEmptyToPrecision)
{
    using BatchDiagonal = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherBatchDiagonal = typename gko::matrix::BatchDiagonal<OtherT>;
    auto empty = OtherBatchDiagonal::create(this->exec);
    auto res = BatchDiagonal::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_num_batch_entries());
}


TYPED_TEST(BatchDiagonal, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;

    auto trans = this->mtx_1->transpose();
    auto trans_as_batch_diag = static_cast<Mtx*>(trans.get());

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, this->mtx_1, 0.0);
}


TYPED_TEST(BatchDiagonal, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;

    auto trans = this->mtx_5->transpose();
    auto trans_as_batch_diag = static_cast<Mtx*>(trans.get());
    auto check =
        Mtx::create(this->exec, gko::transpose(this->mtx_5->get_size()));
    this->exec->copy(this->mtx_5->get_num_stored_elements(),
                     this->mtx_5->get_const_values(), check->get_values());

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, check, 0.0);
}


using namespace std::complex_literals;


template <typename T>
class ComplexBatchDiagonal : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;
    using Mtx = gko::matrix::BatchDiagonal<value_type>;

    ComplexBatchDiagonal()
        : exec(gko::ReferenceExecutor::create()),
          mtx_1(Mtx::create(exec, gko::batch_dim<2>(2, gko::dim<2>(3, 3)))),
          mtx_2(generate_batch_diag_matrix())
    {
        mtx_1->at(0, 0) = 1.0 - 1.0i;
        mtx_1->at(0, 1) = -1.0;
        mtx_1->at(0, 2) = 2.2 + 0.5i;
        mtx_1->at(1, 0) = -2.0;
        mtx_1->at(1, 1) = 2.0 - 3.5i;
        mtx_1->at(1, 2) = -0.5 + 2.2i;
    }

    std::unique_ptr<Mtx> generate_batch_diag_matrix()
    {
        const gko::dim<2> size{3, 5};
        const size_type num_entries = 2;
        const size_type stored = std::min(size[0], size[1]);
        gko::array<value_type> vals(exec, stored * num_entries);
        auto valarr = vals.get_data();
        valarr[0] = 2.5 + 1.0i;
        valarr[1] = -1.5 - 0.5i;
        valarr[2] = 3.0 - 6.0i;
        valarr[3] = 1.5 + 1.5i;
        valarr[4] = 5.5 - 2.5i;
        valarr[5] = -2.0 + 4.0i;
        const gko::batch_dim<2> bsize{num_entries, size};
        return Mtx::create(exec, bsize, std::move(vals));
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx_1;
    std::unique_ptr<Mtx> mtx_2;
};

TYPED_TEST_SUITE(ComplexBatchDiagonal, gko::test::ComplexValueTypes);


TYPED_TEST(ComplexBatchDiagonal, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;

    auto trans = this->mtx_1->transpose();
    auto trans_as_batch_diag = static_cast<Mtx*>(trans.get());

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, this->mtx_1, 0.0);
}


TYPED_TEST(ComplexBatchDiagonal, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;

    auto trans = this->mtx_2->transpose();
    auto trans_as_batch_diag = static_cast<Mtx*>(trans.get());
    auto check =
        Mtx::create(this->exec, gko::transpose(this->mtx_2->get_size()));
    this->exec->copy(this->mtx_2->get_num_stored_elements(),
                     this->mtx_2->get_const_values(), check->get_values());

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, check, 0.0);
}


TYPED_TEST(ComplexBatchDiagonal, SquareMatrixIsConjTransposable)
{
    using Mtx = typename TestFixture::Mtx;

    auto trans = this->mtx_1->conj_transpose();
    auto trans_as_batch_diag = static_cast<Mtx*>(trans.get());
    auto check =
        Mtx::create(this->exec, gko::transpose(this->mtx_1->get_size()));
    for (size_t i = 0; i < check->get_num_stored_elements(); i++) {
        check->get_values()[i] = gko::conj(this->mtx_1->get_const_values()[i]);
    }

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, check, 0.0);
}


TYPED_TEST(ComplexBatchDiagonal, NonSquareMatrixIsConjTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto check =
        Mtx::create(this->exec, gko::transpose(this->mtx_2->get_size()));
    auto valarr = check->get_values();
    valarr[0] = 2.5 - 1.0i;
    valarr[1] = -1.5 + 0.5i;
    valarr[2] = 3.0 + 6.0i;
    valarr[3] = 1.5 - 1.5i;
    valarr[4] = 5.5 + 2.5i;
    valarr[5] = -2.0 - 4.0i;

    auto trans = this->mtx_2->conj_transpose();
    auto trans_as_batch_diag = gko::as<Mtx>(trans.get());

    GKO_ASSERT_BATCH_MTX_NEAR(trans_as_batch_diag, check, 0.0);
}


}  // namespace
