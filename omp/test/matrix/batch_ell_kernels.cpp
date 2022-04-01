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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_ell_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


class BatchEll : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::BatchEll<value_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::BatchDense<value_type>;

    BatchEll()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::OmpExecutor::create()),
          mtx_size(10, gko::dim<2>(12, 7)),
          rand_engine(42)
    {
        mtx = Mtx::create(exec, gko::batch_dim<2>{2, gko::dim<2>{2, 3}},
                          gko::batch_stride{2, 2});
        mtx2 = Mtx::create(exec, gko::batch_dim<2>{2, gko::dim<2>{3, 3}},
                           gko::batch_stride{2, 2});
        create_mtx(mtx.get());
        create_mtx2(mtx2.get());
    }

    void SetUp() {}

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    void create_mtx(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        /*
         * 1   0   2
         * 0   5   0
         *
         * 2   0   1
         * 0   8   0
         */
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 2.0;
        v[3] = 0.0;
        v[4] = 2.0;
        v[5] = 8.0;
        v[6] = 1.0;
        v[7] = 0.0;
    }

    void create_mtx2(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        // It keeps an explict zero
        /*
         *  1    0   2
         *  0    5   4
         *  0    8   0
         *
         *  3    0   9
         *  0    7   6
         *  0   10   0
         */
        c[0] = 0;
        c[1] = 1;
        c[2] = 1;
        c[3] = 2;
        c[4] = 1;
        c[5] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 8.0;
        v[3] = 2.0;
        v[4] = 4.0;
        v[5] = 0.0;
        v[6] = 3.0;
        v[7] = 7.0;
        v[8] = 10.0;
        v[9] = 9.0;
        v[10] = 6.0;
        v[11] = 0.0;
    }

    void assert_equal_batch_ell_matrices(const Mtx* mat1, const Mtx* mat2)
    {
        ASSERT_EQ(mat1->get_num_batch_entries(), mat2->get_num_batch_entries());
        ASSERT_EQ(mat1->get_num_stored_elements(),
                  mat2->get_num_stored_elements());
        ASSERT_EQ(mat1->get_stride(), mat2->get_stride());
        ASSERT_EQ(mat1->get_size(), mat2->get_size());
        for (auto i = 0; i < mat1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(mat1->get_const_values()[i], mat2->get_const_values()[i]);
        }
        for (auto i = 0; i < mat1->get_num_stored_elements() /
                                 mat1->get_num_batch_entries();
             ++i) {
            EXPECT_EQ(mat1->get_const_col_idxs()[i],
                      mat2->get_const_col_idxs()[i]);
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(size_t batch_size, int num_rows,
                                     int num_cols, int min_nnz_row)
    {
        using real_type = typename gko::remove_complex<value_type>;
        return gko::test::generate_uniform_batch_random_matrix<MtxType>(
            batch_size, num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            ref);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> exec;
    std::ranlux48 rand_engine;
    gko::batch_dim<2> mtx_size;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
};


TEST_F(BatchEll, AppliesToDenseVector)
{
    using T = value_type;
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {1.0, -1.0, 3.0}},
                                        this->exec);
    auto y =
        Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{2, 1}, gko::dim<2>{2, 1}}));

    this->mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), T{10.0});
    EXPECT_EQ(y->at(0, 1), T{5.0});
    EXPECT_EQ(y->at(1, 0), T{5.0});
    EXPECT_EQ(y->at(1, 1), T{-8.0});
}


TEST_F(BatchEll, AppliesToDenseMatrix)
{
    using T = value_type;
    auto x = gko::batch_initialize<Vec>(
        {{I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}},
         {I<T>{1.0, 3.0}, I<T>{-1.0, -1.5}, I<T>{3.0, 2.5}}},
        this->exec);
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    this->mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0, 0), T{10.0});
    EXPECT_EQ(y->at(0, 1, 0), T{5.0});
    EXPECT_EQ(y->at(0, 0, 1), T{8.0});
    EXPECT_EQ(y->at(0, 1, 1), T{-7.5});
    EXPECT_EQ(y->at(1, 0, 0), T{5.0});
    EXPECT_EQ(y->at(1, 1, 0), T{-8.0});
    EXPECT_EQ(y->at(1, 0, 1), T{8.5});
    EXPECT_EQ(y->at(1, 1, 1), T{-12.0});
}


TEST_F(BatchEll, AppliesLinearCombinationToDenseVector)
{
    using T = value_type;
    auto alpha = gko::batch_initialize<Vec>({{-1.0}, {1.0}}, this->exec);
    auto beta = gko::batch_initialize<Vec>({{2.0}, {2.0}}, this->exec);
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {-2.0, 1.0, 4.0}},
                                        this->exec);
    auto y = gko::batch_initialize<Vec>({{1.0, 2.0}, {1.0, -2.0}}, this->exec);
    auto umats = this->mtx->unbatch();
    auto umtx0 = umats[0].get();
    auto umtx1 = umats[1].get();
    auto ualphas = alpha->unbatch();
    auto ualpha0 = ualphas[0].get();
    auto ualpha1 = ualphas[1].get();
    auto ubetas = beta->unbatch();
    auto ubeta0 = ubetas[0].get();
    auto ubeta1 = ubetas[1].get();
    auto uxs = x->unbatch();
    auto ux0 = uxs[0].get();
    auto ux1 = uxs[1].get();
    auto uys = y->unbatch();
    auto uy0 = uys[0].get();
    auto uy1 = uys[1].get();

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());
    umtx0->apply(ualpha0, ux0, ubeta0, uy0);
    umtx1->apply(ualpha1, ux1, ubeta1, uy1);

    EXPECT_EQ(y->at(0, 0), uy0->at(0));
    EXPECT_EQ(y->at(0, 1), uy0->at(1));
    EXPECT_EQ(y->at(1, 0), uy1->at(0));
    EXPECT_EQ(y->at(1, 1), uy1->at(1));
}


TEST_F(BatchEll, AppliesLinearCombinationToDenseMatrix)
{
    using T = value_type;
    auto alpha = gko::batch_initialize<Vec>({{1.0}, {-1.0}}, this->exec);
    auto beta = gko::batch_initialize<Vec>({{2.0}, {-2.0}}, this->exec);
    auto x = gko::batch_initialize<Vec>(
        {{I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}},
         {I<T>{2.0, 2.0}, I<T>{-1.0, -1.5}, I<T>{4.0, 2.5}}},
        this->exec);
    auto y = gko::batch_initialize<Vec>(
        {{I<T>{1.0, 0.5}, I<T>{2.0, -1.5}}, {I<T>{2.0, 1.5}, I<T>{2.0, 1.5}}},
        this->exec);

    auto umats = this->mtx->unbatch();
    auto umtx0 = umats[0].get();
    auto umtx1 = umats[1].get();
    auto ualphas = alpha->unbatch();
    auto ualpha0 = ualphas[0].get();
    auto ualpha1 = ualphas[1].get();
    auto ubetas = beta->unbatch();
    auto ubeta0 = ubetas[0].get();
    auto ubeta1 = ubetas[1].get();
    auto uxs = x->unbatch();
    auto ux0 = uxs[0].get();
    auto ux1 = uxs[1].get();
    auto uys = y->unbatch();
    auto uy0 = uys[0].get();
    auto uy1 = uys[1].get();

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());
    umtx0->apply(ualpha0, ux0, ubeta0, uy0);
    umtx1->apply(ualpha1, ux1, ubeta1, uy1);

    EXPECT_EQ(y->at(0, 0, 0), uy0->at(0, 0));
    EXPECT_EQ(y->at(0, 1, 0), uy0->at(1, 0));
    EXPECT_EQ(y->at(0, 0, 1), uy0->at(0, 1));
    EXPECT_EQ(y->at(0, 1, 1), uy0->at(1, 1));
    EXPECT_EQ(y->at(1, 0, 0), uy1->at(0, 0));
    EXPECT_EQ(y->at(1, 1, 0), uy1->at(1, 0));
    EXPECT_EQ(y->at(1, 0, 1), uy1->at(0, 1));
    EXPECT_EQ(y->at(1, 1, 1), uy1->at(1, 1));
}


TEST_F(BatchEll, DetectsMissingDiagonalEntry)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gen_mtx<Mtx>(batch_size, nrows, ncols, nrows / 10);
    gko::test::remove_diagonal_from_row(mtx.get(), nrows / 2);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());
    bool all_diags = true;

    gko::kernels::omp::batch_ell::check_diagonal_entries_exist(exec, omtx.get(),
                                                               all_diags);

    ASSERT_FALSE(all_diags);
}


TEST_F(BatchEll, DetectsPresenceOfAllDiagonalEntries)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());
    bool all_diags = false;

    gko::kernels::omp::batch_ell::check_diagonal_entries_exist(exec, omtx.get(),
                                                               all_diags);

    ASSERT_TRUE(all_diags);
}


TEST_F(BatchEll, AddScaleIdentityIsEquivalentToReference)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto alpha = gko::batch_initialize<Vec>(batch_size, {2.0}, ref);
    auto beta = gko::batch_initialize<Vec>(batch_size, {-1.0}, ref);
    auto dalpha = alpha->clone(exec);
    auto dbeta = beta->clone(exec);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());

    mtx->add_scaled_identity(alpha.get(), beta.get());
    omtx->add_scaled_identity(dalpha.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mtx, omtx, r<value_type>::value);
}


}  // namespace
