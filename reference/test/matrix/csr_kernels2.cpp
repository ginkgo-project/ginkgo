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

#include <ginkgo/core/matrix/csr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Csr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Sellp = gko::matrix::Sellp<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Hybrid = gko::matrix::Hybrid<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4,
                          std::make_shared<typename Mtx::load_balance>(2))),
          mtx2(Mtx::create(exec, gko::dim<2>{2, 3}, 5,
                           std::make_shared<typename Mtx::classical>())),
          mtx3_sorted(Mtx::create(exec, gko::dim<2>(3, 3), 7,
                                  std::make_shared<typename Mtx::classical>())),
          mtx3_unsorted(
              Mtx::create(exec, gko::dim<2>(3, 3), 7,
                          std::make_shared<typename Mtx::classical>()))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
        this->create_mtx3(mtx3_sorted.get(), mtx3_unsorted.get());
    }

    void create_mtx(Mtx *m)
    {
        value_type *v = m->get_values();
        index_type *c = m->get_col_idxs();
        index_type *r = m->get_row_ptrs();
        auto *s = m->get_srow();
        /*
         * 1   3   2
         * 0   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
        s[0] = 0;
    }

    void create_mtx2(Mtx *m)
    {
        value_type *v = m->get_values();
        index_type *c = m->get_col_idxs();
        index_type *r = m->get_row_ptrs();
        // It keeps an explict zero
        /*
         *  1    3   2
         * {0}   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 5;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        c[4] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 0.0;
        v[4] = 5.0;
    }

    void create_mtx3(Mtx *sorted, Mtx *unsorted)
    {
        auto vals_s = sorted->get_values();
        auto cols_s = sorted->get_col_idxs();
        auto rows_s = sorted->get_row_ptrs();
        auto vals_u = unsorted->get_values();
        auto cols_u = unsorted->get_col_idxs();
        auto rows_u = unsorted->get_row_ptrs();
        /* For both versions (sorted and unsorted), this matrix is stored:
         * 0  2  1
         * 3  1  8
         * 2  0  3
         * The unsorted matrix will have the (value, column) pair per row not
         * sorted, which we still consider a valid CSR format.
         */
        rows_s[0] = 0;
        rows_s[1] = 2;
        rows_s[2] = 5;
        rows_s[3] = 7;
        rows_u[0] = 0;
        rows_u[1] = 2;
        rows_u[2] = 5;
        rows_u[3] = 7;

        vals_s[0] = 2.;
        vals_s[1] = 1.;
        vals_s[2] = 3.;
        vals_s[3] = 1.;
        vals_s[4] = 8.;
        vals_s[5] = 2.;
        vals_s[6] = 3.;
        // Each row is stored rotated once to the left
        vals_u[0] = 1.;
        vals_u[1] = 2.;
        vals_u[2] = 1.;
        vals_u[3] = 8.;
        vals_u[4] = 3.;
        vals_u[5] = 3.;
        vals_u[6] = 2.;

        cols_s[0] = 1;
        cols_s[1] = 2;
        cols_s[2] = 0;
        cols_s[3] = 1;
        cols_s[4] = 2;
        cols_s[5] = 0;
        cols_s[6] = 2;
        // The same applies for the columns
        cols_u[0] = 2;
        cols_u[1] = 1;
        cols_u[2] = 1;
        cols_u[3] = 2;
        cols_u[4] = 0;
        cols_u[5] = 2;
        cols_u[6] = 0;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3_sorted;
    std::unique_ptr<Mtx> mtx3_unsorted;
};

TYPED_TEST_SUITE(Csr, gko::test::ValueIndexTypes);


TYPED_TEST(Csr, CanComputeBlockApprox)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx2->get_executor();

    auto mat = gko::initialize<Mtx>({{1.0, 2.0, 0.0, 0.0, 3.0},
                                     {0.0, 3.0, 0.0, 0.0, 0.0},
                                     {0.0, 3.0, 2.5, 1.5, 0.0},
                                     {1.0, 0.0, 1.0, 2.0, 4.0},
                                     {0.0, 1.0, 2.0, 1.5, 3.0}},
                                    exec);
    auto b_sizes = gko::Array<gko::size_type>(exec, {3, 2});
    auto block_mtxs = mat->get_block_approx(b_sizes);

    auto mat1 = gko::initialize<Mtx>(
        {{1.0, 2.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 3.0, 2.5}}, exec);
    auto mat2 =
        gko::initialize<Mtx>({I<T>({2.0, 4.0}), I<T>({1.5, 3.0})}, exec);

    GKO_EXPECT_MTX_NEAR(block_mtxs[0]->get_sub_matrix(), mat1,
                        r<ValueType>::value);
    GKO_EXPECT_MTX_NEAR(block_mtxs[1]->get_sub_matrix(), mat2,
                        r<ValueType>::value);
}


TYPED_TEST(Csr, CanComputeBlockApproxWithOverlap)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx2->get_executor();

    auto mat = gko::initialize<Mtx>({{1.0, 2.0, 0.0, 0.0, 3.0},
                                     {0.0, 3.0, 0.0, 0.0, 0.0},
                                     {0.0, 3.0, 2.5, 1.5, 0.0},
                                     {1.0, 0.0, 1.0, 2.0, 4.0},
                                     {0.0, 1.0, 2.0, 1.5, 3.0}},
                                    exec);
    auto b_sizes = gko::Array<gko::size_type>(exec, {3, 2});
    auto ov = gko::Overlap<gko::size_type>(exec, 2, 1);
    auto block_mtxs = mat->get_block_approx(b_sizes, ov);

    auto mat1 = gko::initialize<Mtx>({{1.0, 2.0, 0.0, 0.0},
                                      {0.0, 3.0, 0.0, 0.0},
                                      {0.0, 3.0, 2.5, 1.5},
                                      {1.0, 0.0, 1.0, 2.0}},
                                     exec);
    auto mat2 = gko::initialize<Mtx>(
        {I<T>({2.5, 1.5, 0.0}), I<T>({1.0, 2.0, 4.0}), I<T>({2.0, 1.5, 3.0})},
        exec);

    GKO_EXPECT_MTX_NEAR(block_mtxs[0]->get_sub_matrix(), mat1,
                        r<ValueType>::value);
    GKO_EXPECT_MTX_NEAR(block_mtxs[1]->get_sub_matrix(), mat2,
                        r<ValueType>::value);
}


TYPED_TEST(Csr, RestrictAppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});
    y->at(0) = T{0.0};
    y->at(1) = T{2489.0};

    auto wmask = gko::OverlapMask{gko::span(0, 1), true};
    this->mtx->apply(x.get(), y.get(), wmask);

    EXPECT_EQ(y->at(0), T{13.0});
    EXPECT_EQ(y->at(1), T{2489.0});
}


TYPED_TEST(Csr, RestrictAppliesToDenseVector2)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>({{2, -1.0, 0.0, 3.0, 0.0, 1.5, -1.0},
                                     {0.0, 0.0, -2.0, -1.0, 2, -1.0, 0.0},
                                     {0.0, 2.0, 2.0, 4.0, 2.0, 1.0, -0.5},
                                     {1.0, 0.0, -2.5, 3.0, -2.5, -1.0, 0.5},
                                     {0.0, 2.0, 2.0, -1.0, 2.5, 1.0, 0.5},
                                     {0.5, -1.5, 1.5, 0.0, -4.0, 0.0, 1.0}},
                                    this->exec);
    auto smat = gko::initialize<Mtx>({{1.0, 0.0, -2.5, 3.0, -2.5, -1.0, 0.5},
                                      {0.0, 2.0, 2.0, -1.0, 2.5, 1.0, 0.5}},
                                     this->exec);
    auto x =
        gko::initialize<Vec>({2.0, 1.0, 4.0, 2.5, -1.5, 0.0, 1.5}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{6, 1});
    y->at(0) = T{0.5};
    y->at(1) = T{0.5};
    y->at(2) = T{1.0};
    y->at(3) = T{0.1};
    y->at(4) = T{0.9};
    y->at(5) = T{1.5};

    auto wmask = gko::OverlapMask{gko::span(3, 5), true};
    auto y2 =
        Vec::create(this->exec, gko::dim<2>{wmask.write_idxs.length(), 1});

    mat->apply(x.get(), y.get(), wmask);
    smat->apply(x.get(), y2.get());

    EXPECT_EQ(y->at(0), T{0.5});
    EXPECT_EQ(y->at(1), T{0.5});
    EXPECT_EQ(y->at(2), T{1.0});
    EXPECT_EQ(y->at(3), y2->at(0));
    EXPECT_EQ(y->at(4), y2->at(1));
    EXPECT_EQ(y->at(5), T{1.5});
}


TYPED_TEST(Csr, RestrictAdvancedAppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>({{2, -1.0, 0.0, 3.0, 0.0, 1.5, -1.0},
                                     {0.0, 0.0, -2.0, -1.0, 2, -1.0, 0.0},
                                     {0.0, 2.0, 2.0, 4.0, 2.0, 1.0, -0.5},
                                     {1.0, 0.0, -2.5, 3.0, -2.5, -1.0, 0.5},
                                     {0.0, 2.0, 2.0, -1.0, 2.5, 1.0, 0.5},
                                     {0.5, -1.5, 1.5, 0.0, -4.0, 0.0, 1.0}},
                                    this->exec);
    auto smat = gko::initialize<Mtx>({{1.0, 0.0, -2.5, 3.0, -2.5, -1.0, 0.5},
                                      {0.0, 2.0, 2.0, -1.0, 2.5, 1.0, 0.5}},
                                     this->exec);
    auto x =
        gko::initialize<Vec>({2.0, 1.0, 4.0, 2.5, -1.5, 0.0, 1.5}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{6, 1});
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    y->at(0) = T{0.5};
    y->at(1) = T{0.5};
    y->at(2) = T{1.0};
    y->at(3) = T{0.1};
    y->at(4) = T{0.9};
    y->at(5) = T{1.5};

    auto wmask = gko::OverlapMask{gko::span(3, 5), true};
    auto y2 =
        Vec::create(this->exec, gko::dim<2>{wmask.write_idxs.length(), 1});
    y2->at(0) = T{0.1};
    y2->at(1) = T{0.9};

    mat->apply(alpha.get(), x.get(), beta.get(), y.get(), wmask);
    smat->apply(alpha.get(), x.get(), beta.get(), y2.get());

    EXPECT_EQ(y->at(0), T{0.5});
    EXPECT_EQ(y->at(1), T{0.5});
    EXPECT_EQ(y->at(2), T{1.0});
    EXPECT_EQ(y->at(3), y2->at(0));
    EXPECT_EQ(y->at(4), y2->at(1));
    EXPECT_EQ(y->at(5), T{1.5});
}


TYPED_TEST(Csr, CanGetSubmatrix)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    /* this->mtx
     * 1   3   2
     * 0   5   0
     */
    auto sub_mat = this->mtx->get_submatrix(gko::span(0, 2), gko::span(0, 2));
    auto ref =
        gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{0.0, 5.0}}, this->exec);

    GKO_ASSERT_MTX_NEAR(sub_mat.get(), ref.get(), 0.0);
}


TYPED_TEST(Csr, CanGetSubmatrix2)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>(
        {
            // clang-format off
            I<T>{1.0, 3.0, 4.5, 0.0, 2.0}, // 0
            I<T>{1.0, 0.0, 4.5, 7.5, 3.0}, // 1
            I<T>{0.0, 3.0, 4.5, 0.0, 2.0}, // 2
            I<T>{0.0,-1.0, 2.5, 0.0, 2.0}, // 3
            I<T>{1.0, 0.0,-1.0, 3.5, 1.0}, // 4
            I<T>{0.0, 1.0, 0.0, 0.0, 2.0}, // 5
            I<T>{0.0, 3.0, 0.0, 7.5, 1.0}  // 6
                                           // clang-format on
        },
        this->exec);
    ASSERT_EQ(mat->get_num_stored_elements(), 23);
    {
        auto sub_mat1 = mat->get_submatrix(gko::span(0, 2), gko::span(0, 2));
        auto ref1 =
            gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{1.0, 0.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1.get(), ref1.get(), 0.0);
    }
    {
        auto sub_mat2 = mat->get_submatrix(gko::span(2, 4), gko::span(0, 2));
        auto ref2 =
            gko::initialize<Mtx>({I<T>{0.0, 3.0}, I<T>{0.0, -1.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat2.get(), ref2.get(), 0.0);
    }
    {
        auto sub_mat3 = mat->get_submatrix(gko::span(0, 2), gko::span(3, 5));
        auto ref3 =
            gko::initialize<Mtx>({I<T>{0.0, 2.0}, I<T>{7.5, 3.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat3.get(), ref3.get(), 0.0);
    }
    {
        auto sub_mat4 = mat->get_submatrix(gko::span(1, 6), gko::span(2, 4));
        /*
           4.5, 7.5
           4.5, 0.0
           2.5, 0.0
           1.0, 3.5
           0.0, 0.0
        */
        auto ref4 = gko::initialize<Mtx>(
            {I<T>{4.5, 7.5}, I<T>{4.5, 0.0}, I<T>{2.5, 0.0}, I<T>{-1.0, 3.5},
             I<T>{0.0, 0.0}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat4.get(), ref4.get(), 0.0);
    }
}


TYPED_TEST(Csr, CanGetSubmatrixWithOverlap)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>(
        {
            // clang-format off
            I<T>{1.0, 3.0, 4.5, 0.0, 2.0}, // 0
            I<T>{1.0, 0.0, 4.5, 7.5, 3.0}, // 1
            I<T>{0.0, 3.0, 4.5, 0.0, 2.0}, // 2
            I<T>{0.0,-1.0, 2.5, 0.0, 2.0}, // 3
            I<T>{1.0, 0.0,-1.0, 3.5, 1.0}, // 4
            I<T>{0.0, 1.0, 0.0, 0.0, 2.0}, // 5
            I<T>{0.0, 3.0, 0.0, 7.5, 1.0}  // 6
                                           // clang-format on
        },
        this->exec);
    ASSERT_EQ(mat->get_num_stored_elements(), 23);
    {
        auto sub_mat1 = mat->get_submatrix(gko::span(0, 2), gko::span(0, 2), {},
                                           {gko::span{2, 3}});
        auto ref1 = gko::initialize<Mtx>(
            {I<T>{1.0, 3.0, 4.5}, I<T>{1.0, 0.0, 4.5}, I<T>{0.0, 3.0, 4.5}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1.get(), ref1.get(), 0.0);
    }
    {
        auto sub_mat2 = mat->get_submatrix(gko::span(5, 7), gko::span(3, 5),
                                           {gko::span(1, 3)}, {});
        auto sub_mat21 =
            mat->get_submatrix(gko::span(5, 7), gko::span(3, 5),
                               {gko::span(1, 2), gko::span(2, 3)}, {});
        auto ref2 = gko::initialize<Mtx>(
            {I<T>{-1.0, 2.5, 0.0, 2.0}, I<T>{0.0, -1.0, 3.5, 1.0},
             I<T>{1.0, 0.0, 0.0, 2.0}, I<T>{3.0, 0.0, 7.5, 1.0}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat2.get(), ref2.get(), 0.0);
        GKO_EXPECT_MTX_NEAR(sub_mat21.get(), ref2.get(), 0.0);
    }
    {
        auto sub_mat3 =
            mat->get_submatrix(gko::span(2, 4), gko::span(1, 3),
                               {gko::span(0, 1)}, {gko::span(3, 5)});
        auto ref3 = gko::initialize<Mtx>(
            {
                I<T>{1.0, 0.0, 4.5, 7.5, 3.0},   // 1
                I<T>{0.0, 3.0, 4.5, 0.0, 2.0},   // 2
                I<T>{0.0, -1.0, 2.5, 0.0, 2.0},  // 3
                I<T>{1.0, 0.0, -1.0, 3.5, 1.0},  // 4
                I<T>{0.0, 1.0, 0.0, 0.0, 2.0}    // 5
            },
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat3.get(), ref3.get(), 0.0);
    }
}


}  // namespace
