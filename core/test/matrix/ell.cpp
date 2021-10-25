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

#include <ginkgo/core/matrix/ell.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Ell : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Ell<value_type, index_type>;

    index_type invalid_index = gko::invalid_index<index_type>();

    Ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Ell<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 3))
    {
        value_type* v = mtx->get_values();
        index_type* c = mtx->get_col_idxs();
        c[0] = 0;
        c[1] = 0;
        c[2] = 1;
        c[3] = 1;
        c[4] = 2;
        c[5] = invalid_index;
        v[0] = 1.0;
        v[1] = 0.0;
        v[2] = 3.0;
        v[3] = 5.0;
        v[4] = 2.0;
        v[5] = 0.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto n = m->get_num_stored_elements_per_row();
        auto p = m->get_stride();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 6);
        EXPECT_EQ(n, 3);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 0);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(c[4], 2);
        EXPECT_EQ(c[5], invalid_index);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{0.0});
        EXPECT_EQ(v[2], value_type{3.0});
        EXPECT_EQ(v[3], value_type{5.0});
        EXPECT_EQ(v[4], value_type{2.0});
        EXPECT_EQ(v[5], value_type{0.0});
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_num_stored_elements_per_row(), 0);
        ASSERT_EQ(m->get_stride(), 0);
    }
};

TYPED_TEST_SUITE(Ell, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ell, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(this->mtx->get_num_stored_elements_per_row(), 3);
    ASSERT_EQ(this->mtx->get_stride(), 2);
}


TYPED_TEST(Ell, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Ell, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Ell, CanBeCreatedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 3.0, 4.0, -1.0, 2.0, 0.0, 0.0, -1.0};
    index_type col_idxs[] = {0, 1, 0, -1, 1, 0, 0, -1};

    auto mtx = gko::matrix::Ell<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 8, values),
        gko::make_array_view(this->exec, 8, col_idxs), 2, 4);

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
}


TYPED_TEST(Ell, CanBeCreatedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const value_type values[] = {1.0, 3.0, 4.0, -1.0, 2.0, 0.0, 0.0, -1.0};
    const index_type col_idxs[] = {0, 1, 0, -1, 1, 0, 0, -1};

    auto mtx = gko::matrix::Ell<value_type, index_type>::create_const(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::const_view(this->exec, 8, values),
        gko::array<index_type>::const_view(this->exec, 8, col_idxs), 2, 4);

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
}


TYPED_TEST(Ell, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Ell, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Ell, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(static_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Ell, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Ell, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->read(
        {{2, 3},
         {{0, 0, 1.0}, {0, 1, 3.0}, {0, 2, 2.0}, {1, 0, 0.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Ell, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 5);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 0, value_type{0.0}));
    EXPECT_EQ(data.nonzeros[4], tpl(1, 1, value_type{5.0}));
}


TYPED_TEST(Ell, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 0, 0.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m.get());
}


class EllBlockDiagonal : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::Ell<value_type, index_type>;

    EllBlockDiagonal() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    std::vector<std::unique_ptr<Mtx>> generate_matrices()
    {
        std::vector<std::unique_ptr<Mtx>> matset;
        auto mat1 = Mtx::create(exec, gko::dim<2>(4, 4), 3, 5);
        value_type* v = mat1->get_values();
        index_type* c = mat1->get_col_idxs();
        // clang-format off
        c[0] = 0; c[5] = 2; c[10] = 2;
        c[1] = 1; c[6] = 2; c[11] = 3;
        c[2] = 0; c[7] = 2; c[12] = 2;
        c[3] = 2; c[8] = 3; c[13] = 3;
        // clang-format on
        for (int i = 0; i < 3 * 5; i++) {
            v[i] = i;
        }
        v[10] = 0.0;
        v[12] = 0.0;
        v[13] = 0.0;
        matset.emplace_back(std::move(mat1));

        auto mat2 = Mtx::create(exec, gko::dim<2>(2, 2), 1, 3);
        v = mat2->get_values();
        c = mat2->get_col_idxs();
        c[0] = 1;
        c[1] = 0;
        v[0] = 300.0;
        v[1] = -301.0;
        v[2] = -1234.4;
        matset.emplace_back(std::move(mat2));

        auto mat3 = Mtx::create(exec, gko::dim<2>(3, 3), 2, 3);
        v = mat3->get_values();
        c = mat3->get_col_idxs();
        c[0] = 1;
        c[3] = 1;
        c[1] = 0;
        c[4] = 1;
        c[2] = 1;
        c[5] = 2;
        for (int i = 0; i < 2 * 3; i++) {
            v[i] = 1.0 + 2 * i;
        }
        v[3] = 0.0;
        matset.emplace_back(std::move(mat3));
        return matset;
    }

    std::unique_ptr<Mtx> get_big_matrix()
    {
        auto mat = Mtx::create(exec, gko::dim<2>(9, 9), 3, 9);
        value_type* v = mat->get_values();
        index_type* c = mat->get_col_idxs();
        // clang-format off
        c[0] = 0; c[ 9] = 2; c[18] = 2;
        c[1] = 1; c[10] = 2; c[19] = 3;
        c[2] = 0; c[11] = 2; c[20] = 2;
        c[3] = 2; c[12] = 3; c[21] = 3;
        // clang-format on
        for (int i = 0; i < 4; i++) {
            v[i] = i;
        }
        for (int i = 9; i < 13; i++) {
            v[i] = i - 4;
        }
        for (int i = 18; i < 22; i++) {
            v[i] = i - 8;
        }
        // clang-format off
        v[18] = 0.0; v[20] = 0.0; v[21] = 0.0;
        c[4] = 5; c[13] = 5; c[22] = 5;
        c[5] = 4; c[14] = 4; c[23] = 4;
        v[4] = 300.0;  v[13] = 0.0; v[22] = 0.0;
        v[5] = -301.0; v[14] = 0.0; v[23] = 0.0;
        c[6] = 7; c[15] = 7; c[24] = 7;
        c[7] = 6; c[16] = 7; c[25] = 7;
        c[8] = 7; c[17] = 8; c[26] = 8;
        // clang-format on
        for (int i = 6; i < 9; i++) {
            v[i] = 1.0 + 2 * (i - 6);
        }
        for (int i = 15; i < 18; i++) {
            v[i] = 1.0 + 2 * (i - 12);
        }
        v[15] = 0.0;
        for (int i = 24; i < 27; i++) {
            v[i] = 0.0;
        }
        // clang-format off
		v[24] = 0.0; v[25] = 0.0; v[26] = 0.0;
        // clang-format on
        return mat;
    }
};


TEST_F(EllBlockDiagonal, GeneratesCorrectBlockDiagonalMatrix)
{
    auto matset = generate_matrices();

    auto bdcsr = gko::create_block_diagonal_matrix(exec, matset);
    auto check = get_big_matrix();
    const auto nnz_per_row = bdcsr->get_num_stored_elements_per_row();

    ASSERT_EQ(bdcsr->get_size(), check->get_size());
    ASSERT_EQ(bdcsr->get_stride(), check->get_stride());
    ASSERT_EQ(nnz_per_row, check->get_num_stored_elements_per_row());
    for (size_t j = 0; j < nnz_per_row; j++) {
        for (size_t irow = 0; irow < bdcsr->get_stride(); irow++) {
            const size_t inz = j * bdcsr->get_stride() + irow;
            ASSERT_EQ(bdcsr->get_const_col_idxs()[inz],
                      check->get_const_col_idxs()[inz]);
            ASSERT_EQ(bdcsr->get_const_values()[inz],
                      check->get_const_values()[inz]);
        }
    }
}

TEST_F(EllBlockDiagonal, ThrowsOnRectangularInput)
{
    std::vector<std::unique_ptr<Mtx>> matrices;
    auto mtx = Mtx::create(exec, gko::dim<2>(3, 6), 10);
    matrices.emplace_back(std::move(mtx));

    ASSERT_THROW(gko::create_block_diagonal_matrix(exec, matrices),
                 gko::DimensionMismatch);
}
