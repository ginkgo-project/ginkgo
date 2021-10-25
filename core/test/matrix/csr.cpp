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

#include <ginkgo/core/matrix/csr.hpp>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Csr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Csr<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, 4,
              std::make_shared<typename Mtx::load_balance>(2)))
    {
        value_type* v = mtx->get_values();
        index_type* c = mtx->get_col_idxs();
        index_type* r = mtx->get_row_ptrs();
        index_type* s = mtx->get_srow();
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

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx* m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        auto s = m->get_const_srow();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
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
        EXPECT_EQ(s[0], 0);
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
        ASSERT_EQ(m->get_const_srow(), nullptr);
    }
};

TYPED_TEST_SUITE(Csr, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Csr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 4);
}


TYPED_TEST(Csr, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Csr, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Csr, CanBeCreatedFromExistingData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type values[] = {1.0, 2.0, 3.0, 4.0};
    index_type col_idxs[] = {0, 1, 1, 0};
    index_type row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::Csr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::make_array_view(this->exec, 4, values),
        gko::make_array_view(this->exec, 4, col_idxs),
        gko::make_array_view(this->exec, 4, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));

    ASSERT_EQ(mtx->get_num_srow_elements(), 1);
    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_const_srow()[0], 0);
}


TYPED_TEST(Csr, CanBeCreatedFromExistingConstData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const value_type values[] = {1.0, 2.0, 3.0, 4.0};
    const index_type col_idxs[] = {0, 1, 1, 0};
    const index_type row_ptrs[] = {0, 2, 3, 4};

    auto mtx = gko::matrix::Csr<value_type, index_type>::create_const(
        this->exec, gko::dim<2>{3, 2},
        gko::array<value_type>::const_view(this->exec, 4, values),
        gko::array<index_type>::const_view(this->exec, 4, col_idxs),
        gko::array<index_type>::const_view(this->exec, 4, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));

    ASSERT_EQ(mtx->get_num_srow_elements(), 1);
    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
    ASSERT_EQ(mtx->get_const_srow()[0], 0);
}


TYPED_TEST(Csr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Csr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Csr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Csr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Csr, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::load_balance>(2));

    m->read({{2, 3}, {{0, 0, 1.0}, {0, 1, 3.0}, {0, 2, 2.0}, {1, 1, 5.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Csr, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec,
                         std::make_shared<typename Mtx::load_balance>(2));
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Csr, GeneratesCorrectMatrixData)
{
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


class CsrBlockDiagonal : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    CsrBlockDiagonal() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    std::vector<std::unique_ptr<Mtx>> generate_csr_matrices()
    {
        std::vector<std::unique_ptr<Mtx>> matset;
        auto mat1 = Mtx::create(exec, gko::dim<2>(4, 4), 8);
        value_type* v = mat1->get_values();
        index_type* r = mat1->get_row_ptrs();
        index_type* c = mat1->get_col_idxs();
        // clang-format off
        r[0] = 0; r[1] = 2; r[2] = 5; r[3] = 7; r[4] = 8;
        c[0] = 0; c[1] = 2;
        c[2] = 1; c[3] = 2; c[4] = 3;
        c[5] = 1; c[6] = 3;
        c[7] = 3;
        // clang-format on
        for (int i = 0; i < 8; i++) {
            v[i] = i;
        }
        matset.emplace_back(std::move(mat1));
        auto mat2 = Mtx::create(exec, gko::dim<2>(3, 3), 6);
        v = mat2->get_values();
        r = mat2->get_row_ptrs();
        c = mat2->get_col_idxs();
        // clang-format off
        r[0] = 0; r[1] = 2; r[2] = 4; r[3] = 6;
        c[0] = 0; c[1] = 2;
        c[2] = 1; c[3] = 2;
        c[4] = 0; c[5] = 1;
        // clang-format on
        for (int i = 0; i < 6; i++) {
            v[i] = 3 * i;
        }
        matset.emplace_back(std::move(mat2));
        auto mat3 = Mtx::create(exec, gko::dim<2>(2, 2), 4);
        v = mat3->get_values();
        r = mat3->get_row_ptrs();
        c = mat3->get_col_idxs();
        // clang-format off
        r[0] = 0; r[1] = 2; r[2] = 4;
        c[0] = 0; c[1] = 1;
        c[2] = 0; c[3] = 1;
        // clang-format on
        for (int i = 0; i < 4; i++) {
            v[i] = 1 + 0.5 * i;
        }
        matset.emplace_back(std::move(mat3));
        return matset;
    }

    std::unique_ptr<Mtx> get_big_matrix()
    {
        auto mat = Mtx::create(exec, gko::dim<2>(9, 9), 18);
        value_type* v = mat->get_values();
        index_type* r = mat->get_row_ptrs();
        index_type* c = mat->get_col_idxs();
        r[0] = 0;
        r[1] = 2;
        r[2] = 5;
        r[3] = 7;
        r[4] = 8;
        r[5] = 10;
        r[6] = 12;
        r[7] = 14;
        r[8] = 16;
        r[9] = 18;

        // clang-format off
        c[0] = 0; c[1] = 2;
        c[2] = 1; c[3] = 2; c[4] = 3;
        c[5] = 1; c[6] = 3;
        c[7] = 3;
        c[8] = 4; c[9] = 6;
        c[10] = 5; c[11] = 6;
        c[12] = 4; c[13] = 5;
        c[14] = 7; c[15] = 8;
        c[16] = 7; c[17] = 8;
        // clang-format on
        for (int i = 0; i < 8; i++) {
            v[i] = i;
        }
        for (int i = 8; i < 14; i++) {
            v[i] = 3 * (i - 8);
        }
        for (int i = 14; i < 18; i++) {
            v[i] = 1 + 0.5 * (i - 14);
        }
        return mat;
    }
};


TEST_F(CsrBlockDiagonal, GeneratesCorrectBlockDiagonalMatrix)
{
    auto matset = generate_csr_matrices();

    auto bdcsr = gko::create_block_diagonal_matrix(exec, matset);
    auto check = get_big_matrix();

    ASSERT_EQ(bdcsr->get_size(), check->get_size());
    for (size_t irow = 0; irow < bdcsr->get_size()[0]; irow++) {
        ASSERT_EQ(bdcsr->get_const_row_ptrs()[irow],
                  check->get_const_row_ptrs()[irow]);
        for (size_t inz = bdcsr->get_const_row_ptrs()[irow];
             inz < bdcsr->get_const_row_ptrs()[irow + 1]; inz++) {
            ASSERT_EQ(bdcsr->get_const_col_idxs()[inz],
                      check->get_const_col_idxs()[inz]);
            ASSERT_EQ(bdcsr->get_const_values()[inz],
                      check->get_const_values()[inz]);
        }
    }
    ASSERT_EQ(bdcsr->get_const_row_ptrs()[bdcsr->get_size()[0]],
              check->get_const_row_ptrs()[bdcsr->get_size()[0]]);
}

TEST_F(CsrBlockDiagonal, ThrowsOnRectangularInput)
{
    std::vector<std::unique_ptr<Mtx>> matrices;
    auto mtx = Mtx::create(exec, gko::dim<2>(3, 6), 10);
    matrices.emplace_back(std::move(mtx));

    ASSERT_THROW(gko::create_block_diagonal_matrix(exec, matrices),
                 gko::DimensionMismatch);
}


}  // namespace
