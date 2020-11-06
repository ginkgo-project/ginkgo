/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/matrix_strategies.hpp>


#include <gtest/gtest.h>
#include <iostream>
#include <limits>


#include "core/components/fixed_block.hpp"
#include "core/test/utils.hpp"


namespace {


namespace matstr = gko::matrix::matrix_strategy;


template <typename ValueIndexType>
class Fbcsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;

    Fbcsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{6, 12}, 36, 3,
                          std::make_shared<matstr::classical<Mtx>>()))
    {
        const int bs = 3;
        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
        index_type *const s = mtx->get_srow();
        r[0] = 0;
        r[1] = 2;
        r[2] = 4;
        c[0] = 1;
        c[1] = 3;
        c[2] = 0;
        c[3] = 2;

        gko::blockutils::DenseBlocksView<value_type, index_type> vals(v, bs,
                                                                      bs);

        if (mtx->get_size()[0] % bs != 0)
            throw gko::BadDimension(__FILE__, __LINE__, __func__, "test fbcsr",
                                    mtx->get_size()[0], mtx->get_size()[1],
                                    "block size does not divide the size!");

        for (index_type ibrow = 0; ibrow < mtx->get_size()[0] / bs; ibrow++) {
            const index_type *const browptr = mtx->get_row_ptrs();
            for (index_type inz = browptr[ibrow]; inz < browptr[ibrow + 1];
                 inz++) {
                const index_type bcolind = mtx->get_col_idxs()[inz];
                const value_type base = (ibrow + 1) * (bcolind + 1);
                for (int ival = 0; ival < bs; ival++)
                    for (int jval = 0; jval < bs; jval++)
                        vals(inz, ival, jval) =
                            base + static_cast<gko::remove_complex<value_type>>(
                                       ival * bs + jval);
            }
        }

        // Some of the entries are set to zero
        vals(0, 2, 0) = gko::zero<value_type>();
        vals(0, 2, 2) = gko::zero<value_type>();
        vals(3, 0, 0) = gko::zero<value_type>();

        for (index_type is = 0; is < mtx->get_num_srow_elements(); is++)
            s[is] = 0;

        // backup for move tests
        orig_size = mtx->get_size();
        orig_rowptrs.resize(3);
        orig_colinds.resize(4);
        orig_vals.resize(36);
        for (index_type i = 0; i < 3; i++) orig_rowptrs[i] = r[i];
        for (index_type i = 0; i < 4; i++) orig_colinds[i] = c[i];
        for (index_type i = 0; i < 36; i++) orig_vals[i] = v[i];
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    gko::dim<2> orig_size;
    std::vector<value_type> orig_vals;
    std::vector<index_type> orig_rowptrs;
    std::vector<index_type> orig_colinds;

    void assert_equal_to_original_mtx(const Mtx *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        auto s = m->get_const_srow();

        const int bs = 3;

        ASSERT_EQ(m->get_size(), orig_size);
        ASSERT_EQ(m->get_num_stored_elements(), orig_vals.size());
        ASSERT_EQ(m->get_block_size(), bs);


        for (index_type irow = 0; irow < orig_size[0] / bs; irow++) {
            const index_type *const rowptr = &orig_rowptrs[0];
            ASSERT_EQ(r[irow], rowptr[irow]);

            for (index_type inz = rowptr[irow]; inz < rowptr[irow + 1]; inz++) {
                ASSERT_EQ(c[inz], orig_colinds[inz]);

                for (int i = 0; i < bs * bs; i++) {
                    // ASSERT_LT(gko::abs(v[inz*bs*bs + i] -
                    // mtx->get_values()[inz*bs*bs + i]),
                    //           std::numeric_limits<gko::remove_complex<value_type>>::epsilon());
                    ASSERT_EQ(v[inz * bs * bs + i],
                              orig_vals[inz * bs * bs + i]);
                }
            }
        }

        ASSERT_EQ(m->get_num_srow_elements(), 0);
        // for(index_type is = 0; is < mtx->get_num_srow_elements(); is++)
        //     ASSERT_EQ(s[is], 0);
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 1);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
        ASSERT_EQ(m->get_const_srow(), nullptr);
    }
};

TYPED_TEST_CASE(Fbcsr, gko::test::ValueIndexTypes);


TYPED_TEST(Fbcsr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(6, 12));
    ASSERT_EQ(this->mtx->get_block_size(), 3);
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 36);
}


TYPED_TEST(Fbcsr, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Fbcsr, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Fbcsr, CanBeCreatedFromExistingData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    constexpr int bs = 3;
    constexpr index_type nbrows = 2;
    constexpr index_type nbcols = 4;
    constexpr index_type bnnz = 4;
    value_type values[bnnz * bs * bs];
    index_type col_idxs[] = {1, 3, 0, 2};
    index_type row_ptrs[] = {0, 2, 4};

    gko::blockutils::DenseBlocksView<value_type, index_type> vals(values, bs,
                                                                  bs);

    for (index_type ibrow = 0; ibrow < nbrows; ibrow++) {
        for (index_type inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             inz++) {
            const index_type bcolind = col_idxs[inz];
            const value_type base = (ibrow + 1) * (bcolind + 1);
            for (int ival = 0; ival < bs; ival++)
                for (int jval = 0; jval < bs; jval++)
                    vals(inz, ival, jval) =
                        base + static_cast<gko::remove_complex<value_type>>(
                                   ival * bs + jval);
        }
    }

    // Some of the entries are set to zero
    vals(0, 2, 0) = gko::zero<value_type>();
    vals(0, 2, 2) = gko::zero<value_type>();
    vals(3, 0, 0) = gko::zero<value_type>();

    auto mtx = gko::matrix::Fbcsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{nbrows * bs, nbcols * bs}, bs,
        gko::Array<value_type>::view(this->exec, bnnz * bs * bs, values),
        gko::Array<index_type>::view(this->exec, bnnz, col_idxs),
        gko::Array<index_type>::view(this->exec, nbrows + 1, row_ptrs),
        std::make_shared<matstr::classical<Mtx>>());

    ASSERT_EQ(mtx->get_num_srow_elements(), 0);
    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
    // ASSERT_EQ(mtx->get_const_srow()[0], 0);
}


TYPED_TEST(Fbcsr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 3.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Fbcsr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Fbcsr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx *>(clone.get()));
}


TYPED_TEST(Fbcsr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Fbcsr, CanBeReadFromMatrixData)
{
    // TODO (script:fbcsr): change the code imported from matrix/csr if needed
    using Mtx = typename TestFixture::Mtx;
    auto m =
        Mtx::create(this->exec, std::make_shared<matstr::classical<Mtx>>());
    m->set_block_size(3);

    // Assuming row-major blocks
    m->read(
        {{6, 12}, {{0, 3, 2.0},   {0, 4, 3.0},  {0, 5, 4.0},  {1, 3, 5.0},
                   {1, 4, 6.0},   {1, 5, 7.0},  {2, 4, 9.0},

                   {0, 9, 4.0},   {0, 10, 5.0}, {0, 11, 6.0}, {1, 9, 7.0},
                   {1, 10, 8.0},  {1, 11, 9.0}, {2, 9, 10.0}, {2, 10, 11.0},
                   {2, 11, 12.0},

                   {3, 0, 2.0},   {3, 1, 3.0},  {3, 2, 4.0},  {4, 0, 5.0},
                   {4, 1, 6.0},   {4, 2, 7.0},  {5, 0, 8.0},  {5, 1, 9.0},
                   {5, 2, 10.0},

                   {3, 7, 7.0},   {3, 8, 8.0},  {4, 6, 9.0},  {4, 7, 10.0},
                   {4, 8, 11.0},  {5, 6, 12.0}, {5, 7, 13.0}, {5, 8, 14.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Fbcsr, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);
    data.ensure_row_major_order();

    ASSERT_EQ(data.size, gko::dim<2>(6, 12));
    ASSERT_EQ(data.nonzeros.size(), 36);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 3, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 4, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 5, value_type{4.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(0, 9, value_type{4.0}));
    EXPECT_EQ(data.nonzeros[4], tpl(0, 10, value_type{5.0}));
    EXPECT_EQ(data.nonzeros[5], tpl(0, 11, value_type{6.0}));

    EXPECT_EQ(data.nonzeros[6], tpl(1, 3, value_type{5.0}));
    EXPECT_EQ(data.nonzeros[7], tpl(1, 4, value_type{6.0}));
    EXPECT_EQ(data.nonzeros[8], tpl(1, 5, value_type{7.0}));
    EXPECT_EQ(data.nonzeros[9], tpl(1, 9, value_type{7.0}));
    EXPECT_EQ(data.nonzeros[10], tpl(1, 10, value_type{8.0}));
    EXPECT_EQ(data.nonzeros[11], tpl(1, 11, value_type{9.0}));

    EXPECT_EQ(data.nonzeros[12], tpl(2, 3, value_type{0.0}));
    EXPECT_EQ(data.nonzeros[13], tpl(2, 4, value_type{9.0}));
    EXPECT_EQ(data.nonzeros[14], tpl(2, 5, value_type{0.0}));
    EXPECT_EQ(data.nonzeros[15], tpl(2, 9, value_type{10.0}));
    EXPECT_EQ(data.nonzeros[16], tpl(2, 10, value_type{11.0}));
    EXPECT_EQ(data.nonzeros[17], tpl(2, 11, value_type{12.0}));

    EXPECT_EQ(data.nonzeros[18], tpl(3, 0, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[19], tpl(3, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[20], tpl(3, 2, value_type{4.0}));
    EXPECT_EQ(data.nonzeros[21], tpl(3, 6, value_type{0.0}));
    EXPECT_EQ(data.nonzeros[22], tpl(3, 7, value_type{7.0}));
    EXPECT_EQ(data.nonzeros[23], tpl(3, 8, value_type{8.0}));

    EXPECT_EQ(data.nonzeros[24], tpl(4, 0, value_type{5.0}));
    EXPECT_EQ(data.nonzeros[25], tpl(4, 1, value_type{6.0}));
    EXPECT_EQ(data.nonzeros[26], tpl(4, 2, value_type{7.0}));
    EXPECT_EQ(data.nonzeros[27], tpl(4, 6, value_type{9.0}));
    EXPECT_EQ(data.nonzeros[28], tpl(4, 7, value_type{10.0}));
    EXPECT_EQ(data.nonzeros[29], tpl(4, 8, value_type{11.0}));

    EXPECT_EQ(data.nonzeros[30], tpl(5, 0, value_type{8.0}));
    EXPECT_EQ(data.nonzeros[31], tpl(5, 1, value_type{9.0}));
    EXPECT_EQ(data.nonzeros[32], tpl(5, 2, value_type{10.0}));
    EXPECT_EQ(data.nonzeros[33], tpl(5, 6, value_type{12.0}));
    EXPECT_EQ(data.nonzeros[34], tpl(5, 7, value_type{13.0}));
    EXPECT_EQ(data.nonzeros[35], tpl(5, 8, value_type{14.0}));
}


}  // namespace
