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


#include <iostream>
#include <limits>


#include <gtest/gtest.h>


#include "core/components/fixed_block.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


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
          fbsample(
              std::static_pointer_cast<const gko::ReferenceExecutor>(exec)),
          mtx(fbsample.generate_fbcsr())
    {
        // backup for move tests
        const value_type *const v = mtx->get_values();
        const index_type *const c = mtx->get_col_idxs();
        const index_type *const r = mtx->get_row_ptrs();
        orig_size = mtx->get_size();
        orig_rowptrs.resize(fbsample.nbrows + 1);
        orig_colinds.resize(fbsample.nbnz);
        orig_vals.resize(fbsample.nnz);
        for (index_type i = 0; i < fbsample.nbrows + 1; i++)
            orig_rowptrs[i] = r[i];
        for (index_type i = 0; i < fbsample.nbnz; i++) orig_colinds[i] = c[i];
        for (index_type i = 0; i < fbsample.nnz; i++) orig_vals[i] = v[i];
    }

    std::shared_ptr<const gko::Executor> exec;
    const gko::testing::FbcsrSample<value_type, index_type> fbsample;
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
                    ASSERT_EQ(v[inz * bs * bs + i],
                              orig_vals[inz * bs * bs + i]);
                }
            }
        }
    }

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 1);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
    }
};

TYPED_TEST_SUITE(Fbcsr, gko::test::ValueIndexTypes);


TYPED_TEST(Fbcsr, SampleGeneratorIsCorrect)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    std::unique_ptr<const Mtx> fbmtx = this->fbsample.generate_fbcsr();
    std::unique_ptr<const Csr> csmtx = this->fbsample.generate_csr();
    const int bs = this->fbsample.bs;
    ASSERT_EQ(bs, fbmtx->get_block_size());

    gko::blockutils::DenseBlocksView<const value_type, index_type> fbvals(
        fbmtx->get_const_values(), bs, bs);

    for (index_type ibrow = 0; ibrow < this->fbsample.nbrows; ibrow++) {
        const index_type *const browptr = fbmtx->get_row_ptrs();
        const index_type numblocksbrow = browptr[ibrow + 1] - browptr[ibrow];
        for (index_type irow = ibrow * bs; irow < ibrow * bs + bs; irow++) {
            const index_type rowstart =
                browptr[ibrow] * bs * bs +
                (irow - ibrow * bs) * numblocksbrow * bs;
            ASSERT_EQ(csmtx->get_const_row_ptrs()[irow], rowstart);
        }

        const index_type *const bcolinds = fbmtx->get_col_idxs();

        for (index_type ibnz = browptr[ibrow]; ibnz < browptr[ibrow + 1];
             ibnz++) {
            const index_type bcol = bcolinds[ibnz];
            const index_type blkoffset_frombrowstart = ibnz - browptr[ibrow];

            for (int ib = 0; ib < bs; ib++) {
                const index_type row = ibrow * bs + ib;
                const index_type inz_rowstart =
                    csmtx->get_const_row_ptrs()[row] +
                    blkoffset_frombrowstart * bs;

                for (int jb = 0; jb < bs; jb++) {
                    const index_type col = bcol * bs + jb;
                    const index_type inz = inz_rowstart + jb;
                    ASSERT_EQ(col, csmtx->get_const_col_idxs()[inz]);
                    ASSERT_EQ(fbvals(ibnz, ib, jb),
                              csmtx->get_const_values()[inz]);
                }
            }
        }
    }
}


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
    using size_type = gko::size_type;

    const int bs = this->fbsample.bs;
    const size_type nbrows = this->fbsample.nbrows;
    const size_type nbcols = this->fbsample.nbcols;
    const size_type bnnz = this->fbsample.nbnz;
    std::unique_ptr<Mtx> refmat = this->fbsample.generate_fbcsr();
    value_type *const values = refmat->get_values();
    index_type *const col_idxs = refmat->get_col_idxs();
    index_type *const row_ptrs = refmat->get_row_ptrs();

    auto mtx = gko::matrix::Fbcsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{nbrows * bs, nbcols * bs}, bs,
        gko::Array<value_type>::view(this->exec, bnnz * bs * bs, values),
        gko::Array<index_type>::view(this->exec, bnnz, col_idxs),
        gko::Array<index_type>::view(this->exec, nbrows + 1, row_ptrs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
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
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->set_block_size(this->fbsample.bs);

    m->read(this->fbsample.generate_matrix_data());

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

    gko::matrix_data<value_type, index_type> refdata =
        this->fbsample.generate_matrix_data_with_explicit_zeros();
    refdata.ensure_row_major_order();

    ASSERT_EQ(data.size, refdata.size);
    ASSERT_EQ(data.nonzeros.size(), refdata.nonzeros.size());
    for (size_t i = 0; i < data.nonzeros.size(); i++)
        ASSERT_EQ(data.nonzeros[i], refdata.nonzeros[i]);
}


TYPED_TEST(Fbcsr, DenseBlocksViewWorksCorrectly)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dbv = gko::blockutils::DenseBlocksView<value_type, index_type>;

    const gko::testing::FbcsrSample2<value_type, index_type> fbsample(
        std::static_pointer_cast<const gko::ReferenceExecutor>(this->exec));

    auto refmtx = fbsample.generate_fbcsr();
    const Dbv testdbv(refmtx->get_values(), fbsample.bs, fbsample.bs);

    std::vector<value_type> ref_dbv_array(fbsample.nnz);
    Dbv refdbv(ref_dbv_array.data(), fbsample.bs, fbsample.bs);
    fbsample.fill_value_blocks_view(refdbv);

    for (index_type ibz = 0; ibz < fbsample.nbnz; ibz++)
        for (int i = 0; i < fbsample.bs; ++i)
            for (int j = 0; j < fbsample.bs; ++j)
                ASSERT_EQ(testdbv(ibz, i, j), refdbv(ibz, i, j));
}


}  // namespace
