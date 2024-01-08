// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <algorithm>
#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/types.hpp>


#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename SampleGenerator>
void check_sample_generator_common(const SampleGenerator sg)
{
    auto fbmtx = sg.generate_fbcsr();
    ASSERT_EQ(fbmtx->get_num_block_rows(), sg.nbrows);
    ASSERT_EQ(fbmtx->get_num_block_cols(), sg.nbcols);
    ASSERT_EQ(fbmtx->get_size()[0], sg.nrows);
    ASSERT_EQ(fbmtx->get_size()[1], sg.ncols);
    ASSERT_EQ(fbmtx->get_num_stored_blocks(), sg.nbnz);
    ASSERT_EQ(fbmtx->get_num_stored_elements(), sg.nnz);
}


template <typename ValueIndexType>
class FbcsrSample : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;

    FbcsrSample() : ref(gko::ReferenceExecutor::create()) {}

    void assert_matrices_are_same(
        gko::ptr_param<const gko::matrix::Fbcsr<value_type, index_type>> bm,
        gko::ptr_param<const gko::matrix::Csr<value_type, index_type>> cm,
        const gko::matrix::Diagonal<value_type>* const diam = nullptr,
        const gko::matrix_data<value_type, index_type>* const md = nullptr)
    {
        if (cm) {
            ASSERT_EQ(bm->get_size(), cm->get_size());
            ASSERT_EQ(bm->get_num_stored_elements(),
                      cm->get_num_stored_elements());
        }
        if (md) {
            ASSERT_EQ(bm->get_size(), md->size);
            ASSERT_EQ(bm->get_num_stored_elements(), md->nonzeros.size());
        }
        if (diam) {
            const gko::size_type minsize =
                std::min(bm->get_size()[0], bm->get_size()[1]);
            ASSERT_EQ(minsize, diam->get_size()[0]);
            ASSERT_EQ(minsize, diam->get_size()[1]);
        }

        const index_type nbrows = bm->get_num_block_rows();
        const int bs = bm->get_block_size();
        const auto nbnz = bm->get_num_stored_blocks();
        gko::acc::range<gko::acc::block_col_major<const value_type, 3>> fbvals(
            std::array<gko::acc::size_type, 3>{
                static_cast<gko::acc::size_type>(nbnz),
                static_cast<gko::acc::size_type>(bs),
                static_cast<gko::acc::size_type>(bs)},
            bm->get_const_values());

        for (index_type ibrow = 0; ibrow < nbrows; ibrow++) {
            const index_type* const browptr = bm->get_const_row_ptrs();
            const index_type numblocksbrow =
                browptr[ibrow + 1] - browptr[ibrow];
            for (index_type irow = ibrow * bs; irow < ibrow * bs + bs; irow++) {
                const index_type rowstart =
                    browptr[ibrow] * bs * bs +
                    (irow - ibrow * bs) * numblocksbrow * bs;
                if (cm) {
                    ASSERT_EQ(cm->get_const_row_ptrs()[irow], rowstart);
                }
            }

            const index_type iz_browstart = browptr[ibrow] * bs * bs;
            const index_type* const bcolinds = bm->get_const_col_idxs();

            for (index_type ibnz = browptr[ibrow]; ibnz < browptr[ibrow + 1];
                 ibnz++) {
                const index_type bcol = bcolinds[ibnz];
                const index_type blkoffset_frombrowstart =
                    ibnz - browptr[ibrow];

                for (int ib = 0; ib < bs; ib++) {
                    const index_type row = ibrow * bs + ib;
                    const index_type inz_rowstart =
                        iz_browstart + ib * numblocksbrow * bs;
                    const index_type inz_blockstart_row =
                        inz_rowstart + blkoffset_frombrowstart * bs;

                    for (int jb = 0; jb < bs; jb++) {
                        const index_type col = bcol * bs + jb;
                        const index_type inz = inz_blockstart_row + jb;
                        if (cm) {
                            ASSERT_EQ(col, cm->get_const_col_idxs()[inz]);
                            ASSERT_EQ(fbvals(ibnz, ib, jb),
                                      cm->get_const_values()[inz]);
                        }
                        if (md) {
                            ASSERT_EQ(row, md->nonzeros[inz].row);
                            ASSERT_EQ(col, md->nonzeros[inz].column);
                            ASSERT_EQ(fbvals(ibnz, ib, jb),
                                      md->nonzeros[inz].value);
                        }
                        if (row == col && diam) {
                            ASSERT_EQ(fbvals(ibnz, ib, jb),
                                      diam->get_const_values()[row]);
                        }
                    }
                }
            }
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};


TYPED_TEST_SUITE(FbcsrSample, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(FbcsrSample, SampleGeneratorsAreCorrect)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using MtxData = gko::matrix_data<value_type, index_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    auto ref = this->ref;
    gko::testing::FbcsrSample<value_type, index_type> fbsample(ref);
    gko::testing::FbcsrSample2<value_type, index_type> fbsample2(ref);

    std::unique_ptr<const Mtx> fbmtx = fbsample.generate_fbcsr();
    std::unique_ptr<const Csr> csmtx = fbsample.generate_csr();
    const MtxData mdata = fbsample.generate_matrix_data_with_explicit_zeros();
    std::unique_ptr<const Mtx> fbmtx2 = fbsample2.generate_fbcsr();
    std::unique_ptr<const Csr> csmtx2 = fbsample2.generate_csr();
    std::unique_ptr<const Diag> diag2 = fbsample2.extract_diagonal();
    const gko::array<index_type> nnzperrow = fbsample2.getNonzerosPerRow();

    check_sample_generator_common(fbsample);
    this->assert_matrices_are_same(fbmtx, csmtx,
                                   static_cast<const Diag*>(nullptr), &mdata);
    check_sample_generator_common(fbsample2);
    this->assert_matrices_are_same(fbmtx2, csmtx2, diag2.get());
    for (index_type irow = 0; irow < fbsample2.nrows; irow++) {
        const index_type* const row_ptrs = csmtx2->get_const_row_ptrs();
        const index_type num_nnz_row = row_ptrs[irow + 1] - row_ptrs[irow];
        ASSERT_EQ(nnzperrow.get_const_data()[irow], num_nnz_row);
        for (index_type iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
            const index_type col = csmtx2->get_const_col_idxs()[iz];
            if (irow == col) {
                ASSERT_EQ(csmtx2->get_const_values()[iz],
                          diag2->get_const_values()[irow]);
            }
        }
    }
    check_sample_generator_common(
        gko::testing::FbcsrSampleUnsorted<value_type, index_type>(ref));
    check_sample_generator_common(
        gko::testing::FbcsrSampleSquare<value_type, index_type>(ref));
}


template <typename ValueIndexType>
class FbcsrSampleComplex : public FbcsrSample<ValueIndexType> {};


TYPED_TEST_SUITE(FbcsrSampleComplex, gko::test::ComplexValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(FbcsrSampleComplex, ComplexSampleGeneratorIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    auto ref = this->ref;
    gko::testing::FbcsrSampleComplex<value_type, index_type> fbsample3(ref);

    std::unique_ptr<const Mtx> fbmtx3 = fbsample3.generate_fbcsr();
    std::unique_ptr<const Csr> csmtx3 = fbsample3.generate_csr();

    check_sample_generator_common(fbsample3);
    this->assert_matrices_are_same(fbmtx3, csmtx3);
}


template <typename ValueIndexType>
class Fbcsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using MtxData = gko::matrix_data<value_type, index_type>;

    Fbcsr()
        : exec(gko::ReferenceExecutor::create()),
          fbsample(exec),
          mtx(fbsample.generate_fbcsr())
    {
        // backup for move tests
        const value_type* const v = mtx->get_values();
        const index_type* const c = mtx->get_col_idxs();
        const index_type* const r = mtx->get_row_ptrs();
        orig_size = mtx->get_size();
        orig_rowptrs.resize(fbsample.nbrows + 1);
        orig_colinds.resize(fbsample.nbnz);
        orig_vals.resize(fbsample.nnz);
        std::copy(r, r + fbsample.nbrows + 1, orig_rowptrs.data());
        std::copy(c, c + fbsample.nbnz, orig_colinds.data());
        std::copy(v, v + fbsample.nnz, orig_vals.data());
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const gko::testing::FbcsrSample<value_type, index_type> fbsample;
    std::unique_ptr<Mtx> mtx;

    gko::dim<2> orig_size;
    std::vector<value_type> orig_vals;
    std::vector<index_type> orig_rowptrs;
    std::vector<index_type> orig_colinds;

    void assert_equal_to_original_mtx(gko::ptr_param<const Mtx> m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();

        const int bs = fbsample.bs;

        ASSERT_EQ(m->get_size(), orig_size);
        ASSERT_EQ(m->get_num_stored_elements(), orig_vals.size());
        ASSERT_EQ(m->get_block_size(), bs);
        ASSERT_EQ(m->get_num_block_rows(), m->get_size()[0] / bs);
        ASSERT_EQ(m->get_num_block_cols(), m->get_size()[1] / bs);

        for (index_type irow = 0; irow < orig_size[0] / bs; irow++) {
            const index_type* const rowptr = &orig_rowptrs[0];
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

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 1);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_NE(m->get_const_row_ptrs(), nullptr);
    }
};

TYPED_TEST_SUITE(Fbcsr, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Fbcsr, GetNumBlocksCorrectlyThrows)
{
    using index_type = typename TestFixture::index_type;
    const index_type vec_sz = 47;
    const int blk_sz = 9;

    ASSERT_THROW(gko::matrix::detail::get_num_blocks(blk_sz, vec_sz),
                 gko::BlockSizeError<decltype(vec_sz)>);
}


TYPED_TEST(Fbcsr, GetNumBlocksWorks)
{
    using index_type = typename TestFixture::index_type;
    const index_type vec_sz = 45;
    const int blk_sz = 9;

    ASSERT_EQ(gko::matrix::detail::get_num_blocks(blk_sz, vec_sz), 5);
}


TYPED_TEST(Fbcsr, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(6, 12));
    ASSERT_EQ(this->mtx->get_block_size(), 3);
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 36);
    ASSERT_EQ(this->mtx->get_num_block_rows(), 2);
    ASSERT_EQ(this->mtx->get_num_block_cols(), 4);
}


TYPED_TEST(Fbcsr, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx);
}


TYPED_TEST(Fbcsr, BlockSizeIsSetCorrectly)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec, 6);

    ASSERT_EQ(m->get_block_size(), 6);
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
    value_type* const values = refmat->get_values();
    index_type* const col_idxs = refmat->get_col_idxs();
    index_type* const row_ptrs = refmat->get_row_ptrs();

    auto mtx = gko::matrix::Fbcsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{nbrows * bs, nbcols * bs}, bs,
        gko::make_array_view(this->exec, bnnz * bs * bs, values),
        gko::make_array_view(this->exec, bnnz, col_idxs),
        gko::make_array_view(this->exec, nbrows + 1, row_ptrs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
}


TYPED_TEST(Fbcsr, CanBeCreatedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using size_type = gko::size_type;
    const int bs = this->fbsample.bs;
    const size_type nbrows = this->fbsample.nbrows;
    const size_type nbcols = this->fbsample.nbcols;
    const size_type bnnz = this->fbsample.nbnz;
    auto refmat = this->fbsample.generate_fbcsr();
    auto values = refmat->get_const_values();
    auto col_idxs = refmat->get_const_col_idxs();
    auto row_ptrs = refmat->get_const_row_ptrs();

    auto mtx = gko::matrix::Fbcsr<value_type, index_type>::create_const(
        this->exec, gko::dim<2>{nbrows * bs, nbcols * bs}, bs,
        gko::array<value_type>::const_view(this->exec, bnnz * bs * bs, values),
        gko::array<index_type>::const_view(this->exec, bnnz, col_idxs),
        gko::array<index_type>::const_view(this->exec, nbrows + 1, row_ptrs));

    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_ptrs(), row_ptrs);
}


TYPED_TEST(Fbcsr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx);

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 3.0;
    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Fbcsr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->move_from(this->mtx);

    this->assert_equal_to_original_mtx(copy);
}


TYPED_TEST(Fbcsr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;

    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx);
    this->mtx->get_values()[1] = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Fbcsr, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Fbcsr, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec, this->fbsample.bs);

    m->read(this->fbsample.generate_matrix_data());

    this->assert_equal_to_original_mtx(m);
}


TYPED_TEST(Fbcsr, CanBeReadFromEmptyMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    using MtxData = typename TestFixture::MtxData;
    auto m = Mtx::create(this->exec, this->fbsample.bs);
    MtxData mdata;
    mdata.size = gko::dim<2>{0, 0};

    m->read(mdata);

    ASSERT_EQ(m->get_size(), (gko::dim<2>{0, 0}));
    ASSERT_EQ(m->get_const_row_ptrs()[0], 0);
    ASSERT_EQ(m->get_const_col_idxs(), nullptr);
    ASSERT_EQ(m->get_const_values(), nullptr);
}


TYPED_TEST(Fbcsr, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using MtxData = typename TestFixture::MtxData;
    MtxData refdata = this->fbsample.generate_matrix_data_with_explicit_zeros();
    refdata.sort_row_major();

    MtxData data;
    this->mtx->write(data);
    data.sort_row_major();

    ASSERT_EQ(data.size, refdata.size);
    ASSERT_EQ(data.nonzeros.size(), refdata.nonzeros.size());
    for (size_t i = 0; i < data.nonzeros.size(); i++) {
        ASSERT_EQ(data.nonzeros[i], refdata.nonzeros[i]);
    }
}


}  // namespace
