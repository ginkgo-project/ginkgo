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


#include <ginkgo/core/matrix/matrix_strategies.hpp>

#include <algorithm>

#include "core/components/fixed_block.hpp"
#include "fbcsr_sample.hpp"

namespace gko {
namespace testing {


namespace matstr = gko::matrix::matrix_strategy;

template <typename ValueType, typename IndexType>
FbcsrSample<ValueType, IndexType>::FbcsrSample(
    const std::shared_ptr<const gko::ReferenceExecutor> rexec)
    : nrows{6},
      ncols{12},
      nnz{36},
      nbrows{2},
      nbcols{4},
      nbnz{4},
      bs{3},
      exec(rexec)
{}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_fbcsr() const
{
    std::unique_ptr<Fbcsr> mtx =
        Fbcsr::create(exec,
                      gko::dim<2>{static_cast<size_type>(nrows),
                                  static_cast<size_type>(ncols)},
                      nnz, bs, std::make_shared<matstr::classical<Fbcsr>>());

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

    gko::blockutils::DenseBlocksView<value_type, index_type> vals(v, bs, bs);

    if (mtx->get_size()[0] % bs != 0)
        throw gko::BadDimension(__FILE__, __LINE__, __func__, "test fbcsr",
                                mtx->get_size()[0], mtx->get_size()[1],
                                "block size does not divide the size!");

    for (index_type ibrow = 0; ibrow < mtx->get_size()[0] / bs; ibrow++) {
        const index_type *const browptr = mtx->get_row_ptrs();
        for (index_type inz = browptr[ibrow]; inz < browptr[ibrow + 1]; inz++) {
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

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_csr() const
{
    std::unique_ptr<Csr> csrm =
        Csr::create(exec, gko::dim<2>{nrows, ncols}, nnz,
                    std::make_shared<typename Csr::classical>());
    index_type *const csrrow = csrm->get_row_ptrs();
    index_type *const csrcols = csrm->get_col_idxs();
    value_type *const csrvals = csrm->get_values();

    csrrow[0] = 0;
    csrrow[1] = 6;
    csrrow[2] = 12;
    csrrow[3] = 18;
    csrrow[4] = 24;
    csrrow[5] = 30;
    csrrow[6] = 36;

    csrcols[0] = 3;
    csrcols[1] = 4;
    csrcols[2] = 5;
    csrcols[6] = 3;
    csrcols[7] = 4;
    csrcols[8] = 5;
    csrcols[12] = 3;
    csrcols[13] = 4;
    csrcols[14] = 5;

    csrcols[3] = 9;
    csrcols[4] = 10;
    csrcols[5] = 11;
    csrcols[9] = 9;
    csrcols[10] = 10;
    csrcols[11] = 11;
    csrcols[15] = 9;
    csrcols[16] = 10;
    csrcols[17] = 11;

    csrcols[18] = 0;
    csrcols[19] = 1;
    csrcols[20] = 2;
    csrcols[24] = 0;
    csrcols[25] = 1;
    csrcols[26] = 2;
    csrcols[30] = 0;
    csrcols[31] = 1;
    csrcols[32] = 2;

    csrcols[21] = 6;
    csrcols[22] = 7;
    csrcols[23] = 8;
    csrcols[27] = 6;
    csrcols[28] = 7;
    csrcols[29] = 8;
    csrcols[33] = 6;
    csrcols[34] = 7;
    csrcols[35] = 8;

    // values
    csrvals[0] = 2;
    csrvals[1] = 3;
    csrvals[2] = 4;
    csrvals[6] = 5;
    csrvals[7] = 6;
    csrvals[8] = 7;
    csrvals[12] = 0;
    csrvals[13] = 9;
    csrvals[14] = 0;

    csrvals[3] = 4;
    csrvals[4] = 5;
    csrvals[5] = 6;
    csrvals[9] = 7;
    csrvals[10] = 8;
    csrvals[11] = 9;
    csrvals[15] = 10;
    csrvals[16] = 11;
    csrvals[17] = 12;

    csrvals[18] = 2;
    csrvals[19] = 3;
    csrvals[20] = 4;
    csrvals[24] = 5;
    csrvals[25] = 6;
    csrvals[26] = 7;
    csrvals[30] = 8;
    csrvals[31] = 9;
    csrvals[32] = 10;

    csrvals[21] = 0;
    csrvals[22] = 7;
    csrvals[23] = 8;
    csrvals[27] = 9;
    csrvals[28] = 10;
    csrvals[29] = 11;
    csrvals[33] = 12;
    csrvals[34] = 13;
    csrvals[35] = 14;

    return csrm;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>>
FbcsrSample<ValueType, IndexType>::generate_dense() const
{
    std::unique_ptr<Dense> densem =
        Dense::create(exec, gko::dim<2>(nrows, ncols));

    for (size_type irow = 0; irow < densem->get_size()[0]; irow++)
        for (size_type jcol = 0; jcol < densem->get_size()[1]; jcol++) {
            densem->at(irow, jcol) = 0;
            if (irow < 3 && jcol >= 3 && jcol < 6)
                densem->at(irow, jcol) = 2.0 + irow * bs + jcol - 3;
            if (irow < 3 && jcol >= 9)
                densem->at(irow, jcol) = 4.0 + irow * bs + jcol - 9;
            if (irow >= 3 && jcol < 3)
                densem->at(irow, jcol) = 2.0 + (irow - 3) * bs + jcol;
            if (irow >= 3 && jcol >= 6 && jcol < 9)
                densem->at(irow, jcol) = 6.0 + (irow - 3) * bs + jcol - 6;
        }

    densem->at(2, 3) = densem->at(2, 5) = densem->at(3, 6) = 0.0;

    return densem;
}

// Assuming row-major blocks
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType>
FbcsrSample<ValueType, IndexType>::generate_matrix_data() const
{
    return MatData(
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
}

// Assuming row-major blocks
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> FbcsrSample<
    ValueType, IndexType>::generate_matrix_data_with_explicit_zeros() const
{
    return MatData({{6, 12}, {{0, 3, 2.0},  {0, 4, 3.0},   {0, 5, 4.0},
                              {1, 3, 5.0},  {1, 4, 6.0},   {1, 5, 7.0},
                              {2, 3, 0.0},  {2, 4, 9.0},   {2, 5, 0.0},

                              {0, 9, 4.0},  {0, 10, 5.0},  {0, 11, 6.0},
                              {1, 9, 7.0},  {1, 10, 8.0},  {1, 11, 9.0},
                              {2, 9, 10.0}, {2, 10, 11.0}, {2, 11, 12.0},

                              {3, 0, 2.0},  {3, 1, 3.0},   {3, 2, 4.0},
                              {4, 0, 5.0},  {4, 1, 6.0},   {4, 2, 7.0},
                              {5, 0, 8.0},  {5, 1, 9.0},   {5, 2, 10.0},

                              {3, 6, 0.0},  {3, 7, 7.0},   {3, 8, 8.0},
                              {4, 6, 9.0},  {4, 7, 10.0},  {4, 8, 11.0},
                              {5, 6, 12.0}, {5, 7, 13.0},  {5, 8, 14.0}}});
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Coo<ValueType, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_coo() const
{
    gko::matrix_data<ValueType, IndexType> mdata =
        generate_matrix_data_with_explicit_zeros();

    using nztype =
        typename gko::matrix_data<ValueType, IndexType>::nonzero_type;
    std::sort(mdata.nonzeros.begin(), mdata.nonzeros.end(),
              [](const nztype &a, const nztype &b) {
                  if (a.row < b.row)
                      return true;
                  else if (a.row > b.row)
                      return false;
                  else if (a.column < b.column)
                      return true;
                  else
                      return false;
              });

    gko::Array<IndexType> rowidx(exec, nnz);
    gko::Array<IndexType> colidx(exec, nnz);
    gko::Array<ValueType> values(exec, nnz);
    for (size_t i = 0; i < mdata.nonzeros.size(); i++) {
        rowidx.get_data()[i] = mdata.nonzeros[i].row;
        colidx.get_data()[i] = mdata.nonzeros[i].column;
        values.get_data()[i] = mdata.nonzeros[i].value;
    }
    auto mat =
        Coo::create(exec, gko::dim<2>{nrows, ncols}, values, colidx, rowidx);
    return mat;
}

#define GKO_DECLARE_FBCSR_TEST_SAMPLE(ValueType, IndexType) \
    class FbcsrSample<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_TEST_SAMPLE);

}  // namespace testing
}  // namespace gko
