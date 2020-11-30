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
#include "core/test/matrix/fbcsr_sample.hpp"

#define FBCSR_TEST_OFFSET 0.000011118888

#define FBCSR_TEST_C_MAG 0.1 + FBCSR_TEST_OFFSET
#define FBCSR_TEST_IMAGINARY \
    sct(std::complex<remove_complex<ValueType>>(0, FBCSR_TEST_C_MAG))

namespace gko {
namespace testing {


namespace matstr = gko::matrix::matrix_strategy;

/** Generates a copy of the given matrix with a different scalar type
 *
 * \tparam AbsValueType The scalar type of the output matrix
 */
template <typename FbcsrType, typename AbsValueType>
static std::unique_ptr<
    gko::matrix::Fbcsr<AbsValueType, typename FbcsrType::index_type>>
generate_acopy_impl(const FbcsrType *const mat)
{
    using index_type = typename FbcsrType::index_type;
    using value_type = typename FbcsrType::value_type;
    using AbsFbcsr = gko::matrix::Fbcsr<AbsValueType, index_type>;
    using classical = matstr::classical<AbsFbcsr>;

    std::shared_ptr<const ReferenceExecutor> exec =
        std::dynamic_pointer_cast<const ReferenceExecutor>(mat->get_executor());

    std::unique_ptr<AbsFbcsr> amat =
        AbsFbcsr::create(exec, mat->get_size(), mat->get_num_stored_elements(),
                         mat->get_block_size(), std::make_shared<classical>());

    const index_type *const colidxs = mat->get_col_idxs();
    const index_type *const rowptrs = mat->get_row_ptrs();
    index_type *const acolidxs = amat->get_col_idxs();
    index_type *const arowptrs = amat->get_row_ptrs();

    for (index_type i = 0;
         i < mat->get_num_stored_elements() /
                 (mat->get_block_size() * mat->get_block_size());
         i++)
        acolidxs[i] = colidxs[i];

    for (index_type i = 0; i < mat->get_size()[0] / mat->get_block_size() + 1;
         i++)
        arowptrs[i] = rowptrs[i];

    return amat;
}


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

    for (index_type ibrow = 0; ibrow < mtx->get_num_block_rows(); ibrow++) {
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

    v[34] += FBCSR_TEST_IMAGINARY;
    v[35] += FBCSR_TEST_IMAGINARY;

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template <typename ValueType, typename IndexType>
template <typename FbcsrType>
void FbcsrSample<ValueType, IndexType>::correct_abs_for_complex_values(
    FbcsrType *const amat) const
{
    using out_value_type = typename FbcsrType::value_type;
    using outreal_type = remove_complex<out_value_type>;
    out_value_type *const avals = amat->get_values();
    if (is_complex<ValueType>()) {
        auto mo = static_cast<outreal_type>(FBCSR_TEST_C_MAG);
        avals[34] = sqrt(pow(static_cast<outreal_type>(13.0), 2) +
                         pow(static_cast<outreal_type>(mo), 2));
        avals[35] = sqrt(pow(static_cast<outreal_type>(14.0), 2) +
                         pow(static_cast<outreal_type>(mo), 2));
    }
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<remove_complex<ValueType>, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_abs_fbcsr_abstype() const
{
    using AbsValueType = typename gko::remove_complex<ValueType>;
    using AbsFbcsr = gko::matrix::Fbcsr<AbsValueType, IndexType>;

    const std::unique_ptr<const Fbcsr> mat = generate_fbcsr();
    std::unique_ptr<AbsFbcsr> amat =
        generate_acopy_impl<Fbcsr, AbsValueType>(mat.get());

    AbsValueType *const avals = amat->get_values();
    const ValueType *const vals = mat->get_values();
    for (IndexType i = 0; i < amat->get_num_stored_elements(); i++)
        avals[i] = abs(vals[i]);

    correct_abs_for_complex_values(amat.get());

    return amat;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_abs_fbcsr() const
{
    const std::unique_ptr<const Fbcsr> mat = generate_fbcsr();
    std::unique_ptr<Fbcsr> amat =
        generate_acopy_impl<Fbcsr, ValueType>(mat.get());

    ValueType *const avals = amat->get_values();
    const ValueType *const vals = mat->get_values();
    for (IndexType i = 0; i < amat->get_num_stored_elements(); i++)
        avals[i] = abs(vals[i]);

    correct_abs_for_complex_values(amat.get());

    return amat;
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

    csrvals[34] += FBCSR_TEST_IMAGINARY;
    csrvals[35] += FBCSR_TEST_IMAGINARY;

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
    densem->at(5, 7) += FBCSR_TEST_IMAGINARY;
    densem->at(5, 8) += FBCSR_TEST_IMAGINARY;

    return densem;
}

// Assuming row-major blocks
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType>
FbcsrSample<ValueType, IndexType>::generate_matrix_data() const
{
    return MatData({{6, 12},
                    {{0, 3, 2.0},
                     {0, 4, 3.0},
                     {0, 5, 4.0},
                     {1, 3, 5.0},
                     {1, 4, 6.0},
                     {1, 5, 7.0},
                     {2, 4, 9.0},

                     {0, 9, 4.0},
                     {0, 10, 5.0},
                     {0, 11, 6.0},
                     {1, 9, 7.0},
                     {1, 10, 8.0},
                     {1, 11, 9.0},
                     {2, 9, 10.0},
                     {2, 10, 11.0},
                     {2, 11, 12.0},

                     {3, 0, 2.0},
                     {3, 1, 3.0},
                     {3, 2, 4.0},
                     {4, 0, 5.0},
                     {4, 1, 6.0},
                     {4, 2, 7.0},
                     {5, 0, 8.0},
                     {5, 1, 9.0},
                     {5, 2, 10.0},

                     {3, 7, 7.0},
                     {3, 8, 8.0},
                     {4, 6, 9.0},
                     {4, 7, 10.0},
                     {4, 8, 11.0},
                     {5, 6, 12.0},
                     {5, 7, sct(13.0) + FBCSR_TEST_IMAGINARY},
                     {5, 8, sct(14.0) + FBCSR_TEST_IMAGINARY}}});
}

// Assuming row-major blocks
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> FbcsrSample<
    ValueType, IndexType>::generate_matrix_data_with_explicit_zeros() const
{
    return MatData({{6, 12},
                    {{0, 3, 2.0},
                     {0, 4, 3.0},
                     {0, 5, 4.0},
                     {1, 3, 5.0},
                     {1, 4, 6.0},
                     {1, 5, 7.0},
                     {2, 3, 0.0},
                     {2, 4, 9.0},
                     {2, 5, 0.0},

                     {0, 9, 4.0},
                     {0, 10, 5.0},
                     {0, 11, 6.0},
                     {1, 9, 7.0},
                     {1, 10, 8.0},
                     {1, 11, 9.0},
                     {2, 9, 10.0},
                     {2, 10, 11.0},
                     {2, 11, 12.0},

                     {3, 0, 2.0},
                     {3, 1, 3.0},
                     {3, 2, 4.0},
                     {4, 0, 5.0},
                     {4, 1, 6.0},
                     {4, 2, 7.0},
                     {5, 0, 8.0},
                     {5, 1, 9.0},
                     {5, 2, 10.0},

                     {3, 6, 0.0},
                     {3, 7, 7.0},
                     {3, 8, 8.0},
                     {4, 6, 9.0},
                     {4, 7, 10.0},
                     {4, 8, 11.0},
                     {5, 6, 12.0},
                     {5, 7, sct(13.0) + FBCSR_TEST_IMAGINARY},
                     {5, 8, sct(14.0) + FBCSR_TEST_IMAGINARY}}});
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

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::SparsityCsr<ValueType, IndexType>>
FbcsrSample<ValueType, IndexType>::generate_sparsity_csr() const
{
    gko::Array<IndexType> colids(exec, nbnz);
    gko::Array<IndexType> rowptrs(exec, nbrows + 1);
    const std::unique_ptr<const Fbcsr> fbmat = generate_fbcsr();
    for (index_type i = 0; i < nbrows + 1; i++)
        rowptrs.get_data()[i] = fbmat->get_row_ptrs()[i];
    for (index_type i = 0; i < nbnz; i++)
        colids.get_data()[i] = fbmat->get_col_idxs()[i];
    return SparCsr::create(exec, gko::dim<2>{nbrows, nbcols}, colids, rowptrs);
}

template <typename ValueType, typename IndexType>
gko::Array<IndexType> FbcsrSample<ValueType, IndexType>::getNonzerosPerRow()
    const
{
    return gko::Array<index_type>(exec, {6, 6, 6, 6, 6, 6});
}

#define GKO_DECLARE_FBCSR_TEST_SAMPLE(ValueType, IndexType) \
    class FbcsrSample<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_TEST_SAMPLE);


template <typename ValueType, typename IndexType>
FbcsrSample2<ValueType, IndexType>::FbcsrSample2(
    const std::shared_ptr<const gko::ReferenceExecutor> rexec)
    : nrows{6},
      ncols{8},
      nnz{16},
      nbrows{3},
      nbcols{4},
      nbnz{4},
      bs{2},
      exec(rexec)
{}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSample2<ValueType, IndexType>::generate_fbcsr() const
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
    r[1] = 1;
    r[2] = 3;
    r[3] = 4;
    c[0] = 0;
    c[1] = 0;
    c[2] = 3;
    c[3] = 2;

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;

    v[0] = 1;
    v[1] = 2;
    v[2] = 3;
    v[3] = 0;
    v[10] = 0;
    v[11] = 0;
    v[12] = -12;
    v[13] = -1;
    v[14] = -2;
    v[15] = -11;

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<remove_complex<ValueType>, IndexType>>
FbcsrSample2<ValueType, IndexType>::generate_abs_fbcsr_abstype() const
{
    using AbsValueType = typename gko::remove_complex<ValueType>;
    using AbsFbcsr = gko::matrix::Fbcsr<AbsValueType, IndexType>;

    const std::unique_ptr<const Fbcsr> mat = generate_fbcsr();
    std::unique_ptr<AbsFbcsr> amat =
        generate_acopy_impl<Fbcsr, AbsValueType>(mat.get());

    AbsValueType *const v = amat->get_values();

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;
    v[0] = 1;
    v[1] = 2;
    v[2] = 3;
    v[3] = 0;
    v[10] = 0;
    v[11] = 0;
    v[12] = 12;
    v[13] = 1;
    v[14] = 2;
    v[15] = 11;

    return amat;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSample2<ValueType, IndexType>::generate_abs_fbcsr() const
{
    const std::unique_ptr<const Fbcsr> mat = generate_fbcsr();
    std::unique_ptr<Fbcsr> amat =
        generate_acopy_impl<Fbcsr, ValueType>(mat.get());

    ValueType *const v = amat->get_values();

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    v[3] = 0.0;
    v[10] = 0.0;
    v[11] = 0.0;
    v[12] = 12.0;
    v[13] = 1.0;
    v[14] = 2.0;
    v[15] = 11.0;

    return amat;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSample2<ValueType, IndexType>::generate_transpose_fbcsr() const
{
    std::unique_ptr<Fbcsr> mtx =
        Fbcsr::create(exec,
                      gko::dim<2>{static_cast<size_type>(ncols),
                                  static_cast<size_type>(nrows)},
                      nnz, bs, std::make_shared<matstr::classical<Fbcsr>>());

    value_type *const v = mtx->get_values();
    index_type *const c = mtx->get_col_idxs();
    index_type *const r = mtx->get_row_ptrs();
    index_type *const s = mtx->get_srow();
    r[0] = 0;
    r[1] = 2;
    r[2] = 2;
    r[3] = 3;
    r[4] = 4;
    c[0] = 0;
    c[1] = 1;
    c[2] = 2;
    c[3] = 1;

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;

    v[0] = 1;
    v[1] = 3;
    v[2] = 2;
    v[3] = 0;
    v[8] = -12;
    v[9] = -2;
    v[10] = -1;
    v[11] = -11;
    v[13] = 0;
    v[15] = 0;

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Diagonal<ValueType>>
FbcsrSample2<ValueType, IndexType>::extract_diagonal() const
{
    gko::Array<ValueType> dvals(exec, nrows);
    ValueType *const dv = dvals.get_data();
    dv[0] = 1;
    dv[1] = 0;
    dv[2] = 0;
    dv[3] = 0;
    dv[4] = -12;
    dv[5] = -11;
    return Diagonal::create(exec, nrows, dvals);
}

template <typename ValueType, typename IndexType>
gko::Array<IndexType> FbcsrSample2<ValueType, IndexType>::getNonzerosPerRow()
    const
{
    return gko::Array<index_type>(exec, {2, 2, 4, 4, 2, 2});
}

template <typename ValueType, typename IndexType>
void FbcsrSample2<ValueType, IndexType>::apply(
    const gko::matrix::Dense<ValueType> *const x,
    gko::matrix::Dense<ValueType> *const y) const
{
    if (x->get_size()[0] != ncols)
        throw gko::BadDimension(__FILE__, __LINE__, __func__, "spmv", nrows,
                                ncols, "");
    if (y->get_size()[0] != nrows)
        throw gko::BadDimension(__FILE__, __LINE__, __func__, "spmv", nrows,
                                ncols, "");
    if (x->get_size()[1] != y->get_size()[1])
        throw gko::BadDimension(__FILE__, __LINE__, __func__, "spmv", nrows,
                                ncols, "");

    const ValueType defv = sct(0.15 + FBCSR_TEST_OFFSET);

    for (index_type k = 0; k < x->get_size()[1]; k++) {
        y->at(0, k) = sct(1.0) * x->at(0, k) + sct(2.0) * x->at(1, k);
        y->at(1, k) = sct(3.0) * x->at(0, k);
        y->at(2, k) =
            defv * (x->at(0, k) + x->at(1, k) + x->at(6, k) + x->at(7, k));
        y->at(3, k) = defv * (x->at(0, k) + x->at(1, k));
        y->at(4, k) = sct(-12.0) * x->at(4, k) - x->at(5, k);
        y->at(5, k) = sct(-2.0) * x->at(4, k) + sct(-11.0) * x->at(5, k);
    }
}

#define GKO_DECLARE_FBCSR_TEST_SAMPLE_2(ValueType, IndexType) \
    class FbcsrSample2<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_TEST_SAMPLE_2);


template <typename ValueType, typename IndexType>
FbcsrSampleSquare<ValueType, IndexType>::FbcsrSampleSquare(
    const std::shared_ptr<const gko::ReferenceExecutor> rexec)
    : nrows{4},
      ncols{4},
      nnz{8},
      nbrows{2},
      nbcols{2},
      nbnz{2},
      bs{2},
      exec(rexec)
{}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSampleSquare<ValueType, IndexType>::generate_fbcsr() const
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
    r[1] = 1;
    r[2] = 2;
    c[0] = 1;
    c[1] = 1;

    for (IndexType i = 0; i < nnz; i++) v[i] = i;

    return mtx;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSampleSquare<ValueType, IndexType>::generate_transpose_fbcsr() const
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
    r[1] = 0;
    r[2] = 2;
    c[0] = 0;
    c[1] = 1;

    gko::blockutils::DenseBlocksView<value_type, index_type> vals(v, bs, bs);
    vals(0, 0, 0) = 0;
    vals(0, 0, 1) = 2;
    vals(0, 1, 0) = 1;
    vals(0, 1, 1) = 3;
    vals(1, 0, 0) = 4;
    vals(1, 0, 1) = 6;
    vals(1, 1, 0) = 5;
    vals(1, 1, 1) = 7;

    return mtx;
}

#define GKO_DECLARE_FBCSR_TEST_SAMPLE_SQUARE(ValueType, IndexType) \
    class FbcsrSampleSquare<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TEST_SAMPLE_SQUARE);


template <typename ValueType, typename IndexType>
FbcsrSampleComplex<ValueType, IndexType>::FbcsrSampleComplex(
    const std::shared_ptr<const gko::ReferenceExecutor> rexec)
    : nrows{6},
      ncols{8},
      nnz{16},
      nbrows{3},
      nbcols{4},
      nbnz{4},
      bs{2},
      exec(rexec)
{}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSampleComplex<ValueType, IndexType>::generate_fbcsr() const
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
    r[1] = 1;
    r[2] = 3;
    r[3] = 4;
    c[0] = 0;
    c[1] = 0;
    c[2] = 3;
    c[3] = 2;

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;

    using namespace std::complex_literals;
    v[0] = 1.0 + 1.15i;
    v[1] = 2.0 + 2.15i;
    v[2] = 3.0 - 3.15i;
    v[3] = 0.0 - 0.15i;
    v[10] = 0.0;
    v[11] = 0.0;
    v[12] = -12.0 + 12.15i;
    v[13] = -1.0 + 1.15i;
    v[14] = -2.0 - 2.15i;
    v[15] = -11.0 - 11.15i;

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Fbcsr<ValueType, IndexType>>
FbcsrSampleComplex<ValueType, IndexType>::generate_conjtranspose_fbcsr() const
{
    std::unique_ptr<Fbcsr> mtx =
        Fbcsr::create(exec,
                      gko::dim<2>{static_cast<size_type>(ncols),
                                  static_cast<size_type>(nrows)},
                      nnz, bs, std::make_shared<matstr::classical<Fbcsr>>());

    value_type *const v = mtx->get_values();
    index_type *const c = mtx->get_col_idxs();
    index_type *const r = mtx->get_row_ptrs();
    index_type *const s = mtx->get_srow();
    r[0] = 0;
    r[1] = 2;
    r[2] = 2;
    r[3] = 3;
    r[4] = 4;
    c[0] = 0;
    c[1] = 1;
    c[2] = 2;
    c[3] = 1;

    for (IndexType i = 0; i < nnz; i++) v[i] = 0.15 + FBCSR_TEST_OFFSET;

    using namespace std::complex_literals;
    v[0] = 1.0 - 1.15i;
    v[1] = 3.0 + 3.15i;
    v[2] = 2.0 - 2.15i;
    v[3] = 0.0 + 0.15i;
    v[8] = -12.0 - 12.15i;
    v[9] = -2.0 + 2.15i;
    v[10] = -1.0 - 1.15i;
    v[11] = -11.0 + 11.15i;
    v[13] = 0;
    v[15] = 0;

    for (index_type is = 0; is < mtx->get_num_srow_elements(); is++) s[is] = 0;

    return mtx;
}

template class FbcsrSampleComplex<std::complex<float>, int>;
template class FbcsrSampleComplex<std::complex<double>, int>;
template class FbcsrSampleComplex<std::complex<float>, long>;
template class FbcsrSampleComplex<std::complex<double>, long>;

}  // namespace testing
}  // namespace gko
