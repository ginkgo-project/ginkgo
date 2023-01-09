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

#ifndef GKO_CORE_TEST_UTILS_BATCH_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_HPP_


#include <random>


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/test/utils/assertions.hpp"


namespace gko {
namespace test {


/**
 * Converts a vector of unique pointers to a vector of shared pointers.
 */
template <typename T>
std::vector<std::shared_ptr<T>> share(std::vector<std::unique_ptr<T>>&& objs)
{
    std::vector<std::shared_ptr<T>> out;
    out.reserve(objs.size());
    for (auto& obj : objs) {
        out.push_back(std::move(obj));
    }
    return out;
}


/**
 * Generates a batch of random matrices of the specified type.
 */
template <typename MatrixType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_uniform_batch_random_matrix(
    const size_type batch_size, const size_type num_rows,
    const size_type num_cols, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    const bool with_all_diagonals, std::shared_ptr<const Executor> exec,
    MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    // generate sparsity pattern
    matrix_data<value_type, index_type> sdata{gko::dim<2>{num_rows, num_cols},
                                              {}};

    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        std::vector<size_type> col_idx(num_cols);
        std::iota(begin(col_idx), end(col_idx), size_type(0));
        const auto nnz_row = static_cast<size_type>(nonzero_dist(engine));
        size_type nnz_in_row =
            std::max(size_type(0), std::min(nnz_row, num_cols));
        std::shuffle(std::begin(col_idx), std::end(col_idx), engine);

        if (with_all_diagonals) {
            if (nnz_in_row == 0) {
                nnz_in_row = 1;
            }
            bool has_diagonal = false;
            for (size_type icol = 0; icol < nnz_in_row; icol++) {
                if (col_idx[icol] == row) {
                    has_diagonal = true;
                }
            }
            if (!has_diagonal) {
                col_idx[0] = row;
            }
        }

        std::for_each(
            std::begin(col_idx), std::begin(col_idx) + nnz_in_row,
            [&](size_type col) { sdata.nonzeros.emplace_back(row, col, 1.0); });
    }

    std::vector<matrix_data<value_type, index_type>> batchmtx;
    batchmtx.reserve(batch_size);

    for (size_t ibatch = 0; ibatch < batch_size; ibatch++) {
        matrix_data<value_type, index_type> data = sdata;
        for (size_type iz = 0; iz < data.nonzeros.size(); ++iz) {
            value_type valnz =
                gko::detail::get_rand_value<value_type>(value_dist, engine);
            if (data.nonzeros[iz].column == data.nonzeros[iz].row &&
                valnz == zero<value_type>()) {
                valnz = 1.0;
            }
            data.nonzeros[iz].value = valnz;
        }

        data.ensure_row_major_order();
        batchmtx.emplace_back(std::move(data));
    }

    // convert to the correct matrix type
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(batchmtx);
    return result;
}


/**
 * Generate a batch of 1D Poisson matrices on the reference executor.
 *
 * @tparam MatrixType  The concrete type of the output matrix.
 *
 * @param exec  The reference executor.
 * @param nrows  The size (number of rows) of the generated matrix
 * @param nbatch  The number of Poisson matrices in the batch
 */
template <typename MatrixType>
std::unique_ptr<MatrixType> create_poisson1d_batch(
    std::shared_ptr<const ReferenceExecutor> exec, const int nrows,
    const size_type nbatch)
{
    using ValueType = typename MatrixType::value_type;
    using IndexType = typename MatrixType::index_type;
    const int nnz = 3 * (nrows - 2) + 4;
    gko::array<IndexType> row_ptrs(exec, nrows + 1);
    {
        const auto ra = row_ptrs.get_data();
        ra[0] = 0;
        ra[1] = 2;
        for (int i = 2; i < nrows; i++) {
            ra[i] = ra[i - 1] + 3;
        }
        ra[nrows] = ra[nrows - 1] + 2;
        GKO_ASSERT(ra[nrows] == nnz);
    }
    gko::array<IndexType> col_idxs(exec, nnz);
    {
        const auto ca = col_idxs.get_data();
        ca[0] = 0;
        ca[1] = 1;
        for (int i = 1; i < nrows - 1; i++) {
            const int rstart = row_ptrs.get_const_data()[i];
            ca[rstart] = i - 1;
            ca[rstart + 1] = i;
            ca[rstart + 2] = i + 1;
        }
        const auto lrstart = row_ptrs.get_const_data()[nrows - 1];
        ca[lrstart] = nrows - 2;
        ca[lrstart + 1] = nrows - 1;
    }
    gko::array<ValueType> vals(exec, nnz * nbatch);
    for (int i = 0; i < nbatch; i++) {
        ValueType* const va = vals.get_data() + i * nnz;
        va[0] = 2.0;
        va[1] = -1.0;
        for (int i = 1; i < nrows - 1; i++) {
            const int rstart = row_ptrs.get_const_data()[i];
            va[rstart] = -1.0;
            va[rstart + 1] = 2.0;
            va[rstart + 2] = -1.0;
        }
        const auto lrstart = row_ptrs.get_const_data()[nrows - 1];
        va[lrstart] = -1.0;
        va[lrstart + 1] = 2.0;
    }
    using Csr = matrix::BatchCsr<ValueType, IndexType>;
    auto csr = Csr::create(
        exec, nbatch,
        gko::dim<2>{static_cast<size_t>(nrows), static_cast<size_t>(nrows)},
        std::move(vals), std::move(col_idxs), std::move(row_ptrs));
    std::vector<gko::matrix_data<ValueType, IndexType>> mdata;
    csr->write(mdata);
    auto mtx = MatrixType::create(exec);
    mtx->read(mdata);
    return mtx;
}


template <typename ValueType>
struct BatchSystem {
    using vec_type = matrix::BatchDense<ValueType>;
    std::unique_ptr<BatchLinOp> A;
    std::unique_ptr<vec_type> b;
};


/**
 * Generate a batch of randomly changed, almost-tridiagonal matrices and
 * a RHS vector.
 */
template <typename MatrixType>
BatchSystem<typename MatrixType::value_type> generate_solvable_batch_system(
    std::shared_ptr<const gko::Executor> exec, const size_type nsystems,
    const int nrows, const int nrhs, const bool symmetric)
{
    using value_type = typename MatrixType::value_type;
    using index_type = int;
    using real_type = remove_complex<value_type>;
    const int nnz = nrows * 3 - 2;
    std::ranlux48 rgen(15);
    std::normal_distribution<real_type> distb(0.5, 0.1);
    auto h_exec = exec->get_master();
    std::vector<real_type> spacings(nsystems * nrows);
    std::generate(spacings.begin(), spacings.end(),
                  [&]() { return distb(rgen); });

    array<value_type> h_allvalues(h_exec, nnz * nsystems);
    for (size_type isys = 0; isys < nsystems; isys++) {
        h_allvalues.get_data()[isys * nnz] = 2.0 / spacings[isys * nrows];
        h_allvalues.get_data()[isys * nnz + 1] = -1.0;
        for (int irow = 0; irow < nrows - 2; irow++) {
            h_allvalues.get_data()[isys * nnz + 2 + irow * 3] = -1.0;
            h_allvalues.get_data()[isys * nnz + 2 + irow * 3 + 1] =
                2.0 / spacings[isys * nrows + irow + 1];
            h_allvalues.get_data()[isys * nnz + 2 + irow * 3 + 2] = -1.0;
            // break some symmetry
            if (!symmetric && irow < 5) {
                h_allvalues.get_data()[isys * nnz + 2 + irow * 3] = -0.75;
            }
        }
        h_allvalues.get_data()[isys * nnz + 2 + (nrows - 2) * 3] = -1.0;
        h_allvalues.get_data()[isys * nnz + 2 + (nrows - 2) * 3 + 1] =
            2.0 / spacings[(isys + 1) * nrows - 1];
        assert(isys * nnz + 2 + (nrows - 2) * 3 + 2 == (isys + 1) * nnz);
    }

    array<index_type> h_rowptrs(h_exec, nrows + 1);
    h_rowptrs.get_data()[0] = 0;
    h_rowptrs.get_data()[1] = 2;
    for (int i = 2; i < nrows; i++) {
        h_rowptrs.get_data()[i] = h_rowptrs.get_data()[i - 1] + 3;
    }
    h_rowptrs.get_data()[nrows] = h_rowptrs.get_data()[nrows - 1] + 2;
    assert(h_rowptrs.get_data()[nrows] == nnz);

    array<index_type> h_colidxs(h_exec, nnz);
    h_colidxs.get_data()[0] = 0;
    h_colidxs.get_data()[1] = 1;
    const int nnz_per_row = 3;
    for (int irow = 1; irow < nrows - 1; irow++) {
        h_colidxs.get_data()[2 + (irow - 1) * nnz_per_row] = irow - 1;
        h_colidxs.get_data()[2 + (irow - 1) * nnz_per_row + 1] = irow;
        h_colidxs.get_data()[2 + (irow - 1) * nnz_per_row + 2] = irow + 1;
        // break some structural symmetry
        if (!symmetric && irow > 1 && irow < 5) {
            h_colidxs.get_data()[2 + (irow - 1) * nnz_per_row] = irow - 2;
        }
    }
    h_colidxs.get_data()[2 + (nrows - 2) * nnz_per_row] = nrows - 2;
    h_colidxs.get_data()[2 + (nrows - 2) * nnz_per_row + 1] = nrows - 1;
    assert(2 + (nrows - 2) * nnz_per_row + 1 == nnz - 1);

    array<value_type> h_allb(h_exec, nrows * nrhs * nsystems);
    for (size_type ib = 0; ib < nsystems; ib++) {
        for (int j = 0; j < nrhs; j++) {
            const value_type bval = distb(rgen);
            for (int i = 0; i < nrows; i++) {
                h_allb.get_data()[ib * nrows * nrhs + i * nrhs + j] = bval;
            }
        }
    }

    auto vec_size = batch_dim<>(nsystems, dim<2>(nrows, 1));
    auto vec_stride = batch_stride(nsystems, 1);
    auto mat_size = batch_dim<>(nsystems, dim<2>(nrows, nrows));
    using Csr = matrix::BatchCsr<value_type, index_type>;
    auto mcsr = Csr::create(exec, mat_size, h_allvalues, h_colidxs, h_rowptrs);
    std::vector<matrix_data<value_type, index_type>> mdata;
    mcsr->write(mdata);
    auto mtx = MatrixType::create(exec);
    mtx->read(mdata);
    BatchSystem<value_type> sys;
    sys.A = gko::give(mtx);
    sys.b = BatchSystem<value_type>::vec_type::create(exec, vec_size, h_allb,
                                                      vec_stride);
    return sys;
}


template <typename MtxType>
void remove_diagonal_from_row(MtxType* const mtx, const int row)
{
    using value_type = typename MtxType::value_type;
    std::vector<gko::matrix_data<value_type, int>> mdata;
    mtx->write(mdata);
    for (size_type i = 0; i < mdata.size(); i++) {
        auto it = std::find_if(
            mdata[i].nonzeros.begin(), mdata[i].nonzeros.end(),
            [row](auto x) { return x.row == row && x.column == row; });
        if (it != mdata[i].nonzeros.end()) {
            mdata[i].nonzeros.erase(it);
        }
    }
    mtx->read(mdata);
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_BATCH_HPP_
