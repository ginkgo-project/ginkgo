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

#ifndef GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_
#define GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fixed_block.hpp"


#define FBCSR_TEST_OFFSET 0.000011118888

#define FBCSR_TEST_C_MAG 0.1 + FBCSR_TEST_OFFSET

#define FBCSR_TEST_IMAGINARY \
    sct(std::complex<remove_complex<ValueType>>(0, FBCSR_TEST_C_MAG))


namespace gko {
namespace testing {


/** Generates the same sample block CSR matrix in different formats
 *
 * This currently a 6 x 12 matrix with 3x3 blocks.
 * Assumes that the layout within each block is row-major.
 * Generates complex data when instantiated with a complex value type.
 */
template <typename ValueType, typename IndexType>
class FbcsrSample {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using MatData = gko::matrix_data<value_type, index_type>;
    using SparCsr = gko::matrix::SparsityCsr<value_type, index_type>;


    const size_type nrows = 6;
    const size_type ncols = 12;
    const size_type nnz = 36;
    const size_type nbrows = 2;
    const size_type nbcols = 4;
    const size_type nbnz = 4;
    const int bs = 3;
    const std::shared_ptr<const gko::Executor> exec;


    FbcsrSample(std::shared_ptr<const gko::ReferenceExecutor> rexec)
        : exec(rexec)
    {}

    /**
     * @return The sample matrix in FBCSR format
     */
    std::unique_ptr<Fbcsr> generate_fbcsr() const
    {
        std::unique_ptr<Fbcsr> mtx =
            Fbcsr::create(exec,
                          gko::dim<2>{static_cast<size_type>(nrows),
                                      static_cast<size_type>(ncols)},
                          nnz, bs);

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
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

        for (index_type ibrow = 0; ibrow < mtx->get_num_block_rows(); ibrow++) {
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

        v[34] += FBCSR_TEST_IMAGINARY;
        v[35] += FBCSR_TEST_IMAGINARY;

        return mtx;
    }

    /**
     * @return Sample matrix in CSR format
     *
     * Keeps explicit zeros.
     */
    std::unique_ptr<Csr> generate_csr() const
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
        csrvals[0] = 2;
        csrcols[1] = 4;
        csrvals[1] = 3;
        csrcols[2] = 5;
        csrvals[2] = 4;
        csrcols[6] = 3;
        csrvals[6] = 5;
        csrcols[7] = 4;
        csrvals[7] = 6;
        csrcols[8] = 5;
        csrvals[8] = 7;
        csrcols[12] = 3;
        csrvals[12] = 0;
        csrcols[13] = 4;
        csrvals[13] = 9;
        csrcols[14] = 5;
        csrvals[14] = 0;

        csrcols[3] = 9;
        csrvals[3] = 4;
        csrcols[4] = 10;
        csrvals[4] = 5;
        csrcols[5] = 11;
        csrvals[5] = 6;
        csrcols[9] = 9;
        csrvals[9] = 7;
        csrcols[10] = 10;
        csrvals[10] = 8;
        csrcols[11] = 11;
        csrvals[11] = 9;
        csrcols[15] = 9;
        csrvals[15] = 10;
        csrcols[16] = 10;
        csrvals[16] = 11;
        csrcols[17] = 11;
        csrvals[17] = 12;

        csrcols[18] = 0;
        csrvals[18] = 2;
        csrcols[19] = 1;
        csrvals[19] = 3;
        csrcols[20] = 2;
        csrvals[20] = 4;
        csrcols[24] = 0;
        csrvals[24] = 5;
        csrcols[25] = 1;
        csrvals[25] = 6;
        csrcols[26] = 2;
        csrvals[26] = 7;
        csrcols[30] = 0;
        csrvals[30] = 8;
        csrcols[31] = 1;
        csrvals[31] = 9;
        csrcols[32] = 2;
        csrvals[32] = 10;

        csrcols[21] = 6;
        csrvals[21] = 0;
        csrcols[22] = 7;
        csrvals[22] = 7;
        csrcols[23] = 8;
        csrvals[23] = 8;
        csrcols[27] = 6;
        csrvals[27] = 9;
        csrcols[28] = 7;
        csrvals[28] = 10;
        csrcols[29] = 8;
        csrvals[29] = 11;
        csrcols[33] = 6;
        csrvals[33] = 12;
        csrcols[34] = 7;
        csrvals[34] = 13;
        csrcols[35] = 8;
        csrvals[35] = 14;

        csrvals[34] += FBCSR_TEST_IMAGINARY;
        csrvals[35] += FBCSR_TEST_IMAGINARY;

        return csrm;
    }

    /**
     * @return Sparsity structure of the matrix
     */
    std::unique_ptr<SparCsr> generate_sparsity_csr() const
    {
        gko::Array<IndexType> colids(exec, nbnz);
        gko::Array<IndexType> rowptrs(exec, nbrows + 1);
        const std::unique_ptr<const Fbcsr> fbmat = generate_fbcsr();
        for (index_type i = 0; i < nbrows + 1; i++)
            rowptrs.get_data()[i] = fbmat->get_row_ptrs()[i];
        for (index_type i = 0; i < nbnz; i++)
            colids.get_data()[i] = fbmat->get_col_idxs()[i];
        return SparCsr::create(exec, gko::dim<2>{nbrows, nbcols}, colids,
                               rowptrs);
    }

    /**
     * @return Array of COO triplets that represent the matrix
     *
     * @note The order of the triplets assumes the blocks are stored row-major
     */
    MatData generate_matrix_data() const
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

    /**
     * @return Array of COO triplets that represent the matrix; includes
     *         explicit zeros
     *
     * @note The order of the triplets assumes the blocks are stored row-major
     */
    MatData generate_matrix_data_with_explicit_zeros() const
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

private:
    /// Enables complex data to be used for complex instantiations...
    template <typename U>
    constexpr std::enable_if_t<!is_complex<U>() || is_complex<ValueType>(),
                               ValueType>
    sct(U u) const
    {
        return static_cast<ValueType>(u);
    }

    /// ... while ignoring imaginary parts for real instantiations
    template <typename U>
    constexpr std::enable_if_t<!is_complex<U>() && !is_complex<ValueType>(),
                               ValueType>
    sct(std::complex<U> cu) const
    {
        return static_cast<ValueType>(cu.real());
    }
};

/**
 * Generates a sample block CSR matrix in different formats.
 * 6 x 8 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSample2 {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Diagonal = gko::matrix::Diagonal<value_type>;
    using DenseBlocksView =
        gko::blockutils::DenseBlocksView<ValueType, IndexType>;


    const size_type nrows = 6;
    const size_type ncols = 8;
    const size_type nnz = 16;
    const size_type nbrows = 3;
    const size_type nbcols = 4;
    const size_type nbnz = 4;
    const int bs = 2;
    const std::shared_ptr<const gko::Executor> exec;


    FbcsrSample2(std::shared_ptr<const gko::ReferenceExecutor> rexec)
        : exec(rexec)
    {}

    std::unique_ptr<Fbcsr> generate_fbcsr() const
    {
        std::unique_ptr<Fbcsr> mtx =
            Fbcsr::create(exec,
                          gko::dim<2>{static_cast<size_type>(nrows),
                                      static_cast<size_type>(ncols)},
                          nnz, bs);

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
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

        return mtx;
    }

    std::unique_ptr<Csr> generate_csr() const
    {
        std::unique_ptr<Csr> mtx =
            Csr::create(exec,
                        gko::dim<2>{static_cast<size_type>(nrows),
                                    static_cast<size_type>(ncols)},
                        nnz, std::make_shared<typename Csr::classical>());

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 2;
        r[2] = 4;
        r[3] = 8;
        r[4] = 12;
        r[5] = 14;
        r[6] = 16;

        c[0] = 0;
        c[1] = 1;
        c[2] = 0;
        c[3] = 1;
        c[4] = 0;
        c[5] = 1;
        c[6] = 6;
        c[7] = 7;
        c[8] = 0;
        c[9] = 1;
        c[10] = 6;
        c[11] = 7;
        c[12] = 4;
        c[13] = 5;
        c[14] = 4;
        c[15] = 5;

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

        return mtx;
    }

    std::unique_ptr<Diagonal> extract_diagonal() const
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

    gko::Array<index_type> getNonzerosPerRow() const
    {
        return gko::Array<index_type>(exec, {2, 2, 4, 4, 2, 2});
    }

    /// Fills a view into a FBCSR values array using the sample matrix's data
    void fill_value_blocks_view(DenseBlocksView &dbv) const
    {
        dbv(0, 0, 0) = 1.0;
        dbv(0, 0, 1) = 2.0;
        dbv(0, 1, 0) = 3.0;
        dbv(0, 1, 1) = 0.0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) dbv(1, i, j) = 0.15 + FBCSR_TEST_OFFSET;
        dbv(2, 0, 0) = 0.15 + FBCSR_TEST_OFFSET;
        dbv(2, 0, 1) = 0.15 + FBCSR_TEST_OFFSET;
        dbv(2, 1, 0) = 0.0;
        dbv(2, 1, 1) = 0.0;
        dbv(3, 0, 0) = -12.0;
        dbv(3, 0, 1) = -1.0;
        dbv(3, 1, 0) = -2.0;
        dbv(3, 1, 1) = -11.0;
    }


private:
    /// Enables use of literals to instantiate value data
    template <typename U>
    constexpr ValueType sct(U u) const
    {
        return static_cast<ValueType>(u);
    }
};

/**
 * @brief Generates the a sample block CSR square matrix and its transpose
 *
 * This currently a 4 x 4 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSampleSquare {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;


    const size_type nrows = 4;
    const size_type ncols = 4;
    const size_type nnz = 8;
    const size_type nbrows = 2;
    const size_type nbcols = 2;
    const size_type nbnz = 2;
    const int bs = 2;
    const std::shared_ptr<const gko::Executor> exec;


    FbcsrSampleSquare(std::shared_ptr<const gko::ReferenceExecutor> rexec)
        : exec(rexec)
    {}

    std::unique_ptr<Fbcsr> generate_fbcsr() const
    {
        std::unique_ptr<Fbcsr> mtx =
            Fbcsr::create(exec,
                          gko::dim<2>{static_cast<size_type>(nrows),
                                      static_cast<size_type>(ncols)},
                          nnz, bs);

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 1;
        r[2] = 2;
        c[0] = 1;
        c[1] = 1;

        for (IndexType i = 0; i < nnz; i++) v[i] = i;

        return mtx;
    }
};

/**
 * @brief Generates a sample block CSR matrix with complex values
 *
 * This is a 6 x 8 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSampleComplex {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;


    static_assert(is_complex<ValueType>(), "Only for complex types!");


    const size_type nrows = 6;
    const size_type ncols = 8;
    const size_type nnz = 16;
    const size_type nbrows = 3;
    const size_type nbcols = 4;
    const size_type nbnz = 4;
    const int bs = 2;
    const std::shared_ptr<const gko::Executor> exec;


    FbcsrSampleComplex(std::shared_ptr<const gko::ReferenceExecutor> rexec)
        : exec(rexec)
    {}

    std::unique_ptr<Fbcsr> generate_fbcsr() const
    {
        std::unique_ptr<Fbcsr> mtx =
            Fbcsr::create(exec,
                          gko::dim<2>{static_cast<size_type>(nrows),
                                      static_cast<size_type>(ncols)},
                          nnz, bs);

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
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

        return mtx;
    }

    std::unique_ptr<Csr> generate_csr() const
    {
        std::unique_ptr<Csr> mtx =
            Csr::create(exec,
                        gko::dim<2>{static_cast<size_type>(nrows),
                                    static_cast<size_type>(ncols)},
                        nnz, std::make_shared<typename Csr::classical>());

        value_type *const v = mtx->get_values();
        index_type *const c = mtx->get_col_idxs();
        index_type *const r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 2;
        r[2] = 4;
        r[3] = 8;
        r[4] = 12;
        r[5] = 14;
        r[6] = 16;

        c[0] = 0;
        c[1] = 1;
        c[2] = 0;
        c[3] = 1;
        c[4] = 0;
        c[5] = 1;
        c[6] = 6;
        c[7] = 7;
        c[8] = 0;
        c[9] = 1;
        c[10] = 6;
        c[11] = 7;
        c[12] = 4;
        c[13] = 5;
        c[14] = 4;
        c[15] = 5;

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

        return mtx;
    }
};

}  // namespace testing
}  // namespace gko

#endif  // GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_
