// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_
#define GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/test/utils.hpp"


namespace gko {
namespace testing {


constexpr double fbcsr_test_offset = 0.000011118888;


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

        value_type* const v = mtx->get_values();
        index_type* const c = mtx->get_col_idxs();
        index_type* const r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 2;
        r[2] = 4;
        c[0] = 1;
        c[1] = 3;
        c[2] = 0;
        c[3] = 2;

        gko::acc::range<gko::acc::block_col_major<value_type, 3>> vals(
            std::array<gko::acc::size_type, 3>{
                static_cast<gko::acc::size_type>(nbnz),
                static_cast<gko::acc::size_type>(bs),
                static_cast<gko::acc::size_type>(bs)},
            v);

        if (mtx->get_size()[0] % bs != 0) {
            throw gko::BadDimension(__FILE__, __LINE__, __func__, "test fbcsr",
                                    mtx->get_size()[0], mtx->get_size()[1],
                                    "block size does not divide the size!");
        }

        for (index_type ibrow = 0; ibrow < mtx->get_num_block_rows(); ibrow++) {
            const index_type* const browptr = mtx->get_row_ptrs();
            for (index_type inz = browptr[ibrow]; inz < browptr[ibrow + 1];
                 inz++) {
                const index_type bcolind = mtx->get_col_idxs()[inz];
                const value_type base = (ibrow + 1) * (bcolind + 1);
                for (int ival = 0; ival < bs; ival++) {
                    for (int jval = 0; jval < bs; jval++) {
                        vals(inz, ival, jval) =
                            base + static_cast<gko::remove_complex<value_type>>(
                                       ival * bs + jval);
                    }
                }
            }
        }

        // Some of the entries are set to zero
        vals(0, 2, 0) = gko::zero<value_type>();
        vals(0, 2, 2) = gko::zero<value_type>();
        vals(3, 0, 0) = gko::zero<value_type>();

        vals(3, 2, 1) += fbcsr_test_imaginary;
        vals(3, 2, 2) += fbcsr_test_imaginary;

        return mtx;
    }

    /**
     * @return Sample matrix in CSR format
     *
     * Keeps explicit zeros.
     */
    std::unique_ptr<Csr> generate_csr() const
    {
        gko::array<index_type> csrrow(exec, {0, 6, 12, 18, 24, 30, 36});
        gko::array<index_type> csrcols(
            exec, {3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11,
                   0, 1, 2, 6, 7,  8,  0, 1, 2, 6, 7,  8,  0, 1, 2, 6, 7,  8});
        // clang-format off
        gko::array<value_type> csrvals(exec, I<value_type>
            {2, 3, 4, 4, 5, 6, 5, 6, 7, 7, 8, 9, 0, 9, 0,
	         10, 11, 12, 2, 3, 4, 0, 7, 8, 5, 6, 7,
	         9, 10, 11, 8, 9, 10, 12,
	         sct<value_type>(13.0) + fbcsr_test_imaginary,
	         sct<value_type>(14.0) + fbcsr_test_imaginary});
        // clang-format on
        return Csr::create(exec, gko::dim<2>{nrows, ncols}, csrvals, csrcols,
                           csrrow);
    }

    /**
     * @return Sparsity structure of the matrix
     */
    std::unique_ptr<SparCsr> generate_sparsity_csr() const
    {
        gko::array<IndexType> colids(exec, nbnz);
        gko::array<IndexType> rowptrs(exec, nbrows + 1);
        const std::unique_ptr<const Fbcsr> fbmat = generate_fbcsr();
        for (index_type i = 0; i < nbrows + 1; i++) {
            rowptrs.get_data()[i] = fbmat->get_const_row_ptrs()[i];
        }
        for (index_type i = 0; i < nbnz; i++) {
            colids.get_data()[i] = fbmat->get_const_col_idxs()[i];
        }
        return SparCsr::create(exec, gko::dim<2>{nbrows, nbcols}, colids,
                               rowptrs);
    }

    /**
     * @return array of COO triplets that represent the matrix
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
                         {5, 7, sct(13.0) + fbcsr_test_imaginary},
                         {5, 8, sct(14.0) + fbcsr_test_imaginary}}});
    }

    /**
     * @return array of COO triplets that represent the matrix; includes
     *         explicit zeros
     *
     * @note The order of the triplets assumes the blocks are stored row-major
     */
    MatData generate_matrix_data_with_explicit_zeros() const
    {
        auto mdata = generate_matrix_data();
        mdata.nonzeros.push_back({2, 3, 0.0});
        mdata.nonzeros.push_back({2, 5, 0.0});
        mdata.nonzeros.push_back({3, 6, 0.0});
        mdata.sort_row_major();
        return mdata;
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

    const ValueType fbcsr_test_imaginary = sct(
        std::complex<remove_complex<ValueType>>(0, 0.1 + fbcsr_test_offset));
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
        gko::array<index_type> r(exec, {0, 1, 3, 4});
        gko::array<index_type> c(exec, {0, 0, 3, 2});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = 0.15 + fbcsr_test_offset;
        }

        v[0] = 1;
        v[1] = 3;
        v[2] = 2;
        v[3] = 0;
        v[9] = 0;
        v[11] = 0;
        v[12] = -12;
        v[13] = -2;
        v[14] = -1;
        v[15] = -11;

        return Fbcsr::create(exec,
                             gko::dim<2>{static_cast<size_type>(nrows),
                                         static_cast<size_type>(ncols)},
                             bs, vals, c, r);
    }

    std::unique_ptr<Csr> generate_csr() const
    {
        gko::array<index_type> r(exec, {0, 2, 4, 8, 12, 14, 16});
        gko::array<index_type> c(
            exec, {0, 1, 0, 1, 0, 1, 6, 7, 0, 1, 6, 7, 4, 5, 4, 5});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = 0.15 + fbcsr_test_offset;
        }
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

        return Csr::create(exec,
                           gko::dim<2>{static_cast<size_type>(nrows),
                                       static_cast<size_type>(ncols)},
                           vals, c, r,
                           std::make_shared<typename Csr::classical>());
    }

    std::unique_ptr<Diagonal> extract_diagonal() const
    {
        gko::array<ValueType> dvals(exec, {1, 0, 0, 0, -12, -11});
        return Diagonal::create(exec, nrows, dvals);
    }

    gko::array<index_type> getNonzerosPerRow() const
    {
        return gko::array<index_type>(exec, {2, 2, 4, 4, 2, 2});
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
 * @brief Generates the a sample block CSR square matrix
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
        gko::array<index_type> c(exec, {1, 1});
        gko::array<index_type> r(exec, {0, 1, 2});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = i;
        }

        return Fbcsr::create(exec,
                             gko::dim<2>{static_cast<size_type>(nrows),
                                         static_cast<size_type>(ncols)},
                             bs, vals, c, r);
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
        gko::array<index_type> r(exec, {0, 1, 3, 4});
        gko::array<index_type> c(exec, {0, 0, 3, 2});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = 0.15 + fbcsr_test_offset;
        }

        v[0] = value_type{1.0, 1.15};
        v[2] = value_type{2.0, 2.15};
        v[1] = value_type{3.0, -3.15};
        v[3] = value_type{0.0, -0.15};
        v[9] = 0.0;
        v[11] = 0.0;
        v[12] = -value_type{12.0, 12.15};
        v[14] = -value_type{1.0, 1.15};
        v[13] = -value_type{2.0, -2.15};
        v[15] = -value_type{11.0, -11.15};

        return Fbcsr::create(exec,
                             gko::dim<2>{static_cast<size_type>(nrows),
                                         static_cast<size_type>(ncols)},
                             bs, vals, c, r);
    }

    std::unique_ptr<Csr> generate_csr() const
    {
        gko::array<index_type> r(exec, {0, 2, 4, 8, 12, 14, 16});
        gko::array<index_type> c(
            exec, {0, 1, 0, 1, 0, 1, 6, 7, 0, 1, 6, 7, 4, 5, 4, 5});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = 0.15 + fbcsr_test_offset;
        }

        v[0] = value_type{1.0, 1.15};
        v[1] = value_type{2.0, 2.15};
        v[2] = value_type{3.0, -3.15};
        v[3] = value_type{0.0, -0.15};
        v[10] = 0.0;
        v[11] = 0.0;
        v[12] = -value_type{12.0, 12.15};
        v[13] = -value_type{1.0, 1.15};
        v[14] = -value_type{2.0, -2.15};
        v[15] = -value_type{11.0, -11.15};

        return Csr::create(exec,
                           gko::dim<2>{static_cast<size_type>(nrows),
                                       static_cast<size_type>(ncols)},
                           vals, c, r,
                           std::make_shared<typename Csr::classical>());
    }
};


/**
 * Generates a fixed-block CSR matrix with longer and unsorted columns
 */
template <typename ValueType, typename IndexType>
class FbcsrSampleUnsorted {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;


    const size_type nbrows = 3;
    const size_type nbcols = 20;
    const size_type nbnz = 30;
    const int bs = 3;
    const size_type nrows = nbrows * bs;
    const size_type ncols = nbcols * bs;
    const size_type nnz = nbnz * bs * bs;
    const std::shared_ptr<const gko::Executor> exec;


    FbcsrSampleUnsorted(std::shared_ptr<const gko::ReferenceExecutor> rexec)
        : exec(rexec)
    {}

    std::unique_ptr<Fbcsr> generate_fbcsr() const
    {
        gko::array<index_type> r(exec, {0, 8, 19, 30});
        gko::array<index_type> c(
            exec, {0,  1,  20, 15, 12, 18, 5, 28, 3,  10, 29, 5,  9,  2,  16,
                   12, 21, 2,  0,  1,  5,  9, 12, 15, 17, 20, 22, 24, 27, 28});
        gko::array<value_type> vals(exec, nnz);
        value_type* const v = vals.get_data();
        for (IndexType i = 0; i < nnz; i++) {
            v[i] = static_cast<value_type>(i + 0.15 + fbcsr_test_offset);
        }

        return Fbcsr::create(exec,
                             gko::dim<2>{static_cast<size_type>(nrows),
                                         static_cast<size_type>(ncols)},
                             bs, vals, c, r);
    }
};


}  // namespace testing
}  // namespace gko

#endif  // GKO_CORE_TEST_MATRIX_FBCSR_SAMPLE_HPP_
