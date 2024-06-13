// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/fb_matrix_generator.hpp"


#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>


#include <gtest/gtest.h>


#include "accessor/block_col_major.hpp"
#include "core/base/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"


namespace {


class BlockMatrixGenerator : public ::testing::Test {
protected:
    using real_type = double;
    using value_type = std::complex<real_type>;

    BlockMatrixGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix<
              gko::matrix::Csr<real_type, int>>(
              nbrows, nbcols, std::normal_distribution<real_type>(10, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          rbmtx(gko::test::generate_fbcsr_from_csr(
              exec, mtx.get(), blk_sz, false, std::default_random_engine(42))),
          rbmtx_dd(gko::test::generate_fbcsr_from_csr(
              exec, mtx.get(), blk_sz, true, std::default_random_engine(42))),
          cbmtx(gko::test::generate_random_fbcsr<value_type>(
              exec, nbrows, nbcols, blk_sz, true, false,
              std::default_random_engine(42)))
    {}

    const int nbrows = 100;
    const int nbcols = nbrows;
    const int blk_sz = 5;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<gko::matrix::Csr<real_type, int>> mtx;
    std::unique_ptr<gko::matrix::Fbcsr<real_type, int>> rbmtx;
    std::unique_ptr<gko::matrix::Fbcsr<real_type, int>> rbmtx_dd;
    std::unique_ptr<gko::matrix::Fbcsr<std::complex<real_type>, int>> cbmtx;

    template <typename InputIterator, typename ValueType>
    ValueType get_nth_moment(int n, ValueType c, InputIterator sample_start,
                             InputIterator sample_end)
    {
        using std::pow;
        ValueType res = 0;
        ValueType num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(tmp - c, n);
            num_elems += 1;
        }
        return res / num_elems;
    }
};


TEST_F(BlockMatrixGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(rbmtx->get_size(), gko::dim<2>(nbrows * blk_sz, nbcols * blk_sz));
    ASSERT_EQ(rbmtx_dd->get_size(),
              gko::dim<2>(nbrows * blk_sz, nbcols * blk_sz));
    ASSERT_EQ(cbmtx->get_size(), gko::dim<2>(nbrows * blk_sz, nbcols * blk_sz));
    ASSERT_EQ(rbmtx->get_block_size(), blk_sz);
    ASSERT_EQ(rbmtx_dd->get_block_size(), blk_sz);
    ASSERT_EQ(cbmtx->get_block_size(), blk_sz);
}


TEST_F(BlockMatrixGenerator, OutputHasCorrectSparsityPattern)
{
    ASSERT_EQ(mtx->get_num_stored_elements(),
              rbmtx->get_num_stored_elements() / blk_sz / blk_sz);
    for (int irow = 0; irow < nbrows; irow++) {
        const int start = mtx->get_const_row_ptrs()[irow];
        const int end = mtx->get_const_row_ptrs()[irow + 1];
        ASSERT_EQ(start, rbmtx->get_const_row_ptrs()[irow]);
        ASSERT_EQ(end, rbmtx->get_const_row_ptrs()[irow + 1]);
        for (int iz = start; iz < end; iz++) {
            ASSERT_EQ(mtx->get_const_col_idxs()[iz],
                      rbmtx->get_const_col_idxs()[iz]);
        }
    }
}


TEST_F(BlockMatrixGenerator, ComplexOutputIsRowDiagonalDominantWhenRequested)
{
    using Dbv_t =
        gko::acc::range<gko::acc::block_col_major<const value_type, 3>>;
    const auto nbnz = cbmtx->get_num_stored_blocks();
    const Dbv_t vals(
        gko::to_std_array<gko::acc::size_type>(nbnz, blk_sz, blk_sz),
        cbmtx->get_const_values());
    const int* const row_ptrs = cbmtx->get_const_row_ptrs();
    const int* const col_idxs = cbmtx->get_const_col_idxs();

    for (int irow = 0; irow < nbrows; irow++) {
        std::vector<real_type> row_del_sum(blk_sz, 0.0);
        std::vector<real_type> diag_val(blk_sz, 0.0);
        bool diagfound{false};
        for (int iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
            if (col_idxs[iz] == irow) {
                diagfound = true;
                for (int i = 0; i < blk_sz; i++) {
                    for (int j = 0; j < blk_sz; j++) {
                        if (i == j) {
                            diag_val[i] = abs(vals(iz, i, i));
                        } else {
                            row_del_sum[i] += abs(vals(iz, i, j));
                        }
                    }
                }
            } else {
                for (int i = 0; i < blk_sz; i++) {
                    for (int j = 0; j < blk_sz; j++) {
                        row_del_sum[i] += abs(vals(iz, i, j));
                    }
                }
            }
        }
        std::vector<real_type> diag_dom(blk_sz);
        for (int i = 0; i < blk_sz; i++) {
            diag_dom[i] = diag_val[i] - row_del_sum[i];
        }

        ASSERT_TRUE(diagfound);
        for (int i = 0; i < blk_sz; i++) {
            ASSERT_GT(diag_val[i], row_del_sum[i]);
        }
    }
}


}  // namespace
