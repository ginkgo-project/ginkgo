/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include "core/test/utils/fb_matrix_generator.hpp"


#include <cmath>
#include <iostream>
#include <random>


#include <gtest/gtest.h>


#include "core/components/fixed_block.hpp"
#include "core/test/utils/matrix_generator.hpp"


namespace {


class BlockMatrixGenerator : public ::testing::Test {
protected:
    using real_type = float;

    BlockMatrixGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix<
              gko::matrix::Csr<real_type, int>>(
              nbrows, nbcols, std::normal_distribution<real_type>(10, 5),
              std::normal_distribution<real_type>(20.0, 5.0), std::ranlux48(42),
              exec)),
          rbmtx(gko::test::generate_fbcsr_from_csr(exec, mtx.get(), blk_sz,
                                                   false, std::ranlux48(42))),
          rbmtx_dd(gko::test::generate_fbcsr_from_csr(exec, mtx.get(), blk_sz,
                                                      true, std::ranlux48(42)))
    {
        const int nnz = mtx->get_num_stored_elements();
        auto cmtx = gko::matrix::Csr<std::complex<float>, int>::create(
            exec, mtx->get_size(), nnz);
        exec->copy(mtx->get_size()[0] + 1, mtx->get_const_row_ptrs(),
                   cmtx->get_row_ptrs());
        exec->copy(nnz, mtx->get_const_col_idxs(), cmtx->get_col_idxs());
        std::copy(mtx->get_const_values(), mtx->get_const_values() + nnz,
                  cmtx->get_values());
        cbmtx = gko::test::generate_fbcsr_from_csr(exec, cmtx.get(), blk_sz,
                                                   false, std::ranlux48(42));
    }

    const int nbrows = 100;
    const int nbcols = 20;
    const int blk_sz = 5;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<gko::matrix::Csr<real_type, int>> mtx;
    std::unique_ptr<gko::matrix::Fbcsr<real_type, int>> rbmtx;
    std::unique_ptr<gko::matrix::Fbcsr<real_type, int>> rbmtx_dd;
    std::unique_ptr<gko::matrix::Fbcsr<std::complex<real_type>, int>> cbmtx;

    // TODO: Move this function out and use for both classes in this file
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

TEST_F(BlockMatrixGenerator, OutputIsRowDiagonalDominantWhenRequested)
{
    using Dbv_t = gko::blockutils::DenseBlocksView<const real_type, int>;
    const Dbv_t vals(rbmtx_dd->get_const_values(), blk_sz, blk_sz);
    const int *const row_ptrs = rbmtx_dd->get_const_row_ptrs();
    const int *const col_idxs = rbmtx_dd->get_const_col_idxs();

    for (int irow = 0; irow < nbrows; irow++) {
        std::vector<real_type> row_del_sum(blk_sz, 0.0);
        std::vector<real_type> diag_val(blk_sz, 0.0);
        for (int iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
            if (col_idxs[iz] == irow) {
                std::cout << "Found diag block";
                for (int i = 0; i < blk_sz; i++)
                    for (int j = 0; j < blk_sz; j++)
                        if (i == j) {
                            std::cout << "Found diag ";
                            diag_val[i] = abs(vals(iz, i, i));
                        } else
                            row_del_sum[i] += abs(vals(iz, i, j));
            } else {
                for (int i = 0; i < blk_sz; i++)
                    for (int j = 0; j < blk_sz; j++)
                        row_del_sum[i] += abs(vals(iz, i, j));
            }
        }

        for (int i = 0; i < blk_sz; i++) {
            if (diag_val[i] <= row_del_sum[i])
                std::cout << "Blk-row " << irow << ": row " << i
                          << ": diag val = " << diag_val[i]
                          << ", off-diag sum = " << row_del_sum[i] << std::endl;
            ASSERT_TRUE(diag_val[i] > row_del_sum[i]);
        }
    }
}

// TEST_F(BlockMatrixGenerator, OutputHasCorrectNonzeroAverageAndDeviation)
// {
//     using std::sqrt;
//     auto average = get_nth_moment(1, 0.0, begin(nnz_per_row_sample),
//                                   end(nnz_per_row_sample));
//     auto deviation = sqrt(get_nth_moment(2, average,
//     begin(nnz_per_row_sample),
//                                          end(nnz_per_row_sample)));

//     // check that average & deviation is within 10% of the required amount
//     ASSERT_NEAR(average, 50.0, 5);
//     ASSERT_NEAR(deviation, 5.0, 0.5);
// }


// TEST_F(BlockMatrixGenerator, OutputHasCorrectValuesAverageAndDeviation)
// {
//     using std::sqrt;
//     auto average =
//         get_nth_moment(1, 0.0, begin(values_sample), end(values_sample));
//     auto deviation = sqrt(
//         get_nth_moment(2, average, begin(values_sample),
//         end(values_sample)));

//     // check that average and deviation is within 10% of the required amount
//     ASSERT_NEAR(average, 20.0, 2.0);
//     ASSERT_NEAR(deviation, 5.0, 0.5);
// }


}  // namespace
