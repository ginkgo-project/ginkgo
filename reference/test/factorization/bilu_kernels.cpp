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


#include <gtest/gtest.h>
#include <cmath>
#include <limits>


#include "bilu_sample.hpp"
#include "core/factorization/bilu_kernels.hpp"
#include "core/factorization/block_factorization_kernels.hpp"
#include "core/test/utils.hpp"
#include "reference/components/fixed_block_operations.hpp"


namespace {


TEST(BiluSample, TriangularFactorsDimensionsAreCorrect)
{
    using value_type = double;
    using index_type = int;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    auto refexec = gko::ReferenceExecutor::create();
    gko::testing::Bilu0Sample<value_type, index_type> bilus(refexec);
    auto mtx = bilus.generate_fbcsr();
    const value_type *const testvals = mtx->get_values();
    const index_type *const testrowptrs = mtx->get_row_ptrs();
    const index_type *const testcolidxs = mtx->get_col_idxs();
    const index_type nbrows = mtx->get_num_block_rows();
    index_type testnbnzL{}, testnbnzU{};
    for (index_type ibrow = 0; ibrow < nbrows; ibrow++) {
        for (index_type ibz = testrowptrs[ibrow]; ibz < testrowptrs[ibrow + 1];
             ibz++) {
            const index_type jbcol = testcolidxs[ibz];
            if (jbcol < ibrow) {
                testnbnzL++;
            } else {
                testnbnzU++;
            }
        }
    }

    auto reffacts = bilus.generate_factors();
    const auto refL =
        std::dynamic_pointer_cast<const Fbcsr>(reffacts->get_operators()[0]);
    const auto refU =
        std::dynamic_pointer_cast<const Fbcsr>(reffacts->get_operators()[1]);

    ASSERT_TRUE(refL);
    ASSERT_TRUE(refU);
    ASSERT_EQ(refL->get_size(), mtx->get_size());
    ASSERT_EQ(refU->get_size(), mtx->get_size());
    ASSERT_EQ(testnbnzL + nbrows,
              refL->get_const_row_ptrs()[refL->get_num_block_rows()]);
    ASSERT_EQ(testnbnzU,
              refU->get_const_row_ptrs()[refU->get_num_block_rows()]);
}


template <typename ValueIndexType>
class Bilu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;

    Bilu()
        : refexec(gko::ReferenceExecutor::create()),
          bilusample(refexec),
          mtx(bilusample.generate_fbcsr()),
          reffacts(bilusample.generate_factors())
    {}

    std::shared_ptr<const gko::ReferenceExecutor> refexec;
    const gko::testing::Bilu0Sample<value_type, index_type> bilusample;
    std::unique_ptr<const Fbcsr> mtx;
    std::unique_ptr<const gko::Composition<value_type>> reffacts;
};

TYPED_TEST_SUITE(Bilu, gko::test::ValueIndexTypes);


TYPED_TEST(Bilu, FixedDenseBlock3InverseNoPivoting)
{
    using value_type = typename TestFixture::value_type;
    using Blk_t = gko::blockutils::FixedBlock<value_type, 3, 3>;
    Blk_t A;
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 4.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 2.0;
    A(2, 2) = 5.0;
    int perm[3];
    for (int i = 0; i < 3; i++) perm[i] = i;

    const bool invflag = gko::kernels::invert_block<value_type, 3>(perm, A);
    ASSERT_TRUE(invflag);
    gko::kernels::permute_block(A, perm);

    constexpr typename gko::remove_complex<value_type> eps =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();

    ASSERT_LE(std::abs(A(0, 0) - static_cast<value_type>(3.0 / 11)), eps);
    ASSERT_LE(std::abs(A(0, 1) - static_cast<value_type>(-1.0 / 22)), eps);
    ASSERT_LE(std::abs(A(0, 2) - static_cast<value_type>(-1.0 / 22)), eps);
    ASSERT_LE(std::abs(A(1, 0) - static_cast<value_type>(-5.0 / 33)), eps);
    ASSERT_LE(std::abs(A(1, 1) - static_cast<value_type>(10.0 / 33)), eps);
    ASSERT_LE(std::abs(A(1, 2) - static_cast<value_type>(-1.0 / 33)), eps);
    ASSERT_LE(std::abs(A(2, 0) - static_cast<value_type>(2.0 / 33)), eps);
    ASSERT_LE(std::abs(A(2, 1) - static_cast<value_type>(-4.0 / 33)), eps);
    ASSERT_LE(std::abs(A(2, 2) - static_cast<value_type>(7.0 / 33)), eps);
}

TEST(BlockInverse, FixedDenseBlock4InversePivoted)
{
    using value_type = float;
    constexpr int bs = 4;
    using Blk_t = gko::blockutils::FixedBlock<value_type, bs, bs>;
    Blk_t A;
    A(0, 0) = 1;
    A(0, 1) = 12;
    A(0, 2) = 3;
    A(0, 3) = 4;
    A(1, 0) = 5;
    A(1, 1) = 5;
    A(1, 2) = -2;
    A(1, 3) = 3;
    A(2, 0) = 2;
    A(2, 1) = -1;
    A(2, 2) = -11;
    A(2, 3) = 3;
    A(3, 0) = 4;
    A(3, 1) = 7;
    A(3, 2) = -6;
    A(3, 3) = 5;
    Blk_t B;
    B(0, 0) = 0.222222;
    B(0, 1) = 0.62963;
    B(0, 2) = 0.37037;
    B(0, 3) = -0.777778;
    B(1, 0) = 1.72222;
    B(1, 1) = 1.7963;
    B(1, 2) = 2.2037;
    B(1, 3) = -3.77778;
    B(2, 0) = -1.22222;
    B(2, 1) = -1.2963;
    B(2, 2) = -1.7037;
    B(2, 3) = 2.77778;
    B(3, 0) = -4.05556;
    B(3, 1) = -4.57407;
    B(3, 2) = -5.42593;
    B(3, 3) = 9.44444;
    int perm[bs];
    for (int i = 0; i < bs; i++) {
        perm[i] = i;
    }

    const bool invflag = gko::kernels::invert_block<value_type, 4>(perm, A);
    gko::kernels::permute_block(A, perm);

    ASSERT_TRUE(invflag);
    const value_type tol = 1e-4;
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            const value_type err = std::abs((A(i, j) - B(i, j)) / B(i, j));
            ASSERT_LE(err, tol);
        }
    }
}

TEST(BlockInverse, FixedDenseBlock7InversePivoted)
{
    using value_type = double;
    constexpr int bs = 7;
    using Blk_t = gko::blockutils::FixedBlock<value_type, bs, bs>;
    Blk_t A;
    A(0, 0) = -11.6651;
    A(0, 1) = -0.603807;
    A(0, 2) = 0.3396;
    A(1, 0) = -9.08963;
    A(1, 1) = -0.789592;
    A(1, 2) = 2.21999;
    A(2, 0) = -3.03475;
    A(2, 1) = 0.438925;
    A(2, 2) = -2.15868;
    A(3, 0) = 5.36474;
    A(3, 1) = 1.84708;
    A(3, 2) = 1.19565;
    A(4, 0) = 26.8763;
    A(4, 1) = 0.474329;
    A(4, 2) = 3.84057;
    A(5, 0) = 9.25151;
    A(5, 1) = 1.16839;
    A(5, 2) = -0.688332;
    A(6, 0) = -12.086;
    A(6, 1) = -1.81079;
    A(6, 2) = -2.95177;

    A(0, 3) = -5.71315;
    A(0, 4) = 0.833828;
    A(0, 5) = 14.6753;
    A(0, 6) = 0.792967;
    A(1, 3) = -2.5276;
    A(1, 4) = 0.85484;
    A(1, 5) = 7.95202;
    A(1, 6) = 1.08117;
    A(2, 3) = 0.339518;
    A(2, 4) = -2.14638;
    A(2, 5) = 4.54195;
    A(2, 6) = -0.103434;
    A(3, 3) = 2.12117;
    A(3, 4) = 0.875747;
    A(3, 5) = -8.16298;
    A(3, 6) = -0.767853;
    A(4, 3) = 9.32757;
    A(4, 4) = 1.33803;
    A(4, 5) = -32.779;
    A(4, 6) = -1.77482;
    A(5, 3) = 3.13859;
    A(5, 4) = 1.01449;
    A(5, 5) = -10.1447;
    A(5, 6) = -1.69143;
    A(6, 3) = -4.71304;
    A(6, 4) = -1.57097;
    A(6, 5) = 17.0216;
    A(6, 6) = 2.35194;

    Blk_t B;
    int perm[bs];
    for (int i = 0; i < bs; i++) {
        perm[i] = i;
        for (int j = 0; j < bs; j++) {
            B(i, j) = A(i, j);
        }
    }

    const bool invflag = gko::kernels::invert_block(perm, B);
    gko::kernels::permute_block(B, perm);

    ASSERT_TRUE(invflag);
    constexpr typename gko::remove_complex<value_type> tol =
        100 * std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    Blk_t prod;
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            prod(i, j) = 0;
            for (int k = 0; k < bs; k++) {
                prod(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            const value_type err = std::abs((A(i, j) - B(i, j)) / B(i, j));
            if (i == j) {
                ASSERT_LE(std::abs(prod(i, j) - 1), tol);
            } else {
                ASSERT_LE(std::abs(prod(i, j)), tol);
            }
        }
    }
}

TYPED_TEST(Bilu, KernelFactorizationSorted)
{
    using Fbcsr = typename TestFixture::Fbcsr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto mtxcopy = this->mtx->clone();
    const int bs = this->mtx->get_block_size();
    const value_type *const testvals = mtxcopy->get_values();
    const index_type *const testrowptrs = mtxcopy->get_row_ptrs();
    const index_type *const testcolidxs = mtxcopy->get_col_idxs();
    const index_type nbrows = mtxcopy->get_num_block_rows();
    const auto refL = std::dynamic_pointer_cast<const Fbcsr>(
        this->reffacts->get_operators()[0]);
    const auto refU = std::dynamic_pointer_cast<const Fbcsr>(
        this->reffacts->get_operators()[1]);

    gko::kernels::reference::bilu_factorization::compute_bilu(this->refexec,
                                                              mtxcopy.get());

    const value_type *const refLvals = refL->get_const_values();
    const index_type *const refLrowptrs = refL->get_const_row_ptrs();
    const index_type *const refLcolidxs = refL->get_const_col_idxs();
    const value_type *const refUvals = refU->get_const_values();
    const index_type *const refUrowptrs = refU->get_const_row_ptrs();
    const index_type *const refUcolidxs = refU->get_const_col_idxs();
    constexpr auto eps =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    index_type lbz = 0, ubz = 0;
    for (index_type ibrow = 0; ibrow < nbrows; ibrow++) {
        for (index_type ibz = testrowptrs[ibrow]; ibz < testrowptrs[ibrow + 1];
             ibz++) {
            const index_type jbcol = testcolidxs[ibz];

            if (jbcol < ibrow) {
                // skip any diagonal blocks, because
                //  test output does not contains diagonal blocks of L
                for (index_type lbrow = 0; lbrow < nbrows; lbrow++)
                    if (refLcolidxs[lbz] == lbrow) {
                        lbz++;
                    }
                // Find the max entry in the block
                auto maxit = std::max_element(
                    refLvals + lbz * bs * bs, refLvals + (lbz + 1) * bs * bs,
                    [](value_type a, value_type b) {
                        return std::abs(a) < std::abs(b);
                    });
                const auto maxel = std::abs(*maxit);
                for (int i = 0; i < bs * bs; i++) {
                    const value_type ts = testvals[ibz * bs * bs + i];
                    const value_type rf = refLvals[lbz * bs * bs + i];
                    ASSERT_LE(std::abs(ts - rf), 2.0 * eps * maxel);
                }
                lbz++;
            } else {
                auto maxit = std::max_element(
                    refUvals + ubz * bs * bs, refUvals + (ubz + 1) * bs * bs,
                    [](value_type a, value_type b) {
                        return std::abs(a) < std::abs(b);
                    });
                const auto maxel = std::abs(*maxit);
                for (int i = 0; i < bs * bs; i++) {
                    const value_type ts = testvals[ibz * bs * bs + i];
                    const value_type rf = refUvals[ubz * bs * bs + i];
                    ASSERT_LE(std::abs(ts - rf), 2.0 * eps * maxel);
                }
                ubz++;
            }
        }
    }
}

}  // namespace
