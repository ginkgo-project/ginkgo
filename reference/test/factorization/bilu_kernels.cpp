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


#include <gtest/gtest.h>
#include <cmath>
#include <limits>


#include "bilu_sample.hpp"
#include "core/factorization/bilu_kernels.hpp"
#include "core/factorization/block_factorization_kernels.hpp"
#include "core/test/utils.hpp"
#include "reference/components/fixed_block_operations.hpp"


namespace {

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


TYPED_TEST(Bilu, FixedDenseBlockInverseNoPivoting)
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

TYPED_TEST(Bilu, KernelFactorizationSorted)
{
    using Fbcsr = typename TestFixture::Fbcsr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    auto mtxcopy = this->mtx->clone();
    gko::kernels::reference::bilu_factorization::compute_bilu(this->refexec,
                                                              mtxcopy.get());

    const auto refL = std::dynamic_pointer_cast<const Fbcsr>(
        this->reffacts->get_operators()[0]);
    const auto refU = std::dynamic_pointer_cast<const Fbcsr>(
        this->reffacts->get_operators()[1]);

    ASSERT_TRUE(refL);
    ASSERT_TRUE(refU);

    ASSERT_EQ(refL->get_size(), mtxcopy->get_size());
    ASSERT_EQ(refU->get_size(), mtxcopy->get_size());

    const int bs = this->mtx->get_block_size();
    const value_type *const testvals = mtxcopy->get_values();
    const index_type *const testrowptrs = mtxcopy->get_row_ptrs();
    const index_type *const testcolidxs = mtxcopy->get_col_idxs();
    const index_type nbrows = mtxcopy->get_num_block_rows();

    index_type testnbnzL{}, testnbnzU{};
    for (index_type ibrow = 0; ibrow < nbrows; ibrow++) {
        for (index_type ibz = testrowptrs[ibrow]; ibz < testrowptrs[ibrow + 1];
             ibz++) {
            const index_type jbcol = testcolidxs[ibz];
            if (jbcol < ibrow)
                testnbnzL++;
            else
                testnbnzU++;
        }
    }

    const value_type *const refLvals = refL->get_values();
    const index_type *const refLrowptrs = refL->get_row_ptrs();
    const index_type *const refLcolidxs = refL->get_col_idxs();
    const value_type *const refUvals = refU->get_values();
    const index_type *const refUrowptrs = refU->get_row_ptrs();
    const index_type *const refUcolidxs = refU->get_col_idxs();

    ASSERT_EQ(testnbnzL + nbrows,
              refL->get_row_ptrs()[refL->get_num_block_rows()]);
    ASSERT_EQ(testnbnzU, refU->get_row_ptrs()[refU->get_num_block_rows()]);

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
