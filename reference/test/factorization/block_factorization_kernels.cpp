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
#include "core/factorization/block_factorization_kernels.hpp"
#include "core/test/utils.hpp"
#include "reference/components/fixed_block_operations.hpp"


namespace {

template <typename IndexType>
class BlockFactorizationRowPtrs : public ::testing::Test {
protected:
    using value_type = std::complex<float>;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;

    BlockFactorizationRowPtrs()
        : refexec(gko::ReferenceExecutor::create()), sample(refexec)
    {}

    std::shared_ptr<const gko::ReferenceExecutor> refexec;
    gko::testing::Bilu0Sample<value_type, index_type> sample;
};

TYPED_TEST_SUITE(BlockFactorizationRowPtrs, gko::test::IndexTypes);


TYPED_TEST(BlockFactorizationRowPtrs, KernelInitializeRowPtrsLUSorted)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const auto origmat = this->sample.generate_fbcsr();
    const auto factors = this->sample.generate_initial_values();
    std::vector<index_type> l_row_ptrs(this->sample.nbrows + 1);
    std::vector<index_type> u_row_ptrs(this->sample.nbrows + 1);

    gko::kernels::reference::factorization ::initialize_row_ptrs_BLU(
        this->refexec, origmat.get(), l_row_ptrs.data(), u_row_ptrs.data());

    const index_type *const ref_l_row_ptrs =
        gko::as<Fbcsr>(factors->get_operators()[0])->get_const_row_ptrs();
    const index_type *const ref_u_row_ptrs =
        gko::as<Fbcsr>(factors->get_operators()[1])->get_const_row_ptrs();
    for (index_type i = 0; i <= this->sample.nbrows; i++) {
        ASSERT_EQ(ref_l_row_ptrs[i], l_row_ptrs[i]);
        ASSERT_EQ(ref_u_row_ptrs[i], u_row_ptrs[i]);
    }
}

TYPED_TEST(BlockFactorizationRowPtrs, KernelInitializeRowPtrsLUUnsorted)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const auto origmat = this->sample.generate_unsorted_fbcsr();
    const auto factors = this->sample.generate_factors();
    std::vector<index_type> l_row_ptrs(this->sample.nbrows + 1);
    std::vector<index_type> u_row_ptrs(this->sample.nbrows + 1);

    gko::kernels::reference::factorization ::initialize_row_ptrs_BLU(
        this->refexec, origmat.get(), l_row_ptrs.data(), u_row_ptrs.data());

    const index_type *const ref_l_row_ptrs =
        gko::as<Fbcsr>(factors->get_operators()[0])->get_const_row_ptrs();
    const index_type *const ref_u_row_ptrs =
        gko::as<Fbcsr>(factors->get_operators()[1])->get_const_row_ptrs();
    for (index_type i = 0; i <= this->sample.nbrows; i++) {
        ASSERT_EQ(ref_l_row_ptrs[i], l_row_ptrs[i]);
        ASSERT_EQ(ref_u_row_ptrs[i], u_row_ptrs[i]);
    }
}

template <typename ValueIndexType>
class BlockFactorization : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Composition = gko::Composition<value_type>;

    BlockFactorization()
        : refexec(gko::ReferenceExecutor::create()), sample(refexec)
    {}

    std::shared_ptr<const gko::ReferenceExecutor> refexec;
    gko::testing::Bilu0Sample<value_type, index_type> sample;
};

TYPED_TEST_SUITE(BlockFactorization, gko::test::ValueIndexTypes);


TYPED_TEST(BlockFactorization, KernelAddDiagonalElementsEmpty)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const gko::dim<2> size{4, 4};
    const int bs = 2;
    auto empty_mtx = Fbcsr::create(this->refexec, size, bs,
                                   std::initializer_list<value_type>{},
                                   std::initializer_list<index_type>{},
                                   std::initializer_list<index_type>{0, 0, 0});
    const auto expected_mtx = Fbcsr::create(
        this->refexec, size, bs,
        std::initializer_list<value_type>{0., 0., 0., 0., 0., 0., 0., 0.},
        std::initializer_list<index_type>{0, 1},
        std::initializer_list<index_type>{0, 1, 2});

    gko::kernels::reference::factorization::add_diagonal_blocks(
        this->refexec, gko::lend(empty_mtx), true);

    GKO_ASSERT_MTX_NEAR(empty_mtx, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(empty_mtx, expected_mtx);
}


TYPED_TEST(BlockFactorization, KernelAddDiagonalElementsNonSquare)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const gko::dim<2> size{9, 6};
    const int bs = 3;
    auto test_mtx = Fbcsr::create(
        this->refexec, size, bs,
        std::initializer_list<value_type>{1., 2., 1., 2., 1., 2., 3., 1., 2.,
                                          0., 2., 3., 1., 2., 3., 6., 7., 8.},
        std::initializer_list<index_type>{0, 1},
        std::initializer_list<index_type>{0, 1, 1, 2});
    const auto expected_mtx = Fbcsr::create(
        this->refexec, size, bs,
        std::initializer_list<value_type>{1., 2., 1., 2., 1., 2., 3., 1., 2.,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 2., 3., 1., 2., 3., 6., 7., 8.},
        std::initializer_list<index_type>{0, 1, 1},
        std::initializer_list<index_type>{0, 1, 2, 3});

    gko::kernels::reference::factorization::add_diagonal_blocks(
        this->refexec, gko::lend(test_mtx), true);

    GKO_ASSERT_MTX_NEAR(test_mtx, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(test_mtx, expected_mtx);
}


TYPED_TEST(BlockFactorization, KernelAddDiagonalElementsNonSquareUnsorted)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const gko::dim<2> size{6, 8};
    const int bs = 2;
    auto test_mtx =
        Fbcsr::create(this->refexec, size, bs,
                      std::initializer_list<value_type>{
                          1., 2., 1., 7., 1., 2., 3., 1., 2., 0., 2., 3.,
                          1., 2., 3., 6., 7., 8., 1., 2., 3., 4., 1., 9.},
                      std::initializer_list<index_type>{0, 2, 0, 3, 2, 1},
                      std::initializer_list<index_type>{0, 1, 4, 6});
    const auto expected_mtx = Fbcsr::create(
        this->refexec, size, bs,
        std::initializer_list<value_type>{
            1., 2., 1., 7., 0., 0., 0., 0., 1., 2., 3., 1., 2., 0.,
            2., 3., 1., 2., 3., 6., 7., 8., 1., 2., 3., 4., 1., 9.},
        std::initializer_list<index_type>{0, 1, 2, 0, 3, 2, 1},
        std::initializer_list<index_type>{0, 1, 5, 7});

    gko::kernels::reference::factorization::add_diagonal_blocks(
        this->refexec, gko::lend(test_mtx), true);

    GKO_ASSERT_MTX_NEAR(test_mtx, expected_mtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(test_mtx, expected_mtx);
}

TYPED_TEST(BlockFactorization, KernelInitializeBLUSorted)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Fbcsr;

    std::unique_ptr<const Fbcsr> matrixx = this->sample.generate_fbcsr();
    const auto reflu = this->sample.generate_initial_values();
    const int bs = this->sample.bs;

    auto refL = gko::as<Fbcsr>(reflu->get_operators()[0]);
    auto refU = gko::as<Fbcsr>(reflu->get_operators()[1]);

    auto testL = refL->clone();
    auto testU = refU->clone();
    const index_type testL_nnz = testL->get_num_stored_elements();
    const index_type testU_nnz = testU->get_num_stored_elements();
    index_type *const testL_colidx = testL->get_col_idxs();
    value_type *const testL_vals = testL->get_values();
    index_type *const testU_colidx = testU->get_col_idxs();
    value_type *const testU_vals = testU->get_values();

    for (index_type i = 0; i < testL_nnz; i++) testL_vals[i] = 0.0;
    for (index_type i = 0; i < testU_nnz; i++) testU_vals[i] = 0.0;
    for (index_type i = 0; i < testL_nnz / bs / bs; i++) testL_colidx[i] = -1;
    for (index_type i = 0; i < testU_nnz / bs / bs; i++) testU_colidx[i] = -1;

    gko::kernels::reference::factorization::initialize_BLU(
        this->refexec, matrixx.get(), testL.get(), testU.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(refL, testL);
    GKO_ASSERT_MTX_NEAR(refL, testL, 0.0);
    GKO_ASSERT_MTX_EQ_SPARSITY(refU, testU);
    GKO_ASSERT_MTX_NEAR(refU, testU, 0.0);
}

}  // namespace
