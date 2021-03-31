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

#include "core/factorization/block_factorization_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"
#include "reference/test/factorization/bilu_sample.hpp"


namespace {


class BlockFactor : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using real_type = gko::remove_complex<value_type>;
    using index_type = gko::int32;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Bds = gko::testing::BlockDiagSample<value_type, index_type>;
    using BILUSample = gko::testing::Bilu0Sample<value_type, index_type>;

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::unique_ptr<const Fbcsr> cyl2d_ref;
    // std::unique_ptr<const Fbcsr> cyl2d_cuda;
    std::unique_ptr<const Fbcsr> rand_ref;
    std::unique_ptr<const Fbcsr> rand_unsrt_ref;
    const value_type tol;

    BlockFactor()
        : rand_engine(18),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          tol{std::numeric_limits<value_type>::epsilon()}
    {}

    void SetUp() override
    {
        std::string file_name(gko::matrices::location_2dcyl1_prefix);
        file_name += ".mtx";
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        const int block_size = 4;
        auto ref_temp = gko::read<Fbcsr>(input_file, ref, block_size);
        input_file.close();
        // Make sure there are diagonal elements present
        gko::kernels::reference::factorization::add_diagonal_blocks(
            ref, gko::lend(ref_temp), false);
        cyl2d_ref = gko::give(ref_temp);

        const index_type rand_dim = 200;
        std::unique_ptr<Csr> rand_csr_ref =
            gko::test::generate_random_matrix<Csr>(
                rand_dim, rand_dim,
                std::uniform_int_distribution<index_type>(0, rand_dim - 1),
                std::normal_distribution<real_type>(0.0, 1.0),
                std::ranlux48(47), ref);
        gko::kernels::reference::factorization::add_diagonal_elements(
            ref, gko::lend(rand_csr_ref), false);
        auto rand_ref_temp = gko::test::generate_fbcsr_from_csr(
            ref, rand_csr_ref.get(), 3, false, std::ranlux48(43));
        rand_ref = gko::give(rand_ref_temp);

        auto rand_unsrt_csr_ref = Csr::create(ref);
        rand_unsrt_csr_ref->copy_from(rand_csr_ref.get());
        if (rand_unsrt_csr_ref->is_sorted_by_column_index()) {
            gko::test::unsort_matrix(rand_unsrt_csr_ref.get(),
                                     std::ranlux48(43));
        }
        auto rand_unsrt_ref_temp = gko::test::generate_fbcsr_from_csr(
            ref, rand_unsrt_csr_ref.get(), 3, false, std::ranlux48(43));
        rand_unsrt_ref = gko::give(rand_unsrt_ref_temp);
    }

    void test_initializeBLU(const Fbcsr *const a_ref)
    {
        auto a_cuda = Fbcsr::create(cuda);
        a_cuda->copy_from(gko::lend(a_ref));

        const int bs = a_ref->get_block_size();
        const gko::size_type num_row_ptrs = a_ref->get_num_block_rows() + 1;
        gko::Array<index_type> l_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> l_row_ptrs_cuda{cuda, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_cuda{cuda, num_row_ptrs};

        gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
            ref, gko::lend(a_ref), l_row_ptrs_ref.get_data(),
            u_row_ptrs_ref.get_data());

        l_row_ptrs_cuda = l_row_ptrs_ref;
        u_row_ptrs_cuda = u_row_ptrs_ref;

        const index_type l_nnz =
            l_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];
        const index_type u_nnz =
            u_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];

        auto test_L =
            Fbcsr::create(cuda, a_cuda->get_size(), l_nnz * bs * bs, bs);
        auto ref_L = Fbcsr::create(ref, a_ref->get_size(), l_nnz * bs * bs, bs);
        auto test_U =
            Fbcsr::create(cuda, a_cuda->get_size(), u_nnz * bs * bs, bs);
        auto ref_U = Fbcsr::create(ref, a_ref->get_size(), u_nnz * bs * bs, bs);

        ref->copy(num_row_ptrs, l_row_ptrs_ref.get_const_data(),
                  ref_L->get_row_ptrs());
        ref->copy(num_row_ptrs, u_row_ptrs_ref.get_const_data(),
                  ref_U->get_row_ptrs());
        cuda->copy(num_row_ptrs, l_row_ptrs_cuda.get_const_data(),
                   test_L->get_row_ptrs());
        cuda->copy(num_row_ptrs, u_row_ptrs_cuda.get_const_data(),
                   test_U->get_row_ptrs());

        gko::kernels::reference::factorization::initialize_BLU(
            ref, a_ref, ref_L.get(), ref_U.get());
        gko::kernels::cuda::factorization::initialize_BLU(
            cuda, a_cuda.get(), test_L.get(), test_U.get());

        GKO_ASSERT_MTX_EQ_SPARSITY(test_L, ref_L);
        GKO_ASSERT_MTX_EQ_SPARSITY(test_U, ref_U);
        GKO_ASSERT_MTX_NEAR(test_L, ref_L, tol);
        GKO_ASSERT_MTX_NEAR(test_U, ref_U, tol);
    }
};


TEST_F(BlockFactor, CudaKernelAddDiagonalBlocksSortedStartingBlockMissing)
{
    Bds bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.gen_ref_1();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.gen_test_1();
    std::unique_ptr<Fbcsr> mtxstart_cuda = Fbcsr::create(cuda);
    mtxstart_cuda->copy_from(gko::lend(mtxstart_ref));

    gko::kernels::cuda::factorization::add_diagonal_blocks(
        cuda, gko::lend(mtxstart_cuda), true);

    ASSERT_TRUE(mtxstart_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(mtxstart_cuda, answer_cuda);
    GKO_ASSERT_MTX_NEAR(mtxstart_cuda, answer_cuda, 0.);
}

TEST_F(BlockFactor, CudaKernelAddDiagonalBlocksSortedEndingBlockMissing)
{
    Bds bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.gen_ref_lastblock();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.gen_test_lastblock();
    std::unique_ptr<Fbcsr> mtxstart_cuda = Fbcsr::create(cuda);
    mtxstart_cuda->copy_from(gko::lend(mtxstart_ref));
    ASSERT_EQ(mtxstart_ref->get_block_size(), mtxstart_cuda->get_block_size());

    gko::kernels::cuda::factorization::add_diagonal_blocks(
        cuda, gko::lend(mtxstart_cuda), true);

    ASSERT_TRUE(mtxstart_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(mtxstart_cuda, answer_cuda);
    GKO_ASSERT_MTX_NEAR(mtxstart_cuda, answer_cuda, 0.);
}

TEST_F(BlockFactor, CudaKernelAddDiagonalBlocksSortedTwoBlocksMissing)
{
    Bds bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.gen_ref_2();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.gen_test_2();
    std::unique_ptr<Fbcsr> mtxstart_cuda = Fbcsr::create(cuda);
    mtxstart_cuda->copy_from(gko::lend(mtxstart_ref));
    ASSERT_EQ(mtxstart_ref->get_block_size(), mtxstart_cuda->get_block_size());

    gko::kernels::cuda::factorization::add_diagonal_blocks(
        cuda, gko::lend(mtxstart_cuda), true);

    ASSERT_TRUE(mtxstart_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(mtxstart_cuda, answer_cuda);
    GKO_ASSERT_MTX_NEAR(mtxstart_cuda, answer_cuda, 0.);
}

TEST_F(BlockFactor, CudaKernelAddDiagonalBlocksUnsortedTwoBlocksMissing)
{
    Bds bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.gen_ref_2_unsorted();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.gen_test_2_unsorted();
    std::unique_ptr<Fbcsr> mtxstart_cuda = Fbcsr::create(cuda);
    mtxstart_cuda->copy_from(gko::lend(mtxstart_ref));
    ASSERT_EQ(mtxstart_ref->get_block_size(), mtxstart_cuda->get_block_size());

    gko::kernels::cuda::factorization::add_diagonal_blocks(
        cuda, gko::lend(mtxstart_cuda), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(mtxstart_cuda, answer_cuda);
    GKO_ASSERT_MTX_NEAR(mtxstart_cuda, answer_cuda, 0.);
}


TEST_F(BlockFactor, CudaKernelInitializeRowPtrsBLUUnsorted)
{
    auto a_cuda = Fbcsr::create(cuda);
    a_cuda->copy_from(gko::lend(rand_unsrt_ref));
    const gko::size_type nbrowsp1 = rand_unsrt_ref->get_num_block_rows() + 1;
    gko::Array<index_type> l_row_ptrs(ref, nbrowsp1);
    gko::Array<index_type> u_row_ptrs(ref, nbrowsp1);
    gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
        ref, rand_unsrt_ref.get(), l_row_ptrs.get_data(),
        u_row_ptrs.get_data());

    gko::Array<index_type> d_test_l_row_ptrs(cuda, nbrowsp1);
    gko::Array<index_type> d_test_u_row_ptrs(cuda, nbrowsp1);
    gko::kernels::cuda::factorization::initialize_row_ptrs_BLU(
        cuda, a_cuda.get(), d_test_l_row_ptrs.get_data(),
        d_test_u_row_ptrs.get_data());

    gko::Array<index_type> test_l_row_ptrs(ref, d_test_l_row_ptrs);
    gko::Array<index_type> test_u_row_ptrs(ref, d_test_u_row_ptrs);
    for (index_type i = 0; i < nbrowsp1; i++) {
        ASSERT_EQ(l_row_ptrs.get_const_data()[i],
                  test_l_row_ptrs.get_const_data()[i]);
        ASSERT_EQ(u_row_ptrs.get_const_data()[i],
                  test_u_row_ptrs.get_const_data()[i]);
    }
}


TEST_F(BlockFactor, CudaKernelInitializeBLUSorted4)
{
    EXPECT_TRUE(cyl2d_ref->is_sorted_by_column_index());
    test_initializeBLU(cyl2d_ref.get());
}


TEST_F(BlockFactor, CudaKernelInitializeBLUSortedRandom3)
{
    test_initializeBLU(rand_ref.get());
}


}  // namespace
