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

//#include "core/factorization/par_bilu_kernels.hpp"


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
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/block_factorization_kernels.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"
#include "reference/test/factorization/bilu_sample.hpp"


namespace {


class ParBilu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Bds = gko::testing::BlockDiagSample<value_type, index_type>;

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::unique_ptr<const Fbcsr> cyl2d_ref;
    std::unique_ptr<const Fbcsr> cyl2d_cuda;
    const value_type tol;

    ParBilu()
        : rand_engine(18),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          tol{std::numeric_limits<value_type>::epsilon()}
    // csr_ref(nullptr),
    // csr_cuda(nullptr)
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
        auto cuda_temp = Fbcsr::create(cuda);
        cuda_temp->copy_from(gko::lend(ref_temp));
        // Make sure there are diagonal elements present
        gko::kernels::reference::factorization::add_diagonal_blocks(
            ref, gko::lend(ref_temp), false);
        gko::kernels::cuda::factorization::add_diagonal_blocks(
            cuda, gko::lend(cuda_temp), false);
        cyl2d_ref = gko::give(ref_temp);
        cyl2d_cuda = gko::give(cuda_temp);
    }

    void initialize_row_ptrs(std::unique_ptr<const Fbcsr> mat_ref,
                             std::unique_ptr<const Fbcsr> mat_cuda,
                             index_type *l_row_ptrs_ref,
                             index_type *u_row_ptrs_ref,
                             index_type *l_row_ptrs_cuda,
                             index_type *u_row_ptrs_cuda)
    {
        gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
            ref, gko::lend(mat_ref), l_row_ptrs_ref, u_row_ptrs_ref);
        gko::kernels::cuda::factorization::initialize_row_ptrs_BLU(
            cuda, gko::lend(mat_cuda), l_row_ptrs_cuda, u_row_ptrs_cuda);
    }

    template <typename ToType, typename FromType>
    static std::unique_ptr<ToType> static_unique_ptr_cast(
        std::unique_ptr<FromType> &&from)
    {
        return std::unique_ptr<ToType>{static_cast<ToType *>(from.release())};
    }
};


TEST_F(ParBilu, CudaKernelAddDiagonalBlocksSortedStartingBlockMissing)
{
    Bds bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.gen_ref_1();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.gen_test_1();
    std::unique_ptr<Fbcsr> mtxstart_cuda = Fbcsr::create(cuda);
    mtxstart_cuda->copy_from(gko::lend(mtxstart_ref));
    ASSERT_EQ(mtxstart_ref->get_block_size(), mtxstart_cuda->get_block_size());
    ASSERT_EQ(mtxstart_cuda->get_block_size(), bds.bs);

    gko::kernels::cuda::factorization::add_diagonal_blocks(
        cuda, gko::lend(mtxstart_cuda), true);

    ASSERT_TRUE(mtxstart_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(mtxstart_cuda, answer_cuda);
    GKO_ASSERT_MTX_NEAR(mtxstart_cuda, answer_cuda, 0.);
}

TEST_F(ParBilu, CudaKernelAddDiagonalBlocksSortedEndingBlockMissing)
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

TEST_F(ParBilu, CudaKernelAddDiagonalBlocksSortedTwoBlocksMissing)
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

TEST_F(ParBilu, CudaKernelAddDiagonalBlocksUnsortedTwoBlocksMissing)
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

TEST_F(ParBilu, CudaKernelInitializeBLUSorted)
{
    const int bs = cyl2d_ref->get_block_size();
    const gko::size_type num_row_ptrs = cyl2d_ref->get_num_block_rows() + 1;
    gko::Array<index_type> l_row_ptrs_ref{ref, num_row_ptrs};
    gko::Array<index_type> u_row_ptrs_ref{ref, num_row_ptrs};
    gko::Array<index_type> l_row_ptrs_cuda{cuda, num_row_ptrs};
    gko::Array<index_type> u_row_ptrs_cuda{cuda, num_row_ptrs};

    gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
        ref, gko::lend(cyl2d_ref), l_row_ptrs_ref.get_data(),
        u_row_ptrs_ref.get_data());

    l_row_ptrs_cuda = l_row_ptrs_ref;
    u_row_ptrs_cuda = u_row_ptrs_ref;

    const index_type l_nnz = l_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];
    const index_type u_nnz = u_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];

    auto test_L =
        Fbcsr::create(cuda, cyl2d_cuda->get_size(), l_nnz * bs * bs, bs);
    auto ref_L = Fbcsr::create(ref, cyl2d_ref->get_size(), l_nnz * bs * bs, bs);
    auto test_U =
        Fbcsr::create(cuda, cyl2d_cuda->get_size(), u_nnz * bs * bs, bs);
    auto ref_U = Fbcsr::create(ref, cyl2d_ref->get_size(), u_nnz * bs * bs, bs);

    const bool issorted = cyl2d_ref->is_sorted_by_column_index();
    EXPECT_TRUE(issorted);

    ref->copy(num_row_ptrs, l_row_ptrs_ref.get_const_data(),
              ref_L->get_row_ptrs());
    ref->copy(num_row_ptrs, u_row_ptrs_ref.get_const_data(),
              ref_U->get_row_ptrs());
    cuda->copy(num_row_ptrs, l_row_ptrs_cuda.get_const_data(),
               test_L->get_row_ptrs());
    cuda->copy(num_row_ptrs, u_row_ptrs_cuda.get_const_data(),
               test_U->get_row_ptrs());

    gko::kernels::reference::factorization::initialize_BLU(
        ref, cyl2d_ref.get(), ref_L.get(), ref_U.get());
    gko::kernels::cuda::factorization::initialize_BLU(
        cuda, cyl2d_cuda.get(), test_L.get(), test_U.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(test_L, ref_L);
    GKO_ASSERT_MTX_EQ_SPARSITY(test_U, ref_U);
    GKO_ASSERT_MTX_NEAR(test_L, ref_L, tol);
    GKO_ASSERT_MTX_NEAR(test_U, ref_U, tol);
}


}  // namespace
