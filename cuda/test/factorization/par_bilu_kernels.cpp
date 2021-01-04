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

//#include "core/factorization/par_bilu_kernels.hpp"


#include <algorithm>
#include <fstream>
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

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;

    ParBilu()
        : rand_engine(18),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref))
    // csr_ref(nullptr),
    // csr_cuda(nullptr)
    {}

    // void SetUp() override
    // {
    //     std::string file_name(gko::matrices::location_ani4_mtx);
    //     auto input_file = std::ifstream(file_name, std::ios::in);
    //     if (!input_file) {
    //         FAIL() << "Could not find the file \"" << file_name
    //                << "\", which is required for this test.\n";
    //     }
    //     auto csr_ref_temp = gko::read<Csr>(input_file, ref);
    //     auto csr_cuda_temp = Csr::create(cuda);
    //     csr_cuda_temp->copy_from(gko::lend(csr_ref_temp));
    //     // Make sure there are diagonal elements present
    //     gko::kernels::reference::factorization::add_diagonal_elements(
    //         ref, gko::lend(csr_ref_temp), false);
    //     gko::kernels::cuda::factorization::add_diagonal_elements(
    //         cuda, gko::lend(csr_cuda_temp), false);
    //     csr_ref = gko::give(csr_ref_temp);
    //     csr_cuda = gko::give(csr_cuda_temp);
    // }

    template <typename Mtx>
    std::unique_ptr<Mtx> gen_mtx(index_type num_rows, index_type num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(0, num_cols - 1),
            std::normal_distribution<value_type>(0.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Fbcsr> gen_unsorted_mtx(const index_type num_rows,
                                            const index_type num_cols)
    {
        using std::swap;
        auto mtx = gen_mtx<Fbcsr>(num_rows, num_cols);
        auto values = mtx->get_values();
        auto col_idxs = mtx->get_col_idxs();
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const int bs = mtx->get_block_size();
        for (int row = 0; row < num_rows; ++row) {
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const int num_row_elements = row_end - row_start;
            auto idx_dist = std::uniform_int_distribution<index_type>(
                row_start, row_end - 1);
            for (int i = 0; i < num_row_elements / 2; ++i) {
                auto idx1 = idx_dist(rand_engine);
                auto idx2 = idx_dist(rand_engine);
                if (idx1 != idx2) {
                    for (int i = 0; i < bs * bs; i++)
                        swap(values[idx1 * bs * bs + i],
                             values[idx2 * bs * bs + i]);
                    swap(col_idxs[idx1], col_idxs[idx2]);
                }
            }
        }
        return mtx;
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


TEST_F(ParBilu, CudaKernelAddDiagonalBlocksSorted1)
{
    gko::testing::BlockDiagSample<value_type, index_type> bds(ref);
    std::unique_ptr<const Fbcsr> answer_ref = bds.generate_ref_1();
    std::unique_ptr<Fbcsr> answer_cuda = Fbcsr::create(cuda);
    answer_cuda->copy_from(gko::lend(answer_ref));
    auto mtxstart_ref = bds.generate_test_1();
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


}  // namespace
