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

#include "core/factorization/par_ilu_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


class ParIlu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::shared_ptr<const Csr> csr_ref;
    std::shared_ptr<const Csr> csr_cuda;

    ParIlu()
        : rand_engine(18),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          csr_ref(nullptr),
          csr_cuda(nullptr)
    {}

    void SetUp() override
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        auto csr_ref_temp = gko::read<Csr>(input_file, ref);
        auto csr_cuda_temp = Csr::create(cuda);
        csr_cuda_temp->copy_from(gko::lend(csr_ref_temp));
        // Make sure there are diagonal elements present
        gko::kernels::reference::factorization::add_diagonal_elements(
            ref, gko::lend(csr_ref_temp), false);
        gko::kernels::cuda::factorization::add_diagonal_elements(
            cuda, gko::lend(csr_cuda_temp), false);
        csr_ref = gko::give(csr_ref_temp);
        csr_cuda = gko::give(csr_cuda_temp);
    }

    template <typename Mtx>
    std::unique_ptr<Mtx> gen_mtx(index_type num_rows, index_type num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(0, num_cols - 1),
            std::normal_distribution<value_type>(0.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Csr> gen_unsorted_mtx(index_type num_rows,
                                          index_type num_cols)
    {
        using std::swap;
        auto mtx = gen_mtx<Csr>(num_rows, num_cols);
        auto values = mtx->get_values();
        auto col_idxs = mtx->get_col_idxs();
        const auto row_ptrs = mtx->get_const_row_ptrs();
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
                    swap(values[idx1], values[idx2]);
                    swap(col_idxs[idx1], col_idxs[idx2]);
                }
            }
        }
        return mtx;
    }

    void initialize_row_ptrs(index_type *l_row_ptrs_ref,
                             index_type *u_row_ptrs_ref,
                             index_type *l_row_ptrs_cuda,
                             index_type *u_row_ptrs_cuda)
    {
        gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
            ref, gko::lend(csr_ref), l_row_ptrs_ref, u_row_ptrs_ref);
        gko::kernels::cuda::factorization::initialize_row_ptrs_l_u(
            cuda, gko::lend(csr_cuda), l_row_ptrs_cuda, u_row_ptrs_cuda);
    }

    void initialize_lu(std::unique_ptr<Csr> *l_ref, std::unique_ptr<Csr> *u_ref,
                       std::unique_ptr<Csr> *l_cuda,
                       std::unique_ptr<Csr> *u_cuda)
    {
        auto num_row_ptrs = csr_ref->get_size()[0] + 1;
        gko::Array<index_type> l_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> l_row_ptrs_cuda{cuda, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_cuda{cuda, num_row_ptrs};

        initialize_row_ptrs(
            l_row_ptrs_ref.get_data(), u_row_ptrs_ref.get_data(),
            l_row_ptrs_cuda.get_data(), u_row_ptrs_cuda.get_data());
        // Since `initialize_row_ptrs` was already tested, it is expected that
        // `*_ref` and `*_cuda` contain identical values
        auto l_nnz = l_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];
        auto u_nnz = u_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];

        *l_ref = Csr::create(ref, csr_ref->get_size(), l_nnz);
        *u_ref = Csr::create(ref, csr_ref->get_size(), u_nnz);
        *l_cuda = Csr::create(cuda, csr_cuda->get_size(), l_nnz);
        *u_cuda = Csr::create(cuda, csr_cuda->get_size(), u_nnz);
        // Copy the already initialized `row_ptrs` to the new matrices
        ref->copy(num_row_ptrs, l_row_ptrs_ref.get_data(),
                  (*l_ref)->get_row_ptrs());
        ref->copy(num_row_ptrs, u_row_ptrs_ref.get_data(),
                  (*u_ref)->get_row_ptrs());
        cuda->copy(num_row_ptrs, l_row_ptrs_cuda.get_data(),
                   (*l_cuda)->get_row_ptrs());
        cuda->copy(num_row_ptrs, u_row_ptrs_cuda.get_data(),
                   (*u_cuda)->get_row_ptrs());

        gko::kernels::reference::factorization::initialize_l_u(
            ref, gko::lend(csr_ref), gko::lend(*l_ref), gko::lend(*u_ref));
        gko::kernels::cuda::factorization::initialize_l_u(
            cuda, gko::lend(csr_cuda), gko::lend(*l_cuda), gko::lend(*u_cuda));
    }

    template <typename ToType, typename FromType>
    static std::unique_ptr<ToType> static_unique_ptr_cast(
        std::unique_ptr<FromType> &&from)
    {
        return std::unique_ptr<ToType>{static_cast<ToType *>(from.release())};
    }

    void compute_lu(std::unique_ptr<Csr> *l_ref, std::unique_ptr<Csr> *u_ref,
                    std::unique_ptr<Csr> *l_cuda, std::unique_ptr<Csr> *u_cuda,
                    gko::size_type iterations = 0)
    {
        auto coo_ref = Coo::create(ref);
        csr_ref->convert_to(gko::lend(coo_ref));
        auto coo_cuda = Coo::create(cuda);
        csr_cuda->convert_to(gko::lend(coo_cuda));
        initialize_lu(l_ref, u_ref, l_cuda, u_cuda);
        auto u_transpose_lin_op_ref = (*u_ref)->transpose();
        auto u_transpose_csr_ref =
            static_unique_ptr_cast<Csr>(std::move(u_transpose_lin_op_ref));
        auto u_transpose_lin_op_cuda = (*u_cuda)->transpose();
        auto u_transpose_csr_cuda =
            static_unique_ptr_cast<Csr>(std::move(u_transpose_lin_op_cuda));

        gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
            ref, iterations, gko::lend(coo_ref), gko::lend(*l_ref),
            gko::lend(u_transpose_csr_ref));
        gko::kernels::cuda::par_ilu_factorization::compute_l_u_factors(
            cuda, iterations, gko::lend(coo_cuda), gko::lend(*l_cuda),
            gko::lend(u_transpose_csr_cuda));
        auto u_lin_op_ref = u_transpose_csr_ref->transpose();
        *u_ref = static_unique_ptr_cast<Csr>(std::move(u_lin_op_ref));
        auto u_lin_op_cuda = u_transpose_csr_cuda->transpose();
        *u_cuda = static_unique_ptr_cast<Csr>(std::move(u_lin_op_cuda));
    }
};


TEST_F(ParIlu, CudaKernelAddDiagonalElementsSortedEquivalentToRef)
{
    index_type num_rows{600};
    index_type num_cols{600};
    auto mtx_ref = gen_mtx<Csr>(num_rows, num_cols);
    auto mtx_cuda = Csr::create(cuda);
    mtx_cuda->copy_from(gko::lend(mtx_ref));

    gko::kernels::reference::factorization::add_diagonal_elements(
        ref, gko::lend(mtx_ref), true);
    gko::kernels::cuda::factorization::add_diagonal_elements(
        cuda, gko::lend(mtx_cuda), true);

    ASSERT_TRUE(mtx_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx_ref, mtx_cuda, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx_ref, mtx_cuda);
}


TEST_F(ParIlu, CudaKernelAddDiagonalElementsUnsortedEquivalentToRef)
{
    index_type num_rows{600};
    index_type num_cols{600};
    auto mtx_ref = gen_unsorted_mtx(num_rows, num_cols);
    auto mtx_cuda = Csr::create(cuda);
    mtx_cuda->copy_from(gko::lend(mtx_ref));

    gko::kernels::reference::factorization::add_diagonal_elements(
        ref, gko::lend(mtx_ref), false);
    gko::kernels::cuda::factorization::add_diagonal_elements(
        cuda, gko::lend(mtx_cuda), false);

    ASSERT_FALSE(mtx_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx_ref, mtx_cuda, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx_ref, mtx_cuda);
}


TEST_F(ParIlu, CudaKernelAddDiagonalElementsNonSquareEquivalentToRef)
{
    index_type num_rows{600};
    index_type num_cols{500};
    auto mtx_ref = gen_mtx<Csr>(num_rows, num_cols);
    auto mtx_cuda = Csr::create(cuda);
    mtx_cuda->copy_from(gko::lend(mtx_ref));

    gko::kernels::reference::factorization::add_diagonal_elements(
        ref, gko::lend(mtx_ref), true);
    gko::kernels::cuda::factorization::add_diagonal_elements(
        cuda, gko::lend(mtx_cuda), true);

    ASSERT_TRUE(mtx_ref->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx_ref, mtx_cuda, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx_ref, mtx_cuda);
}


TEST_F(ParIlu, KernelInitializeRowPtrsLUEquivalentToRef)
{
    auto num_row_ptrs = csr_ref->get_size()[0] + 1;
    gko::Array<index_type> l_row_ptrs_array_ref(ref, num_row_ptrs);
    gko::Array<index_type> u_row_ptrs_array_ref(ref, num_row_ptrs);
    gko::Array<index_type> l_row_ptrs_array_cuda(cuda, num_row_ptrs);
    gko::Array<index_type> u_row_ptrs_array_cuda(cuda, num_row_ptrs);

    initialize_row_ptrs(
        l_row_ptrs_array_ref.get_data(), u_row_ptrs_array_ref.get_data(),
        l_row_ptrs_array_cuda.get_data(), u_row_ptrs_array_cuda.get_data());

    GKO_ASSERT_ARRAY_EQ(l_row_ptrs_array_ref, l_row_ptrs_array_cuda);
    GKO_ASSERT_ARRAY_EQ(u_row_ptrs_array_ref, u_row_ptrs_array_cuda);
}


TEST_F(ParIlu, KernelInitializeParILUIsEquivalentToRef)
{
    std::unique_ptr<Csr> l_ref{};
    std::unique_ptr<Csr> u_ref{};
    std::unique_ptr<Csr> l_cuda{};
    std::unique_ptr<Csr> u_cuda{};

    initialize_lu(&l_ref, &u_ref, &l_cuda, &u_cuda);

    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
}


TEST_F(ParIlu, KernelComputeParILUIsEquivalentToRef)
{
    std::unique_ptr<Csr> l_ref{};
    std::unique_ptr<Csr> u_ref{};
    std::unique_ptr<Csr> l_cuda{};
    std::unique_ptr<Csr> u_cuda{};

    compute_lu(&l_ref, &u_ref, &l_cuda, &u_cuda);

    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, 5e-2);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, 5e-2);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
}


TEST_F(ParIlu, KernelComputeParILUWithMoreIterationsIsEquivalentToRef)
{
    std::unique_ptr<Csr> l_ref{};
    std::unique_ptr<Csr> u_ref{};
    std::unique_ptr<Csr> l_cuda{};
    std::unique_ptr<Csr> u_cuda{};
    gko::size_type iterations{200};

    compute_lu(&l_ref, &u_ref, &l_cuda, &u_cuda, iterations);

    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
}


}  // namespace
