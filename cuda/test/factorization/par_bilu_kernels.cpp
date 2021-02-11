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

#include "core/factorization/par_bilu_kernels.hpp"


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


#include "core/factorization/bilu_kernels.hpp"
#include "core/factorization/block_factorization_kernels.hpp"
#include "core/factorization/par_bilu_kernels.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"
#include "reference/test/factorization/bilu_sample.hpp"


namespace {


class ParBilu : public ::testing::Test {
protected:
    // using value_type = gko::default_precision;
    using value_type = float;
    using real_type = gko::remove_complex<value_type>;
    using index_type = gko::int32;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using BILUSample = gko::testing::Bilu0Sample<value_type, index_type>;

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::unique_ptr<const Fbcsr> cyl2d_ref;
    std::unique_ptr<const Fbcsr> rand_ref;
    const value_type tol = std::numeric_limits<real_type>::epsilon();

    ParBilu()
        : rand_engine(18),
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref))
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
        auto ref_temp = gko::read<Fbcsr>(input_file, ref, 4);
        input_file.close();
        // Make sure there are diagonal elements present
        gko::kernels::reference::factorization::add_diagonal_blocks(
            ref, gko::lend(ref_temp), false);
        cyl2d_ref = gko::give(ref_temp);

        const int num_brows = 90;
        rand_ref = gko::test::generate_random_fbcsr<value_type, index_type>(
            ref, std::ranlux48(43), num_brows, num_brows, 4, true, false);
    }

    /*
     * Initialize L and U factor matrices
     */
    void initialize_bilu(const Fbcsr *const ref_mat,
                         std::shared_ptr<Fbcsr> *const l_factor,
                         std::shared_ptr<Fbcsr> *const u_factor)
    {
        const auto exec = ref;
        const gko::size_type num_brows = ref_mat->get_num_block_rows();
        const int bs = ref_mat->get_block_size();
        gko::Array<index_type> l_row_ptrs{exec, num_brows + 1};
        gko::Array<index_type> u_row_ptrs{exec, num_brows + 1};
        gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
            exec, ref_mat, l_row_ptrs.get_data(), u_row_ptrs.get_data());
        const auto l_nbnz = l_row_ptrs.get_data()[num_brows];
        const auto u_nbnz = u_row_ptrs.get_data()[num_brows];
        gko::Array<index_type> l_col_idxs(exec, l_nbnz);
        gko::Array<value_type> l_vals(exec, l_nbnz * bs * bs);
        gko::Array<index_type> u_col_idxs(exec, u_nbnz);
        gko::Array<value_type> u_vals(exec, u_nbnz * bs * bs);
        *l_factor =
            Fbcsr::create(exec, ref_mat->get_size(), bs, std::move(l_vals),
                          std::move(l_col_idxs), std::move(l_row_ptrs));
        *u_factor =
            Fbcsr::create(exec, ref_mat->get_size(), bs, std::move(u_vals),
                          std::move(u_col_idxs), std::move(u_row_ptrs));
        gko::kernels::reference::factorization::initialize_BLU(
            ref, ref_mat, l_factor->get(), u_factor->get());
    }

    template <typename ToType, typename FromType>
    static std::unique_ptr<ToType> static_unique_ptr_cast(
        std::unique_ptr<FromType> &&from)
    {
        return std::unique_ptr<ToType>{static_cast<ToType *>(from.release())};
    }


    void compute_bilu(const Fbcsr *const mat_ref, const int iterations,
                      std::shared_ptr<Fbcsr> *const l_ref,
                      std::shared_ptr<Fbcsr> *const u_ref,
                      std::shared_ptr<Fbcsr> *const l_cuda,
                      std::shared_ptr<Fbcsr> *const u_cuda)
    {
        auto mat_cuda = Fbcsr::create(cuda);
        mat_ref->convert_to(gko::lend(mat_cuda));
        std::shared_ptr<Fbcsr> l_init_ref, u_init_ref;
        initialize_bilu(mat_ref, &l_init_ref, &u_init_ref);
        *l_cuda = Fbcsr::create(cuda);
        l_init_ref->convert_to(l_cuda->get());
        auto u_transpose_ref = gko::as<Fbcsr>(u_init_ref->transpose());
        auto u_transpose_cuda = Fbcsr::create(cuda);
        u_transpose_cuda->copy_from(gko::lend(u_transpose_ref));

        auto mat_ref_copy = Fbcsr::create(ref);
        mat_ref_copy->copy_from(mat_ref);
        gko::kernels::reference::bilu_factorization::compute_bilu(
            ref, gko::lend(mat_ref_copy));
        initialize_bilu(mat_ref_copy.get(), l_ref, u_ref);

        gko::kernels::cuda::par_bilu_factorization::compute_bilu_factors(
            cuda, iterations, gko::lend(mat_cuda), gko::lend(*l_cuda),
            gko::lend(u_transpose_cuda));
        auto u_lin_op_cuda = u_transpose_cuda->transpose();
        *u_cuda = static_unique_ptr_cast<Fbcsr>(std::move(u_lin_op_cuda));
    }

    void compute_bilu_2(const Fbcsr *const mat_ref, const int iterations,
                        std::shared_ptr<Fbcsr> *const l_ref,
                        std::shared_ptr<Fbcsr> *const u_ref,
                        std::shared_ptr<Fbcsr> *const l_cuda,
                        std::shared_ptr<Fbcsr> *const u_cuda)
    {
        auto mat_cuda = Fbcsr::create(cuda);
        mat_cuda->copy_from(gko::lend(mat_ref));
        initialize_bilu(mat_ref, l_ref, u_ref);
        *l_cuda = Fbcsr::create(cuda);
        (*l_ref)->convert_to(l_cuda->get());
        auto u_transpose_ref = gko::as<Fbcsr>((*u_ref)->transpose());
        auto u_transpose_cuda = Fbcsr::create(cuda);
        u_transpose_cuda->copy_from(gko::lend(u_transpose_ref));

        // exact factorization with 1 iteration
        gko::kernels::reference::par_bilu_factorization::compute_bilu_factors(
            ref, 1, mat_ref, l_ref->get(), u_transpose_ref.get());
        auto u_lin_op_ref = u_transpose_ref->transpose();
        *u_ref = static_unique_ptr_cast<Fbcsr>(std::move(u_lin_op_ref));

        gko::kernels::cuda::par_bilu_factorization::compute_bilu_factors(
            cuda, iterations, gko::lend(mat_cuda), gko::lend(*l_cuda),
            gko::lend(u_transpose_cuda));
        auto u_lin_op_cuda = u_transpose_cuda->transpose();
        *u_cuda = static_unique_ptr_cast<Fbcsr>(std::move(u_lin_op_cuda));
    }
};


TEST_F(ParBilu, CudaKernelBLUSortedSampleBS3)
{
    BILUSample bilusample(ref);
    auto refmat = bilusample.generate_fbcsr();
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 1;

    compute_bilu_2(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda, &u_cuda);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, tol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, tol);
}

TEST_F(ParBilu, CudaKernelBLUSortedRandomBS4)
{
    auto refmat = Fbcsr::create(ref);
    refmat->copy_from(rand_ref.get());
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    // const int iterations = rand_ref->get_num_stored_blocks()/10;
    const int iterations = 8;
    printf(" Num its set to %d.\n", iterations);

    compute_bilu_2(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda, &u_cuda);

    // For BS 7, initial error in L (reported by the macro) is ~1.0
    const double ttol = 10 * tol;
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, ttol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, ttol);
}


}  // namespace
