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
#include "core/test/factorization/block_factorization_test_utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"
#include "reference/test/factorization/bilu_sample.hpp"


namespace {


template <typename ValueIndexType>
class ParBilu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using BILUSample = gko::testing::Bilu0Sample<value_type, index_type>;

    std::ranlux48 rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::unique_ptr<const Fbcsr> cyl2d_ref;
    const real_type tol = std::numeric_limits<real_type>::epsilon();

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
        gko::test::initialize_bilu(mat_ref, &l_init_ref, &u_init_ref);
        *l_cuda = Fbcsr::create(cuda);
        l_init_ref->convert_to(l_cuda->get());
        auto u_transpose_ref = gko::as<Fbcsr>(u_init_ref->transpose());
        auto u_transpose_cuda = Fbcsr::create(cuda);
        u_transpose_cuda->copy_from(gko::lend(u_transpose_ref));

        auto mat_ref_copy = Fbcsr::create(ref);
        mat_ref_copy->copy_from(mat_ref);
        gko::kernels::reference::bilu_factorization::compute_bilu(
            ref, gko::lend(mat_ref_copy));
        gko::test::initialize_bilu(mat_ref_copy.get(), l_ref, u_ref);

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
        gko::test::initialize_bilu(mat_ref, l_ref, u_ref);
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

    void jacobi_bilu(const Fbcsr *const mat_ref, const int iterations,
                     std::shared_ptr<Fbcsr> *const l_ref,
                     std::shared_ptr<Fbcsr> *const u_ref,
                     std::shared_ptr<Fbcsr> *const l_cuda,
                     std::shared_ptr<Fbcsr> *const u_cuda)
    {
        auto mat_cuda = Fbcsr::create(cuda);
        mat_cuda->copy_from(gko::lend(mat_ref));
        gko::test::initialize_bilu(mat_ref, l_ref, u_ref);
        *l_cuda = Fbcsr::create(cuda);
        (*l_ref)->convert_to(l_cuda->get());
        auto u_transpose_ref = gko::as<Fbcsr>((*u_ref)->transpose());
        auto u_transpose_cuda = Fbcsr::create(cuda);
        u_transpose_cuda->copy_from(gko::lend(u_transpose_ref));

        gko::kernels::reference::par_bilu_factorization::
            compute_bilu_factors_jacobi(ref, iterations, mat_ref, l_ref->get(),
                                        u_transpose_ref.get());
        auto u_lin_op_ref = u_transpose_ref->transpose();
        *u_ref = static_unique_ptr_cast<Fbcsr>(std::move(u_lin_op_ref));

        gko::kernels::cuda::par_bilu_factorization::compute_bilu_factors_jacobi(
            cuda, iterations, gko::lend(mat_cuda), gko::lend(*l_cuda),
            gko::lend(u_transpose_cuda));
        auto u_lin_op_cuda = u_transpose_cuda->transpose();
        *u_cuda = static_unique_ptr_cast<Fbcsr>(std::move(u_lin_op_cuda));
    }

    void jacobi_bilu_notrans(const Fbcsr *const mat_ref, const int iterations,
                             std::shared_ptr<Fbcsr> *const l_ref,
                             std::shared_ptr<Fbcsr> *const u_ref,
                             std::shared_ptr<Fbcsr> *const l_cuda,
                             std::shared_ptr<Fbcsr> *const u_cuda)
    {
        auto mat_cuda = Fbcsr::create(cuda);
        mat_cuda->copy_from(gko::lend(mat_ref));
        gko::test::initialize_bilu(mat_ref, l_ref, u_ref);
        *l_cuda = Fbcsr::create(cuda);
        *u_cuda = Fbcsr::create(cuda);
        (*l_ref)->convert_to(l_cuda->get());
        (*u_ref)->convert_to(u_cuda->get());

        gko::kernels::reference::par_bilu_factorization::
            compute_bilu_factors_jacobi(ref, iterations, mat_ref, l_ref->get(),
                                        u_ref->get());

        gko::kernels::cuda::par_bilu_factorization::compute_bilu_factors_jacobi(
            cuda, iterations, gko::lend(mat_cuda), gko::lend(*l_cuda),
            gko::lend(*u_cuda));
    }
};

using SomeTypes =
    ::testing::Types<std::tuple<double, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;

TYPED_TEST_SUITE(ParBilu, SomeTypes);


TYPED_TEST(ParBilu, CudaKernelBLUSortedSampleBS3)
{
    using BILUSample = typename TestFixture::BILUSample;
    using Fbcsr = typename TestFixture::Fbcsr;
    BILUSample bilusample(this->ref);
    auto refmat = bilusample.generate_fbcsr();
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 1;

    this->compute_bilu_2(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                         &u_cuda);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, this->tol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, this->tol);
}

TYPED_TEST(ParBilu, CudaKernelBLUSortedRandomBS4)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const int num_brows = 90;
    auto refmat = gko::test::generate_random_fbcsr<value_type, index_type>(
        this->ref, std::ranlux48(43), num_brows, num_brows, 4, true, false);
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 8;

    this->compute_bilu_2(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                         &u_cuda);

    const double ttol = 10 * this->tol;
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, ttol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, ttol);
}

TYPED_TEST(ParBilu, CudaKernelBLUSortedRandomBS7)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const bool diag_dom = true;
    const bool unsort = false;
    const int num_brows = 50;
    auto refmat = gko::test::generate_random_fbcsr<value_type, index_type>(
        this->ref, std::ranlux48(43), num_brows, num_brows, 7, diag_dom,
        unsort);
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 50;

    this->compute_bilu_2(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                         &u_cuda);

    // For BS 7, initial error in L (reported by the macro) is ~1.0
    const double ttol = 10 * this->tol;
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, ttol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, ttol);
}

TYPED_TEST(ParBilu, CudaKernelBLUJacobiSortedSampleBS3)
{
    using BILUSample = typename TestFixture::BILUSample;
    using Fbcsr = typename TestFixture::Fbcsr;
    BILUSample bilusample(this->ref);
    auto refmat = bilusample.generate_fbcsr();
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 1;

    this->jacobi_bilu(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                      &u_cuda);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, this->tol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, this->tol);
}

// This test passes for block sizes upto 4, but not 7.
TYPED_TEST(ParBilu, CudaKernelBLUJacobiSortedRandomBS4)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const bool diag_dom = true;
    const bool unsort = false;
    const int num_brows = 50;
    const int bs = 4;
    auto refmat = gko::test::generate_random_fbcsr<value_type, index_type>(
        this->ref, std::ranlux48(43), num_brows, num_brows, bs, diag_dom,
        unsort);
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 5;

    this->jacobi_bilu(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                      &u_cuda);

    const double ttol = 5 * this->tol;
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, ttol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, ttol);
}

TYPED_TEST(ParBilu, CudaKernelBLUJacobiSortedRandomBS7)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Fbcsr;
    const bool diag_dom = true;
    const bool unsort = false;
    const int num_brows = 10;
    const int bs = 7;
    auto refmat = gko::test::generate_random_fbcsr<value_type, index_type>(
        this->ref, std::ranlux48(43), num_brows, num_brows, bs, diag_dom,
        unsort);
    std::shared_ptr<Fbcsr> l_ref, u_ref, l_cuda, u_cuda;
    const int iterations = 1;

    this->jacobi_bilu(refmat.get(), iterations, &l_ref, &u_ref, &l_cuda,
                      &u_cuda);

    const double ttol = 5 * this->tol;
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_cuda);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_cuda);
    GKO_ASSERT_MTX_NEAR(l_ref, l_cuda, ttol);
    GKO_ASSERT_MTX_NEAR(u_ref, u_cuda, ttol);
}


}  // namespace
