/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <fstream>
#include <string>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class ParIlu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    ParIlu()
        : ref(gko::ReferenceExecutor::create()),
          omp(gko::OmpExecutor::create()),
          csr_ref(nullptr),
          csr_omp(nullptr)
    {}

    void SetUp() override
    {
        std::string file_name("ani1.mtx");
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\".\n"
                      "Please make sure you call this test from the "
                      "directory where this binary is located, so the "
                      "matrix file can be found.\n";
        }
        csr_ref = gko::read<Csr>(input_file, ref);
        auto csr_omp_temp = Csr::create(omp);
        csr_omp_temp->copy_from(gko::lend(csr_ref));
        csr_omp = gko::give(csr_omp_temp);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::shared_ptr<const Csr> csr_ref;
    std::shared_ptr<const Csr> csr_omp;
};


TEST_F(ParIlu, OmpKernelComputeNnzLUEquivalentToRef)
{
    gko::size_type l_nnz_ref{};
    gko::size_type u_nnz_ref{};
    gko::size_type l_nnz_omp{};
    gko::size_type u_nnz_omp{};

    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_ref), &l_nnz_ref, &u_nnz_ref);
    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_omp), &l_nnz_omp, &u_nnz_omp);

    ASSERT_EQ(l_nnz_omp, l_nnz_ref);
    ASSERT_EQ(u_nnz_omp, u_nnz_ref);
}


TEST_F(ParIlu, KernelInitializeParILUIsEquivalentToRef)
{
    gko::size_type l_nnz_ref{};
    gko::size_type u_nnz_ref{};
    gko::size_type l_nnz_omp{};
    gko::size_type u_nnz_omp{};
    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_ref), &l_nnz_ref, &u_nnz_ref);
    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_omp), &l_nnz_omp, &u_nnz_omp);
    auto l_ref = Csr::create(ref, csr_ref->get_size(), l_nnz_ref);
    auto u_ref = Csr::create(ref, csr_ref->get_size(), u_nnz_ref);
    auto l_omp = Csr::create(ref, csr_omp->get_size(), l_nnz_omp);
    auto u_omp = Csr::create(ref, csr_omp->get_size(), u_nnz_omp);

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_ref), gko::lend(l_ref), gko::lend(u_ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_omp), gko::lend(l_omp), gko::lend(u_omp));

    GKO_ASSERT_MTX_NEAR(l_ref, l_omp, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_ref, u_omp, 1e-14);
}


TEST_F(ParIlu, KernelComputeParILUIsEquivalentToRef)
{
    gko::size_type l_nnz_ref{};
    gko::size_type u_nnz_ref{};
    gko::size_type l_nnz_omp{};
    gko::size_type u_nnz_omp{};
    gko::size_type iterations{};
    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_ref), &l_nnz_ref, &u_nnz_ref);
    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_omp), &l_nnz_omp, &u_nnz_omp);
    auto l_ref = Csr::create(ref, csr_ref->get_size(), l_nnz_ref);
    auto u_ref = Csr::create(ref, csr_ref->get_size(), u_nnz_ref);
    auto l_omp = Csr::create(ref, csr_omp->get_size(), l_nnz_omp);
    auto u_omp = Csr::create(ref, csr_omp->get_size(), u_nnz_omp);
    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_ref), gko::lend(l_ref), gko::lend(u_ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_omp), gko::lend(l_omp), gko::lend(u_omp));
    auto coo_ref = Coo::create(ref);
    csr_ref->convert_to(gko::lend(coo_ref));
    auto u_transpose_ref = u_ref->transpose();
    gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
        ref, iterations, gko::lend(coo_ref), gko::lend(l_ref),
        gko::lend(u_ref));
    auto coo_omp = Coo::create(omp);
    csr_omp->convert_to(gko::lend(coo_omp));
    auto u_transpose_omp = u_omp->transpose();

    gko::kernels::omp::par_ilu_factorization::compute_l_u_factors(
        omp, iterations, gko::lend(coo_omp), gko::lend(l_omp),
        gko::lend(u_omp));

    GKO_ASSERT_MTX_NEAR(l_ref, l_omp, .3);
    GKO_ASSERT_MTX_NEAR(u_ref, u_omp, .3);
}


}  // namespace
