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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <fstream>
#include <string>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/test/utils/assertions.hpp"


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
          csr_Ref(nullptr),
          csr_Omp(nullptr)
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
        csr_Ref = gko::read<Csr>(input_file, ref);

        auto csr_omp_temp = Csr::create(omp);
        csr_omp_temp->copy_from(gko::lend(csr_Ref));
        csr_Omp = gko::give(csr_omp_temp);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::shared_ptr<const Csr> csr_Ref;
    std::shared_ptr<const Csr> csr_Omp;
};


TEST_F(ParIlu, OmpKernelComputeNnzLUEquivalentToRef)
{
    gko::size_type l_nnz_Ref{};
    gko::size_type u_nnz_Ref{};
    gko::size_type l_nnz_Omp{};
    gko::size_type u_nnz_Omp{};

    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Ref), &l_nnz_Ref, &u_nnz_Ref);

    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Omp), &l_nnz_Omp, &u_nnz_Omp);

    ASSERT_EQ(l_nnz_Omp, l_nnz_Ref);
    ASSERT_EQ(u_nnz_Omp, u_nnz_Ref);
}


TEST_F(ParIlu, KernelInitializeParILUIsEquivalentToRef)
{
    gko::size_type l_nnz_Ref{};
    gko::size_type u_nnz_Ref{};
    gko::size_type l_nnz_Omp{};
    gko::size_type u_nnz_Omp{};

    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Ref), &l_nnz_Ref, &u_nnz_Ref);

    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Omp), &l_nnz_Omp, &u_nnz_Omp);

    auto l_Ref = Csr::create(ref, csr_Ref->get_size(), l_nnz_Ref);
    auto u_Ref = Csr::create(ref, csr_Ref->get_size(), u_nnz_Ref);

    auto l_Omp = Csr::create(ref, csr_Omp->get_size(), l_nnz_Omp);
    auto u_Omp = Csr::create(ref, csr_Omp->get_size(), u_nnz_Omp);

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Ref), gko::lend(l_Ref), gko::lend(u_Ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Omp), gko::lend(l_Omp), gko::lend(u_Omp));

    GKO_ASSERT_MTX_NEAR(l_Ref, l_Omp, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_Ref, u_Omp, 1e-14);
}


TEST_F(ParIlu, KernelComputeParILUIsEquivalentToRef)
{
    gko::size_type l_nnz_Ref{};
    gko::size_type u_nnz_Ref{};
    gko::size_type l_nnz_Omp{};
    gko::size_type u_nnz_Omp{};

    gko::size_type iterations{};

    gko::kernels::reference::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Ref), &l_nnz_Ref, &u_nnz_Ref);

    gko::kernels::omp::par_ilu_factorization::compute_nnz_l_u(
        ref, gko::lend(csr_Omp), &l_nnz_Omp, &u_nnz_Omp);

    auto l_Ref = Csr::create(ref, csr_Ref->get_size(), l_nnz_Ref);
    auto u_Ref = Csr::create(ref, csr_Ref->get_size(), u_nnz_Ref);

    auto l_Omp = Csr::create(ref, csr_Omp->get_size(), l_nnz_Omp);
    auto u_Omp = Csr::create(ref, csr_Omp->get_size(), u_nnz_Omp);

    gko::kernels::reference::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Ref), gko::lend(l_Ref), gko::lend(u_Ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Omp), gko::lend(l_Omp), gko::lend(u_Omp));

    auto coo_Ref = Coo::create(ref);
    csr_Ref->convert_to(gko::lend(coo_Ref));

    auto u_transpose_Ref = u_Ref->transpose();

    gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
        ref, iterations, gko::lend(coo_Ref), gko::lend(l_Ref),
        gko::lend(u_Ref));

    auto coo_Omp = Coo::create(omp);
    csr_Omp->convert_to(gko::lend(coo_Omp));

    auto u_transpose_Omp = u_Omp->transpose();

    gko::kernels::omp::par_ilu_factorization::compute_l_u_factors(
        omp, iterations, gko::lend(coo_Omp), gko::lend(l_Omp),
        gko::lend(u_Omp));

    GKO_ASSERT_MTX_NEAR(l_Ref, l_Omp, .3);
    GKO_ASSERT_MTX_NEAR(u_Ref, u_Omp, .3);
}


}  // namespace
