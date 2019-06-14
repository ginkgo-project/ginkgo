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
          csr_Ref(Csr::create(ref)),
          csr_Omp(Csr::create(omp)),
          // clang-format off
        mtx_ani1(gko::initialize<Dense>({{    1.0000,   -0.4376,    0.1540,         0,         0,         0,   -0.3634,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {   -0.4376,    1.0000,         0,   -0.3765,         0,         0,    0.1363,   -0.1057,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {    0.1540,         0,    1.0000,         0,         0,         0,   -0.3297,         0,    0.1200,   -0.4233,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,   -0.3765,         0,    1.0000,   -0.5245,         0,         0,    0.0388,         0,         0,   -0.0425,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,   -0.5245,    1.0000,   -0.5879,         0,         0,         0,         0,    0.0716,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,   -0.5879,    1.0000,         0,         0,         0,         0,   -0.0253,   -0.0314,   -0.2689,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {   -0.3634,    0.1363,   -0.3297,         0,         0,         0,    1.0000,   -0.4432,         0,    0.1826,         0,         0,         0,   -0.1909,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,   -0.1057,         0,    0.0388,         0,         0,   -0.4432,    1.0000,         0,         0,   -0.5051,         0,         0,    0.1128,   -0.0932,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,    0.1200,         0,         0,         0,         0,         0,    1.0000,   -0.2984,         0,         0,         0,         0,         0,    0.1074,         0,   -0.3868,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,   -0.4233,         0,         0,         0,    0.1826,         0,   -0.2984,    1.0000,         0,         0,         0,   -0.4792,         0,         0,         0,    0.1073,   -0.1520,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,   -0.0425,    0.0716,   -0.0253,         0,   -0.5051,         0,         0,    1.0000,   -0.3851,         0,         0,   -0.0219,         0,         0,         0,         0,   -0.0185,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,   -0.0314,         0,         0,         0,         0,   -0.3851,    1.0000,   -0.3977,         0,         0,         0,         0,         0,         0,    0.0699,   -0.2465,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,   -0.2689,         0,         0,         0,         0,         0,   -0.3977,    1.0000,         0,         0,         0,         0,         0,         0,         0,    0.1897,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,   -0.1909,    0.1128,         0,   -0.4792,         0,         0,         0,    1.0000,   -0.4509,         0,         0,         0,    0.1189,         0,         0,   -0.1159,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,   -0.0932,         0,         0,   -0.0219,         0,         0,   -0.4509,    1.0000,         0,         0,         0,         0,   -0.4741,         0,    0.0682,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,    0.1074,         0,         0,         0,         0,         0,         0,    1.0000,    0.1632,   -0.2887,         0,         0,         0,         0,   -0.3573,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.1632,    1.0000,         0,         0,         0,         0,         0,   -0.2744,   -0.4287,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,   -0.3868,    0.1073,         0,         0,         0,         0,         0,   -0.2887,         0,    1.0000,   -0.4037,         0,         0,         0,    0.1758,         0,   -0.2050,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.1520,         0,         0,         0,    0.1189,         0,         0,         0,   -0.4037,    1.0000,         0,         0,   -0.4946,         0,         0,   -0.0446,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.0185,    0.0699,         0,         0,   -0.4741,         0,         0,         0,         0,    1.0000,   -0.4056,   -0.1159,         0,         0,         0,    0.1376,   -0.1664,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.2465,    0.1897,         0,         0,         0,         0,         0,         0,   -0.4056,    1.0000,         0,         0,         0,         0,         0,   -0.0169,         0,         0,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.1159,    0.0682,         0,         0,         0,   -0.4946,   -0.1159,         0,    1.0000,         0,         0,    0.0495,   -0.4664,         0,         0,    0.0808,         0,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.3573,   -0.2744,    0.1758,         0,         0,         0,         0,    1.0000,    0.1180,   -0.3740,         0,         0,         0,         0,   -0.2244,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.4287,         0,         0,         0,         0,         0,    0.1180,    1.0000,         0,         0,         0,         0,         0,   -0.4560,         0,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.2050,   -0.0446,         0,         0,    0.0495,   -0.3740,         0,    1.0000,         0,         0,         0,   -0.3698,    0.1860,   -0.1570,         0,         0,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.1376,         0,   -0.4664,         0,         0,         0,    1.0000,   -0.5246,         0,   -0.1676,         0,         0,         0,   -0.0424,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.1664,   -0.0169,         0,         0,         0,         0,   -0.5246,    1.0000,   -0.1991,         0,         0,         0,         0,    0.1811,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.1991,    1.0000,         0,         0,         0,    0.1269,   -0.2610,         0,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.0808,         0,         0,   -0.3698,   -0.1676,         0,         0,    1.0000,         0,   -0.2986,         0,   -0.4855,    0.1958,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.2244,   -0.4560,    0.1860,         0,         0,         0,         0,    1.0000,   -0.2961,         0,         0,   -0.2931,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.1570,         0,         0,         0,   -0.2986,   -0.2961,    1.0000,         0,         0,   -0.2635,         0,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.1269,         0,         0,         0,    1.0000,   -0.3972,         0,   -0.0101,         0},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.0424,    0.1811,   -0.2610,   -0.4855,         0,         0,   -0.3972,    1.0000,   -0.1328,   -0.1827,    0.1721},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.1958,   -0.2931,   -0.2635,         0,   -0.1328,    1.0000,         0,   -0.5043},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,   -0.0101,   -0.1827,         0,    1.0000,   -0.5431},
                                         {         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,    0.1721,   -0.5043,   -0.5431,    1.0000}},
                                         ref))
        //clang-format on   
    {                                
        mtx_ani1->convert_to(gko::lend(csr_Ref));
        mtx_ani1->convert_to(gko::lend(csr_Omp));
    }
    
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::shared_ptr<Dense>mtx_ani1;
    std::shared_ptr<Csr>csr_Ref;
    std::shared_ptr<Csr>csr_Omp;
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
        ref, gko::lend(csr_Ref), gko::lend(l_Ref),
        gko::lend(u_Ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Omp), gko::lend(l_Omp),
        gko::lend(u_Omp));

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
        ref, gko::lend(csr_Ref), gko::lend(l_Ref),
        gko::lend(u_Ref));
    gko::kernels::omp::par_ilu_factorization::initialize_l_u(
        ref, gko::lend(csr_Omp), gko::lend(l_Omp),
        gko::lend(u_Omp));
    
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
