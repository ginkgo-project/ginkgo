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


#include <algorithm>
#include <fstream>
#include <memory>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


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

    void compute_nnz(index_type *l_row_ptrs_ref, index_type *u_row_ptrs_ref,
                     index_type *l_row_ptrs_omp, index_type *u_row_ptrs_omp)
    {
        gko::kernels::reference::par_ilu_factorization::initialize_row_ptrs_l_u(
            ref, gko::lend(csr_ref), l_row_ptrs_ref, u_row_ptrs_ref);
        gko::kernels::omp::par_ilu_factorization::initialize_row_ptrs_l_u(
            omp, gko::lend(csr_omp), l_row_ptrs_omp, u_row_ptrs_omp);
    }

    void initialize_lu(std::unique_ptr<Csr> *l_ref, std::unique_ptr<Csr> *u_ref,
                       std::unique_ptr<Csr> *l_omp, std::unique_ptr<Csr> *u_omp)
    {
        auto num_row_ptrs = csr_ref->get_size()[0] + 1;
        gko::Array<index_type> l_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_ref{ref, num_row_ptrs};
        gko::Array<index_type> l_row_ptrs_omp{omp, num_row_ptrs};
        gko::Array<index_type> u_row_ptrs_omp{omp, num_row_ptrs};

        compute_nnz(l_row_ptrs_ref.get_data(), u_row_ptrs_ref.get_data(),
                    l_row_ptrs_omp.get_data(), u_row_ptrs_omp.get_data());
        // Since `compute_nnz` was already tested, it is expected that `*_ref`
        // and `*_omp` contain identical values
        auto l_nnz = l_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];
        auto u_nnz = u_row_ptrs_ref.get_const_data()[num_row_ptrs - 1];

        *l_ref = Csr::create(ref, csr_ref->get_size(), l_nnz);
        *u_ref = Csr::create(ref, csr_ref->get_size(), u_nnz);
        *l_omp = Csr::create(omp, csr_omp->get_size(), l_nnz);
        *u_omp = Csr::create(omp, csr_omp->get_size(), u_nnz);
        // Copy the already initialized `row_ptrs` to the new matrices
        ref->copy_from(gko::lend(ref), num_row_ptrs, l_row_ptrs_ref.get_data(),
                       (*l_ref)->get_row_ptrs());
        ref->copy_from(gko::lend(ref), num_row_ptrs, u_row_ptrs_ref.get_data(),
                       (*u_ref)->get_row_ptrs());
        omp->copy_from(gko::lend(omp), num_row_ptrs, l_row_ptrs_omp.get_data(),
                       (*l_omp)->get_row_ptrs());
        omp->copy_from(gko::lend(omp), num_row_ptrs, u_row_ptrs_omp.get_data(),
                       (*u_omp)->get_row_ptrs());

        gko::kernels::reference::par_ilu_factorization::initialize_l_u(
            ref, gko::lend(csr_ref), gko::lend(*l_ref), gko::lend(*u_ref));
        gko::kernels::omp::par_ilu_factorization::initialize_l_u(
            omp, gko::lend(csr_omp), gko::lend(*l_omp), gko::lend(*u_omp));
    }
};


TEST_F(ParIlu, OmpKernelComputeNnzLUEquivalentToRef)
{
    auto num_row_ptrs = csr_ref->get_size()[0] + 1;
    gko::Array<index_type> l_row_ptrs_array_ref(ref, num_row_ptrs);
    gko::Array<index_type> u_row_ptrs_array_ref(ref, num_row_ptrs);
    gko::Array<index_type> l_row_ptrs_array_omp(omp, num_row_ptrs);
    gko::Array<index_type> u_row_ptrs_array_omp(omp, num_row_ptrs);
    auto l_row_ptrs_ref = l_row_ptrs_array_ref.get_data();
    auto u_row_ptrs_ref = u_row_ptrs_array_ref.get_data();
    auto l_row_ptrs_omp = l_row_ptrs_array_omp.get_data();
    auto u_row_ptrs_omp = u_row_ptrs_array_omp.get_data();

    compute_nnz(l_row_ptrs_ref, u_row_ptrs_ref, l_row_ptrs_omp, u_row_ptrs_omp);

    ASSERT_TRUE(std::equal(l_row_ptrs_ref, l_row_ptrs_ref + num_row_ptrs,
                           l_row_ptrs_omp));
    ASSERT_TRUE(std::equal(u_row_ptrs_ref, u_row_ptrs_ref + num_row_ptrs,
                           u_row_ptrs_omp));
}


TEST_F(ParIlu, KernelInitializeParILUIsEquivalentToRef)
{
    std::unique_ptr<Csr> l_ref{};
    std::unique_ptr<Csr> u_ref{};
    std::unique_ptr<Csr> l_omp{};
    std::unique_ptr<Csr> u_omp{};

    initialize_lu(&l_ref, &u_ref, &l_omp, &u_omp);

    GKO_ASSERT_MTX_NEAR(l_ref, l_omp, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_ref, u_omp, 1e-14);
}


TEST_F(ParIlu, KernelComputeParILUIsEquivalentToRef)
{
    std::unique_ptr<Csr> l_ref{};
    std::unique_ptr<Csr> u_ref{};
    std::unique_ptr<Csr> l_omp{};
    std::unique_ptr<Csr> u_omp{};
    gko::size_type iterations{};
    auto coo_ref = Coo::create(ref);
    csr_ref->convert_to(gko::lend(coo_ref));
    auto coo_omp = Coo::create(omp);
    csr_omp->convert_to(gko::lend(coo_omp));
    initialize_lu(&l_ref, &u_ref, &l_omp, &u_omp);
    auto u_transpose_lin_op_ref = u_ref->transpose();
    auto u_transpose_ptr_ref = static_cast<Csr *>(u_transpose_lin_op_ref.get());
    auto u_transpose_lin_op_omp = u_omp->transpose();
    auto u_transpose_ptr_omp = static_cast<Csr *>(u_transpose_lin_op_omp.get());

    gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
        ref, iterations, gko::lend(coo_ref), gko::lend(l_ref),
        u_transpose_ptr_ref);
    gko::kernels::omp::par_ilu_factorization::compute_l_u_factors(
        omp, iterations, gko::lend(coo_omp), gko::lend(l_omp),
        u_transpose_ptr_omp);

    GKO_ASSERT_MTX_NEAR(l_ref, l_omp, 5e-3);
    GKO_ASSERT_MTX_NEAR(u_transpose_ptr_ref, u_transpose_ptr_omp, 5e-3);
}


}  // namespace
