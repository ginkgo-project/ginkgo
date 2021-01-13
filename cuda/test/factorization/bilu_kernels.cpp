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

#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/factorization/bilu.hpp>
#include <ginkgo/core/factorization/ilu.hpp>


#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/factorization/bilu_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "cuda/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


class BiluCuda : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    const value_type eps;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::ranlux48 rand_engine;
    std::shared_ptr<Fbcsr> mat_ref;
    std::shared_ptr<Fbcsr> mat_cuda;
    std::shared_ptr<Csr> csr_ref;
    std::shared_ptr<Csr> csr_cuda;

    BiluCuda()
        : eps{std::numeric_limits<value_type>::epsilon()},
          ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          rand_engine(1337)
    {}

    void SetUp() override
    {
        const std::string file_name =
            gko::matrices::location_2dcyl1_prefix + std::string(".mtx");
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        const gko::matrix_data<value_type, index_type> matdata =
            gko::read_raw(input_file);
        input_file.close();
        mat_ref = Fbcsr::create(ref);
        mat_ref->set_block_size(4);
        mat_ref->read(matdata);
        mat_cuda = Fbcsr::create(cuda);
        mat_cuda->copy_from(gko::lend(mat_ref));

        csr_ref = Csr::create(ref, mat_ref->get_size(),
                              mat_ref->get_num_stored_elements());
        mat_ref->convert_to(csr_ref.get());
        csr_cuda = Csr::create(cuda);
        csr_cuda->copy_from(gko::lend(csr_ref));
    }
};


TEST_F(BiluCuda, ComputeUnsplitILUFbcsrIsEquivalentToCsr)
{
    auto mtxcuda = this->mat_cuda->clone();
    gko::kernels::cuda::bilu_factorization::compute_bilu(this->cuda,
                                                         mtxcuda.get());
    auto csr_cuda_copy = this->csr_cuda->clone();
    gko::kernels::cuda::ilu_factorization::compute_lu(this->cuda,
                                                      csr_cuda_copy.get());

    GKO_ASSERT_MTX_NEAR(csr_cuda_copy, mtxcuda, 100 * eps);
    GKO_ASSERT_MTX_EQ_SPARSITY(csr_cuda_copy, mtxcuda);
}


// TEST_F(BiluCuda, ComputeILUIsEquivalentToRefSorted)
// {
//     auto ref_fact = gko::factorization::ParIlu<>::build()
//                         .with_skip_sorting(true)
//                         .on(ref)
//                         ->generate(csr_ref);
//     auto cuda_fact = gko::factorization::Ilu<>::build()
//                          .with_skip_sorting(true)
//                          .on(cuda)
//                          ->generate(csr_cuda);

//     GKO_ASSERT_MTX_NEAR(ref_fact->get_l_factor(), cuda_fact->get_l_factor(),
//                         1e-14);
//     GKO_ASSERT_MTX_NEAR(ref_fact->get_u_factor(), cuda_fact->get_u_factor(),
//                         1e-14);
//     GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_l_factor(),
//                                cuda_fact->get_l_factor());
//     GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_u_factor(),
//                                cuda_fact->get_u_factor());
// }


// TEST_F(Ilu, ComputeILUIsEquivalentToRefUnsorted)
// {
//     gko::test::unsort_matrix(gko::lend(csr_ref), rand_engine);
//     csr_cuda->copy_from(gko::lend(csr_ref));

//     auto ref_fact =
//         gko::factorization::ParIlu<>::build().on(ref)->generate(csr_ref);
//     auto cuda_fact =
//         gko::factorization::Ilu<>::build().on(cuda)->generate(csr_cuda);

//     GKO_ASSERT_MTX_NEAR(ref_fact->get_l_factor(), cuda_fact->get_l_factor(),
//                         1e-14);
//     GKO_ASSERT_MTX_NEAR(ref_fact->get_u_factor(), cuda_fact->get_u_factor(),
//                         1e-14);
//     GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_l_factor(),
//                                cuda_fact->get_l_factor());
//     GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_u_factor(),
//                                cuda_fact->get_u_factor());
// }


}  // namespace
