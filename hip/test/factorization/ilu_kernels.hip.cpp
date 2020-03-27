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

#include <ginkgo/core/factorization/ilu.hpp>


#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>


#include "hip/test/utils.hip.hpp"
#include "matrices/config.hpp"


namespace {


class Ilu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    std::shared_ptr<Csr> csr_ref;
    std::shared_ptr<Csr> csr_hip;

    Ilu()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref))
    {}

    void SetUp() override
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        csr_ref = gko::read<Csr>(input_file, ref);
        csr_hip = Csr::create(hip);
        csr_hip->copy_from(gko::lend(csr_ref));
    }
};


TEST_F(Ilu, ComputeILUIsEquivalentToRef)
{
    auto ref_fact =
        gko::factorization::ParIlu<>::build().on(ref)->generate(csr_ref);
    auto hip_fact =
        gko::factorization::Ilu<>::build().on(hip)->generate(csr_hip);

    GKO_ASSERT_MTX_NEAR(ref_fact->get_l_factor(), hip_fact->get_l_factor(),
                        1e-14);
    GKO_ASSERT_MTX_NEAR(ref_fact->get_u_factor(), hip_fact->get_u_factor(),
                        1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_l_factor(),
                               hip_fact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(ref_fact->get_u_factor(),
                               hip_fact->get_u_factor());
}


TEST_F(Ilu, SetsCorrectStrategy)
{
    // workaround for NVCC disabled with_* functions
    auto hip_fact_fact = gko::factorization::Ilu<>::build();
    hip_fact_fact.l_strategy = std::make_shared<Csr::merge_path>();
    hip_fact_fact.u_strategy = std::make_shared<Csr::load_balance>(hip);
    auto hip_fact = hip_fact_fact.on(hip)->generate(csr_hip);

    ASSERT_EQ(hip_fact->get_l_factor()->get_strategy()->get_name(),
              "merge_path");
    ASSERT_EQ(hip_fact->get_u_factor()->get_strategy()->get_name(),
              "load_balance");
}


}  // namespace
