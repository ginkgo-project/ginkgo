/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


class Ilu : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Ilu() : rand_engine(1337)
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        mtx = gko::read<Csr>(input_file, ref);
        dmtx = gko::clone(exec, mtx);
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> dmtx;
};


TEST_F(Ilu, ComputeILUIsEquivalentToRefSorted)
{
    auto fact = gko::factorization::Ilu<>::build()
                    .with_skip_sorting(true)
                    .on(ref)
                    ->generate(mtx);
    auto dfact = gko::factorization::Ilu<>::build()
                     .with_skip_sorting(true)
                     .on(exec)
                     ->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), dfact->get_l_factor(), 1e-14);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), dfact->get_u_factor(), 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_l_factor(), dfact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_u_factor(), dfact->get_u_factor());
}


TEST_F(Ilu, ComputeILUIsEquivalentToRefUnsorted)
{
    gko::test::unsort_matrix(mtx.get(), rand_engine);
    dmtx->copy_from(mtx.get());

    auto fact = gko::factorization::Ilu<>::build().on(ref)->generate(mtx);
    auto dfact = gko::factorization::Ilu<>::build().on(exec)->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), dfact->get_l_factor(), 1e-14);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), dfact->get_u_factor(), 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_l_factor(), dfact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_u_factor(), dfact->get_u_factor());
}


TEST_F(Ilu, SetsCorrectStrategy)
{
    auto dfact = gko::factorization::Ilu<>::build()
                     .with_l_strategy(std::make_shared<Csr::merge_path>())
                     .with_u_strategy(std::make_shared<Csr::load_balance>(exec))
                     .on(exec)
                     ->generate(dmtx);

    ASSERT_EQ(dfact->get_l_factor()->get_strategy()->get_name(), "merge_path");
    ASSERT_EQ(dfact->get_u_factor()->get_strategy()->get_name(),
              "load_balance");
}
