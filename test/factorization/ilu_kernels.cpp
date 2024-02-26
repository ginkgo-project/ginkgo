// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    gko::test::unsort_matrix(mtx, rand_engine);
    dmtx->copy_from(mtx);

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
