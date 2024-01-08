// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


class Ic : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Ic() : rand_engine(6794)
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


TEST_F(Ic, ComputeICIsEquivalentToRefSorted)
{
    auto fact = gko::factorization::Ic<>::build()
                    .with_skip_sorting(true)
                    .on(ref)
                    ->generate(mtx);
    auto dfact = gko::factorization::Ic<>::build()
                     .with_skip_sorting(true)
                     .on(exec)
                     ->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), dfact->get_l_factor(), 1e-14);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), dfact->get_lt_factor(), 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_l_factor(), dfact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_lt_factor(), dfact->get_lt_factor());
}


TEST_F(Ic, ComputeICIsEquivalentToRefUnsorted)
{
    gko::test::unsort_matrix(mtx, rand_engine);
    dmtx->copy_from(mtx);

    auto fact = gko::factorization::Ic<>::build().on(ref)->generate(mtx);
    auto dfact = gko::factorization::Ic<>::build().on(exec)->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), dfact->get_l_factor(), 1e-14);
    GKO_ASSERT_MTX_NEAR(fact->get_lt_factor(), dfact->get_lt_factor(), 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_l_factor(), dfact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_lt_factor(), dfact->get_lt_factor());
}


TEST_F(Ic, SetsCorrectStrategy)
{
    auto dfact = gko::factorization::Ic<>::build()
                     .with_l_strategy(std::make_shared<Csr::merge_path>())
                     .on(exec)
                     ->generate(dmtx);

    ASSERT_EQ(dfact->get_l_factor()->get_strategy()->get_name(), "merge_path");
    ASSERT_EQ(dfact->get_lt_factor()->get_strategy()->get_name(), "merge_path");
}
