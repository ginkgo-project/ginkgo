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
#include "test/utils/common_fixture.hpp"


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


TEST_F(Ic, ComputeICBySyncfreeIsEquivalentToRefSorted)
{
    auto fact = gko::factorization::Ic<>::build()
                    .with_skip_sorting(true)
                    .on(ref)
                    ->generate(mtx);
    auto dfact =
        gko::factorization::Ic<>::build()
            .with_skip_sorting(true)
            .with_algorithm(gko::factorization::factorize_algorithm::syncfree)
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


TEST_F(Ic, ComputeICWithBitmapIsEquivalentToRefBySyncfree)
{
    // diag + full first row and column
    // the third and forth row use bitmap for lookup table
    auto mtx = gko::share(gko::initialize<Csr>({{1.0, 1.0, 1.0, 1.0},
                                                {1.0, 2.0, 0.0, 0.0},
                                                {1.0, 0.0, 2.0, 0.0},
                                                {1.0, 0.0, 0.0, 2.0}},
                                               this->ref));
    auto dmtx = gko::share(mtx->clone(this->exec));

    auto factory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(gko::factorization::factorize_algorithm::syncfree)
            .on(this->ref);
    auto dfactory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(gko::factorization::factorize_algorithm::syncfree)
            .on(this->exec);

    auto ic = factory->generate(mtx);
    auto dic = dfactory->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(ic->get_l_factor(), dic->get_l_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(ic->get_lt_factor(), dic->get_lt_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_l_factor(), dic->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_lt_factor(), dic->get_lt_factor());
}


TEST_F(Ic, ComputeICWithHashmapIsEquivalentToRefBySyncfree)
{
    int n = 68;
    gko::matrix_data<value_type, index_type> data(gko::dim<2>(n, n));
    for (int i = 0; i < n; i++) {
        if (i == n - 2 || i == n - 3) {
            data.nonzeros.emplace_back(i, i, value_type{2});
        } else {
            data.nonzeros.emplace_back(i, i, gko::one<value_type>());
        }
    }
    // the following rows use hashmap for lookup table
    // add dependence
    data.nonzeros.emplace_back(n - 3, 0, gko::one<value_type>());
    data.nonzeros.emplace_back(0, n - 3, gko::one<value_type>());
    // add a entry whose col idx is not shown in the above row
    data.nonzeros.emplace_back(0, n - 2, gko::one<value_type>());
    data.nonzeros.emplace_back(n - 2, 0, gko::one<value_type>());
    data.sort_row_major();
    auto mtx = gko::share(Csr::create(this->ref));
    mtx->read(data);
    auto dmtx = gko::share(mtx->clone(this->exec));
    auto factory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(gko::factorization::factorize_algorithm::syncfree)
            .on(this->ref);
    auto dfactory =
        gko::factorization::Ic<value_type, index_type>::build()
            .with_algorithm(gko::factorization::factorize_algorithm::syncfree)
            .on(this->exec);

    auto ic = factory->generate(mtx);
    auto dic = dfactory->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(ic->get_l_factor(), dic->get_l_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(ic->get_lt_factor(), dic->get_lt_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_l_factor(), dic->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(ic->get_lt_factor(), dic->get_lt_factor());
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
