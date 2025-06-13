// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <memory>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/log/logger.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "matrices/config.hpp"
#include "test/utils/common_fixture.hpp"


struct CheckOperationLogger : gko::log::Logger {
    void on_operation_launched(const gko::Executor*,
                               const gko::Operation* op) const override
    {
        std::string s = op->get_name();
        if (s.find("sparselib") != std::string::npos) {
            contains_sparselib = true;
        }
    }

    mutable bool contains_sparselib = false;
};


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


TEST_F(Ilu, UsesSyncFreeAlgorithm)
{
    auto logger = std::make_shared<CheckOperationLogger>();
    exec->add_logger(logger);

    auto dfact =
        gko::factorization::Ilu<>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec)
            ->generate(dmtx);

    ASSERT_FALSE(logger->contains_sparselib);
}


TEST_F(Ilu, UsesSparseLibAlgorithm)
{
    auto logger = std::make_shared<CheckOperationLogger>();
    exec->add_logger(logger);

    auto dfact =
        gko::factorization::Ilu<>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::sparselib)
            .on(exec)
            ->generate(dmtx);

#ifdef GKO_COMPILING_OMP
    // OMP does not have sparselib algorithm
    ASSERT_FALSE(logger->contains_sparselib);
#else
    ASSERT_TRUE(logger->contains_sparselib);
#endif
}


TEST_F(Ilu, ComputeILUBySyncfreeIsEquivalentToRefSorted)
{
    auto fact =
        gko::factorization::Ilu<>::build()
            .with_skip_sorting(true)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(ref)
            ->generate(mtx);
    auto dfact =
        gko::factorization::Ilu<>::build()
            .with_skip_sorting(true)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec)
            ->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(fact->get_l_factor(), dfact->get_l_factor(), 1e-14);
    GKO_ASSERT_MTX_NEAR(fact->get_u_factor(), dfact->get_u_factor(), 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_l_factor(), dfact->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(fact->get_u_factor(), dfact->get_u_factor());
}


TEST_F(Ilu, ComputeILUWithBitmapIsEquivalentToRefBySyncfree)
{
    // diag + full first row and column
    // the third and forth row use bitmap for lookup table
    auto mtx = gko::share(gko::initialize<Csr>({{1.0, 1.0, 1.0, 1.0},
                                                {1.0, 1.0, 0.0, 0.0},
                                                {1.0, 0.0, 1.0, 0.0},
                                                {1.0, 0.0, 0.0, 1.0}},
                                               this->ref));
    auto dmtx = gko::share(mtx->clone(this->exec));

    auto factory =
        gko::factorization::Ilu<value_type, index_type>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);
    auto dfactory =
        gko::factorization::Ilu<value_type, index_type>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->exec);

    auto ilu = factory->generate(mtx);
    auto dilu = dfactory->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(ilu->get_l_factor(), dilu->get_l_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(ilu->get_u_factor(), dilu->get_u_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(ilu->get_l_factor(), dilu->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(ilu->get_u_factor(), dilu->get_u_factor());
}


TEST_F(Ilu, ComputeILUWithHashmapIsEquivalentToRefBySyncfree)
{
    int n = 68;
    // the first row and second last row use hashmap for lookup table
    gko::matrix_data<value_type, index_type> data(gko::dim<2>(n, n));
    for (int i = 0; i < n; i++) {
        data.nonzeros.emplace_back(i, i, gko::one<value_type>());
    }
    // add dependence
    data.nonzeros.emplace_back(n - 3, 0, gko::one<value_type>());
    // add a entry whose col idx is not shown in the above row
    data.nonzeros.emplace_back(0, n - 2, gko::one<value_type>());
    data.sort_row_major();
    auto mtx = gko::share(Csr::create(this->ref));
    mtx->read(data);
    auto dmtx = gko::share(mtx->clone(this->exec));
    auto factory =
        gko::factorization::Ilu<value_type, index_type>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);
    auto dfactory =
        gko::factorization::Ilu<value_type, index_type>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->exec);

    auto ilu = factory->generate(mtx);
    auto dilu = dfactory->generate(dmtx);

    GKO_ASSERT_MTX_NEAR(ilu->get_l_factor(), dilu->get_l_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(ilu->get_u_factor(), dilu->get_u_factor(),
                        r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(ilu->get_l_factor(), dilu->get_l_factor());
    GKO_ASSERT_MTX_EQ_SPARSITY(ilu->get_u_factor(), dilu->get_u_factor());
}


TEST_F(Ilu, SetsCorrectStrategy)
{
    auto dfact =
        gko::factorization::Ilu<>::build()
            .with_l_strategy(std::make_shared<Csr::merge_path>())
#ifdef GKO_COMPILING_OMP
            .with_u_strategy(std::make_shared<Csr::merge_path>())
#else
            .with_u_strategy(std::make_shared<Csr::load_balance>(exec))
#endif
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec)
            ->generate(dmtx);

    ASSERT_EQ(dfact->get_l_factor()->get_strategy()->get_name(), "merge_path");
#ifdef GKO_COMPILING_OMP
    ASSERT_EQ(dfact->get_u_factor()->get_strategy()->get_name(), "merge_path");
#else
    ASSERT_EQ(dfact->get_u_factor()->get_strategy()->get_name(),
              "load_balance");
#endif
}


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
