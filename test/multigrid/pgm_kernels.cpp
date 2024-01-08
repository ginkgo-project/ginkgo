// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"


#include <fstream>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class Pgm : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using RowGatherer = gko::matrix::RowGatherer<index_type>;
    using Diag = gko::matrix::Diagonal<value_type>;

    Pgm() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    gko::array<index_type> gen_array(gko::size_type num, index_type min_val,
                                     index_type max_val)
    {
        return gko::test::generate_random_array<index_type>(
            num, std::uniform_int_distribution<>(min_val, max_val), rand_engine,
            ref);
    }

    gko::array<index_type> gen_agg_array(gko::size_type num,
                                         gko::size_type num_agg)
    {
        auto agg_array = gen_array(num, 0, num_agg - 1);
        auto agg_array_val = agg_array.get_data();
        std::vector<index_type> select_agg(num);
        std::iota(select_agg.begin(), select_agg.end(), 0);
        // use the first num_agg item as the aggregated index.
        std::shuffle(select_agg.begin(), select_agg.end(), rand_engine);
        // the value of agg_array is the i-th of aggregate group
        for (gko::size_type i = 0; i < num; i++) {
            agg_array_val[i] = select_agg[agg_array_val[i]];
        }
        // the aggregated group must contain the identifier-th element
        // agg_val[i] == i holds in the aggregated group whose identifier is i
        for (gko::size_type i = 0; i < num_agg; i++) {
            auto agg_idx = select_agg[i];
            agg_array_val[agg_idx] = agg_idx;
        }
        return agg_array;
    }

    void initialize_data()
    {
        m = 597;
        n = 300;
        int nrhs = 3;

        agg = gen_agg_array(m, n);
        // only use 0 ~ n-2 and ensure the end isolated and not yet finished
        unfinished_agg = gen_array(m, -1, n - 2);
        unfinished_agg.get_data()[n - 1] = -1;
        strongest_neighbor = gen_array(m, 0, n - 2);
        strongest_neighbor.get_data()[n - 1] = n - 1;
        coarse_vector = gen_mtx(n, nrhs);
        fine_vector = gen_mtx(m, nrhs);

        auto weight_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                m, m, std::uniform_int_distribution<>(m, m),
                std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::utils::make_symmetric(weight_data);
        gko::utils::make_diag_dominant(weight_data);
        weight_csr = Csr::create(ref);
        weight_csr->read(weight_data);
        // only works for real value cases.
        weight_csr->compute_absolute_inplace();
        weight_diag = weight_csr->extract_diagonal();

        auto system_data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                m, m, std::uniform_int_distribution<>(m, m),
                std::normal_distribution<value_type>(-1.0, 1.0), rand_engine);
        gko::utils::make_hpd(system_data);
        system_mtx = Csr::create(ref);
        system_mtx->read(system_data);

        d_agg = gko::array<index_type>(exec, agg);
        d_unfinished_agg = gko::array<index_type>(exec, unfinished_agg);
        d_strongest_neighbor = gko::array<index_type>(exec, strongest_neighbor);
        d_coarse_vector = gko::clone(exec, coarse_vector);
        d_fine_vector = gko::clone(exec, fine_vector);
        d_weight_csr = gko::clone(exec, weight_csr);
        d_weight_diag = gko::clone(exec, weight_diag);
        d_system_mtx = gko::clone(exec, system_mtx);
    }

    std::default_random_engine rand_engine;

    gko::array<index_type> agg;
    gko::array<index_type> unfinished_agg;
    gko::array<index_type> strongest_neighbor;

    gko::array<index_type> d_agg;
    gko::array<index_type> d_unfinished_agg;
    gko::array<index_type> d_strongest_neighbor;

    std::unique_ptr<Mtx> coarse_vector;
    std::unique_ptr<Mtx> fine_vector;
    std::unique_ptr<Diag> weight_diag;
    std::unique_ptr<Csr> weight_csr;
    std::shared_ptr<Csr> system_mtx;

    std::unique_ptr<Mtx> d_coarse_vector;
    std::unique_ptr<Mtx> d_fine_vector;
    std::unique_ptr<Diag> d_weight_diag;
    std::unique_ptr<Csr> d_weight_csr;
    std::shared_ptr<Csr> d_system_mtx;

    gko::size_type n;
    gko::size_type m;
};


TEST_F(Pgm, MatchEdgeIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;

    gko::kernels::reference::pgm::match_edge(ref, strongest_neighbor, x);
    gko::kernels::EXEC_NAMESPACE::pgm::match_edge(exec, d_strongest_neighbor,
                                                  d_x);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(Pgm, CountUnaggIsEquivalentToRef)
{
    initialize_data();
    index_type num_unagg;
    index_type d_num_unagg;

    gko::kernels::reference::pgm::count_unagg(ref, unfinished_agg, &num_unagg);
    gko::kernels::EXEC_NAMESPACE::pgm::count_unagg(exec, d_unfinished_agg,
                                                   &d_num_unagg);

    ASSERT_EQ(d_num_unagg, num_unagg);
}


TEST_F(Pgm, RenumberIsEquivalentToRef)
{
    initialize_data();
    index_type num_agg;
    index_type d_num_agg;

    gko::kernels::reference::pgm::renumber(ref, agg, &num_agg);
    gko::kernels::EXEC_NAMESPACE::pgm::renumber(exec, d_agg, &d_num_agg);

    ASSERT_EQ(d_num_agg, num_agg);
    GKO_ASSERT_ARRAY_EQ(d_agg, agg);
    ASSERT_EQ(num_agg, n);
}


TEST_F(Pgm, FindStrongestNeighborIsEquivalentToRef)
{
    initialize_data();
    auto snb = strongest_neighbor;
    auto d_snb = d_strongest_neighbor;

    gko::kernels::reference::pgm::find_strongest_neighbor(
        ref, weight_csr.get(), weight_diag.get(), agg, snb);
    gko::kernels::EXEC_NAMESPACE::pgm::find_strongest_neighbor(
        exec, d_weight_csr.get(), d_weight_diag.get(), d_agg, d_snb);

    GKO_ASSERT_ARRAY_EQ(d_snb, snb);
}


TEST_F(Pgm, AssignToExistAggIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;
    auto intermediate_agg = x;
    auto d_intermediate_agg = d_x;

    gko::kernels::reference::pgm::assign_to_exist_agg(
        ref, weight_csr.get(), weight_diag.get(), x, intermediate_agg);
    gko::kernels::EXEC_NAMESPACE::pgm::assign_to_exist_agg(
        exec, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(Pgm, AssignToExistAggUnderteminsticIsEquivalentToRef)
{
    initialize_data();
    auto d_x = d_unfinished_agg;
    auto d_intermediate_agg = gko::array<index_type>(exec, 0);
    index_type d_num_unagg;

    gko::kernels::EXEC_NAMESPACE::pgm::assign_to_exist_agg(
        exec, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);
    gko::kernels::EXEC_NAMESPACE::pgm::count_unagg(exec, d_agg, &d_num_unagg);

    // only test whether all elements are aggregated.
    GKO_ASSERT_EQ(d_num_unagg, 0);
}


TEST_F(Pgm, GenerateMgLevelIsEquivalentToRef)
{
    initialize_data();
    auto mg_level_factory = gko::multigrid::Pgm<value_type, int>::build()
                                .with_deterministic(true)
                                .with_skip_sorting(true)
                                .on(ref);
    auto d_mg_level_factory = gko::multigrid::Pgm<value_type, int>::build()
                                  .with_deterministic(true)
                                  .with_skip_sorting(true)
                                  .on(exec);

    auto mg_level = mg_level_factory->generate(system_mtx);
    auto d_mg_level = d_mg_level_factory->generate(d_system_mtx);
    auto row_gatherer = gko::as<RowGatherer>(mg_level->get_prolong_op());
    auto d_row_gatherer = gko::as<RowGatherer>(d_mg_level->get_prolong_op());
    auto row_gather_view = gko::array<index_type>::const_view(
        row_gatherer->get_executor(), row_gatherer->get_size()[0],
        row_gatherer->get_const_row_idxs());
    auto d_row_gather_view = gko::array<index_type>::const_view(
        d_row_gatherer->get_executor(), d_row_gatherer->get_size()[0],
        d_row_gatherer->get_const_row_idxs());

    GKO_ASSERT_MTX_NEAR(gko::as<SparsityCsr>(d_mg_level->get_restrict_op()),
                        gko::as<SparsityCsr>(mg_level->get_restrict_op()),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_coarse_op()),
                        gko::as<Csr>(mg_level->get_coarse_op()),
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_row_gather_view, row_gather_view);
}


TEST_F(Pgm, GenerateMgLevelIsEquivalentToRefOnUnsortedMatrix)
{
    initialize_data();
    gko::test::unsort_matrix(system_mtx, rand_engine);
    d_system_mtx = gko::clone(exec, system_mtx);
    auto mg_level_factory = gko::multigrid::Pgm<value_type, int>::build()
                                .with_deterministic(true)
                                .on(ref);
    auto d_mg_level_factory = gko::multigrid::Pgm<value_type, int>::build()
                                  .with_deterministic(true)
                                  .on(exec);

    auto mg_level = mg_level_factory->generate(system_mtx);
    auto d_mg_level = d_mg_level_factory->generate(d_system_mtx);
    auto row_gatherer = gko::as<RowGatherer>(mg_level->get_prolong_op());
    auto d_row_gatherer = gko::as<RowGatherer>(d_mg_level->get_prolong_op());
    auto row_gather_view = gko::array<index_type>::const_view(
        row_gatherer->get_executor(), row_gatherer->get_size()[0],
        row_gatherer->get_const_row_idxs());
    auto d_row_gather_view = gko::array<index_type>::const_view(
        d_row_gatherer->get_executor(), d_row_gatherer->get_size()[0],
        d_row_gatherer->get_const_row_idxs());

    GKO_ASSERT_MTX_NEAR(gko::as<SparsityCsr>(d_mg_level->get_restrict_op()),
                        gko::as<SparsityCsr>(mg_level->get_restrict_op()),
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_coarse_op()),
                        gko::as<Csr>(mg_level->get_coarse_op()),
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_row_gather_view, row_gather_view);
}
