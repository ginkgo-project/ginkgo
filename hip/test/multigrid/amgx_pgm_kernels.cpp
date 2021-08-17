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

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <fstream>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/multigrid/amgx_pgm_kernels.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class AmgxPgm : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    AmgxPgm() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    gko::Array<index_type> gen_array(gko::size_type num, index_type min_val,
                                     index_type max_val)
    {
        return gko::test::generate_random_array<index_type>(
            num, std::uniform_int_distribution<>(min_val, max_val), rand_engine,
            ref);
    }

    gko::Array<index_type> gen_agg_array(gko::size_type num,
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
        auto weight = gen_mtx(m, m);
        make_weight(weight.get());
        weight_csr = Csr::create(ref);
        weight->convert_to(weight_csr.get());
        weight_diag = weight_csr->extract_diagonal();
        auto system_dense = gen_mtx(m, m);
        gko::test::make_hpd(system_dense.get());
        system_mtx = Csr::create(ref);
        system_dense->convert_to(system_mtx.get());

        d_agg.set_executor(hip);
        d_unfinished_agg.set_executor(hip);
        d_strongest_neighbor.set_executor(hip);
        d_coarse_vector = Mtx::create(hip);
        d_fine_vector = Mtx::create(hip);
        d_weight_csr = Csr::create(hip);
        d_weight_diag = Diag::create(hip);
        d_system_mtx = Csr::create(hip);
        d_agg = agg;
        d_unfinished_agg = unfinished_agg;
        d_strongest_neighbor = strongest_neighbor;
        d_coarse_vector->copy_from(coarse_vector.get());
        d_fine_vector->copy_from(fine_vector.get());
        d_weight_csr->copy_from(weight_csr.get());
        d_weight_diag->copy_from(weight_diag.get());
        d_system_mtx->copy_from(system_mtx.get());
    }

    void make_weight(Mtx *mtx)
    {
        gko::test::make_symmetric(mtx);
        // only works for real value cases
        mtx->compute_absolute_inplace();
        gko::test::make_diag_dominant(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

    gko::Array<index_type> agg;
    gko::Array<index_type> unfinished_agg;
    gko::Array<index_type> strongest_neighbor;

    gko::Array<index_type> d_agg;
    gko::Array<index_type> d_unfinished_agg;
    gko::Array<index_type> d_strongest_neighbor;

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


TEST_F(AmgxPgm, MatchEdgeIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;

    gko::kernels::reference::amgx_pgm::match_edge(ref, strongest_neighbor, x);
    gko::kernels::hip::amgx_pgm::match_edge(hip, d_strongest_neighbor, d_x);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(AmgxPgm, CountUnaggIsEquivalentToRef)
{
    initialize_data();
    index_type num_unagg;
    index_type d_num_unagg;

    gko::kernels::reference::amgx_pgm::count_unagg(ref, unfinished_agg,
                                                   &num_unagg);
    gko::kernels::hip::amgx_pgm::count_unagg(hip, d_unfinished_agg,
                                             &d_num_unagg);

    ASSERT_EQ(d_num_unagg, num_unagg);
}


TEST_F(AmgxPgm, RenumberIsEquivalentToRef)
{
    initialize_data();
    index_type num_agg;
    index_type d_num_agg;

    gko::kernels::reference::amgx_pgm::renumber(ref, agg, &num_agg);
    gko::kernels::hip::amgx_pgm::renumber(hip, d_agg, &d_num_agg);

    ASSERT_EQ(d_num_agg, num_agg);
    GKO_ASSERT_ARRAY_EQ(d_agg, agg);
    ASSERT_EQ(num_agg, n);
}


TEST_F(AmgxPgm, FindStrongestNeighborIsEquivalentToRef)
{
    initialize_data();
    auto snb = strongest_neighbor;
    auto d_snb = d_strongest_neighbor;

    gko::kernels::reference::amgx_pgm::find_strongest_neighbor(
        ref, weight_csr.get(), weight_diag.get(), agg, snb);
    gko::kernels::hip::amgx_pgm::find_strongest_neighbor(
        hip, d_weight_csr.get(), d_weight_diag.get(), d_agg, d_snb);

    GKO_ASSERT_ARRAY_EQ(d_snb, snb);
}


TEST_F(AmgxPgm, AssignToExistAggIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;
    auto intermediate_agg = x;
    auto d_intermediate_agg = d_x;

    gko::kernels::reference::amgx_pgm::assign_to_exist_agg(
        ref, weight_csr.get(), weight_diag.get(), x, intermediate_agg);
    gko::kernels::hip::amgx_pgm::assign_to_exist_agg(
        hip, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(AmgxPgm, AssignToExistAggUnderteminsticIsEquivalentToRef)
{
    initialize_data();
    auto d_x = d_unfinished_agg;
    auto d_intermediate_agg = gko::Array<index_type>(hip, 0);
    index_type d_num_unagg;

    gko::kernels::hip::amgx_pgm::assign_to_exist_agg(
        hip, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);
    gko::kernels::hip::amgx_pgm::count_unagg(hip, d_agg, &d_num_unagg);

    // only test whether all elements are aggregated.
    GKO_ASSERT_EQ(d_num_unagg, 0);
}


TEST_F(AmgxPgm, GenerateMgLevelIsEquivalentToRef)
{
    initialize_data();
    auto mg_level_factory = gko::multigrid::AmgxPgm<double, int>::build()
                                .with_deterministic(true)
                                .on(ref);
    auto d_mg_level_factory = gko::multigrid::AmgxPgm<double, int>::build()
                                  .with_deterministic(true)
                                  .on(hip);

    auto mg_level = mg_level_factory->generate(system_mtx);
    auto d_mg_level = d_mg_level_factory->generate(d_system_mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_restrict_op()),
                        gko::as<Csr>(mg_level->get_restrict_op()), 1e-14);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_coarse_op()),
                        gko::as<Csr>(mg_level->get_coarse_op()), 1e-14);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_prolong_op()),
                        gko::as<Csr>(mg_level->get_prolong_op()), 1e-14);
}


}  // namespace
