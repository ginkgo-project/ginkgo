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

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <fstream>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/multigrid/amgx_pgm_kernels.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "cuda/test/utils.hpp"


namespace {


template <typename Array, typename ValueDistribution, typename Engine>
Array generate_random_array(gko::size_type num, ValueDistribution &&value_dist,
                            Engine &&engine,
                            std::shared_ptr<const gko::Executor> exec)
{
    using value_type = typename Array::value_type;
    Array array_host(exec->get_master(), num);
    auto val = array_host.get_data();
    for (int i = 0; i < num; i++) {
        val[i] =
            gko::test::detail::get_rand_value<value_type>(value_dist, engine);
    }
    Array array(exec);
    array = array_host;
    return array;
}


class AmgxPgm : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    AmgxPgm() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
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
        return generate_random_array<gko::Array<index_type>>(
            num, std::uniform_int_distribution<>(min_val, max_val), rand_engine,
            ref);
    }

    void initialize_data()
    {
        int m = 597;
        n = 300;
        int nrhs = 3;

        agg = gen_array(m, 0, n - 1);
        unfinished_agg = gen_array(m, -1, n - 1);
        strongest_neighbor = gen_array(m, 0, n - 1);
        coarse_vector = gen_mtx(n, nrhs);
        fine_vector = gen_mtx(m, nrhs);
        auto weight = gen_mtx(m, m);
        make_weight(weight.get());
        weight_csr = Csr::create(ref);
        weight->convert_to(weight_csr.get());
        weight_diag = weight_csr->extract_diagonal();

        d_agg.set_executor(cuda);
        d_unfinished_agg.set_executor(cuda);
        d_strongest_neighbor.set_executor(cuda);
        d_coarse_vector = Mtx::create(cuda);
        d_fine_vector = Mtx::create(cuda);
        d_weight_csr = Csr::create(cuda);
        d_weight_diag = Mtx::create(cuda);
        d_agg = agg;
        d_unfinished_agg = unfinished_agg;
        d_strongest_neighbor = strongest_neighbor;
        d_coarse_vector->copy_from(coarse_vector.get());
        d_fine_vector->copy_from(fine_vector.get());
        d_weight_csr->copy_from(weight_csr.get());
        d_weight_diag->copy_from(weight_diag.get());
    }

    void make_symetric(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            for (int j = i + 1; j < mtx->get_size()[1]; ++j) {
                mtx->at(i, j) = mtx->at(j, i);
            }
        }
    }

    // only for real value
    void make_absoulte(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            for (int j = 0; j < mtx->get_size()[1]; ++j) {
                mtx->at(i, j) = abs(mtx->at(i, j));
            }
        }
    }

    void make_diag_dominant(Mtx *mtx)
    {
        using std::abs;
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            auto sum = gko::zero<Mtx::value_type>();
            for (int j = 0; j < mtx->get_size()[1]; ++j) {
                sum += abs(mtx->at(i, j));
            }
            mtx->at(i, i) = sum;
        }
    }

    void make_spd(Mtx *mtx)
    {
        make_symetric(mtx);
        make_diag_dominant(mtx);
    }

    void make_weight(Mtx *mtx)
    {
        make_symetric(mtx);
        make_absoulte(mtx);
        make_diag_dominant(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    gko::Array<index_type> agg;
    gko::Array<index_type> unfinished_agg;
    gko::Array<index_type> strongest_neighbor;

    gko::Array<index_type> d_agg;
    gko::Array<index_type> d_unfinished_agg;
    gko::Array<index_type> d_strongest_neighbor;

    std::unique_ptr<Mtx> coarse_vector;
    std::unique_ptr<Mtx> fine_vector;
    std::unique_ptr<Mtx> weight_diag;
    std::unique_ptr<Csr> weight_csr;

    std::unique_ptr<Mtx> d_coarse_vector;
    std::unique_ptr<Mtx> d_fine_vector;
    std::unique_ptr<Mtx> d_weight_diag;
    std::unique_ptr<Csr> d_weight_csr;

    int n;
};


TEST_F(AmgxPgm, RestrictApplyIsEquivalentToRef)
{
    initialize_data();
    // fine->coarse
    auto x = Mtx::create_with_config_of(gko::lend(coarse_vector));
    auto d_x = Mtx::create_with_config_of(gko::lend(d_coarse_vector));

    gko::kernels::reference::amgx_pgm::restrict_apply(
        ref, agg, fine_vector.get(), x.get());
    gko::kernels::cuda::amgx_pgm::restrict_apply(
        cuda, d_agg, d_fine_vector.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(AmgxPgm, ProlongApplyaddIsEquivalentToRef)
{
    initialize_data();
    // coarse->fine
    auto x = fine_vector->clone();
    auto d_x = d_fine_vector->clone();

    gko::kernels::reference::amgx_pgm::prolong_applyadd(
        ref, agg, coarse_vector.get(), x.get());
    gko::kernels::cuda::amgx_pgm::prolong_applyadd(
        cuda, d_agg, d_coarse_vector.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(AmgxPgm, MatchEdgeIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;

    gko::kernels::reference::amgx_pgm::match_edge(ref, strongest_neighbor, x);
    gko::kernels::cuda::amgx_pgm::match_edge(cuda, d_strongest_neighbor, d_x);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(AmgxPgm, CountUnaggIsEquivalentToRef)
{
    initialize_data();
    gko::size_type num_unagg;
    gko::size_type d_num_unagg;

    gko::kernels::reference::amgx_pgm::count_unagg(ref, agg, &num_unagg);
    gko::kernels::cuda::amgx_pgm::count_unagg(cuda, d_agg, &d_num_unagg);

    ASSERT_EQ(d_num_unagg, num_unagg);
}


TEST_F(AmgxPgm, RenumberIsEquivalentToRef)
{
    initialize_data();
    auto x = unfinished_agg;
    auto d_x = d_unfinished_agg;
    gko::size_type num_agg;
    gko::size_type d_num_agg;

    gko::kernels::reference::amgx_pgm::renumber(ref, agg, &num_agg);
    gko::kernels::cuda::amgx_pgm::renumber(cuda, d_agg, &d_num_agg);

    ASSERT_EQ(d_num_agg, num_agg);
    GKO_ASSERT_ARRAY_EQ(d_agg, agg);
    ASSERT_LE(num_agg, 300);
}


TEST_F(AmgxPgm, FindStrongestNeighborIsEquivalentToRef)
{
    initialize_data();
    auto snb = strongest_neighbor;
    auto d_snb = d_strongest_neighbor;

    gko::kernels::reference::amgx_pgm::find_strongest_neighbor(
        ref, weight_csr.get(), weight_diag.get(), agg, snb);
    gko::kernels::cuda::amgx_pgm::find_strongest_neighbor(
        cuda, d_weight_csr.get(), d_weight_diag.get(), d_agg, d_snb);

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
    gko::kernels::cuda::amgx_pgm::assign_to_exist_agg(
        cuda, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);

    GKO_ASSERT_ARRAY_EQ(d_x, x);
}


TEST_F(AmgxPgm, AssignToExistAggUnderteminsticIsEquivalentToRef)
{
    initialize_data();
    auto d_x = d_unfinished_agg;
    auto d_intermediate_agg = gko::Array<index_type>(cuda, 0);
    gko::size_type d_num_unagg;

    gko::kernels::cuda::amgx_pgm::assign_to_exist_agg(
        cuda, d_weight_csr.get(), d_weight_diag.get(), d_x, d_intermediate_agg);
    gko::kernels::cuda::amgx_pgm::count_unagg(cuda, d_agg, &d_num_unagg);

    // only test whether all elements are aggregated.
    GKO_ASSERT_EQ(d_num_unagg, 0);
}


TEST_F(AmgxPgm, GenerateMtxIsEquivalentToRef)
{
    initialize_data();
    auto csr_coarse = Csr::create(ref, gko::dim<2>{n, n}, 0);
    auto d_csr_coarse = Csr::create(cuda, gko::dim<2>{n, n}, 0);
    auto csr_temp = Csr::create(ref, gko::dim<2>{n, n},
                                weight_csr->get_num_stored_elements());
    auto d_csr_temp = Csr::create(cuda, gko::dim<2>{n, n},
                                  d_weight_csr->get_num_stored_elements());

    gko::kernels::cuda::amgx_pgm::amgx_pgm_generate(
        cuda, d_weight_csr.get(), d_agg, d_csr_coarse.get(), d_csr_temp.get());
    gko::kernels::reference::amgx_pgm::amgx_pgm_generate(
        ref, weight_csr.get(), agg, csr_coarse.get(), csr_temp.get());

    GKO_ASSERT_MTX_NEAR(d_csr_coarse, csr_coarse, 1e-14);
}


}  // namespace
