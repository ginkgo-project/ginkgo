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

#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/bddc.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueIndexType>
class BddcFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using real_type = typename gko::remove_complex<value_type>;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Bddc =
        gko::experimental::distributed::preconditioner::Bddc<value_type,
                                                             index_type>;
    using Direct = gko::experimental::solver::Direct<value_type, index_type>;
    using Lu = gko::experimental::factorization::Lu<value_type, index_type>;
    using Gmres = gko::solver::Gmres<value_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, index_type>;
    using Mtx = gko::experimental::distributed::Matrix<value_type, index_type,
                                                       index_type>;
    using Vec = gko::experimental::distributed::Vector<value_type>;
    using LocalMtx = gko::matrix::Csr<value_type, index_type>;
    using LocalVec = gko::matrix::Dense<value_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    using mat_data = gko::matrix_data<value_type, index_type>;
    using part =
        gko::experimental::distributed::Partition<index_type, index_type>;

    BddcFactory()
        : exec(gko::ReferenceExecutor::create()),
          jacobi_factory(Jacobi::build().on(exec)),
          mtx(Mtx::create(exec, MPI_COMM_WORLD)),
          interface_dofs{{3},      {21},     {24},     {27},    {45},
                         {10, 17}, {22, 23}, {25, 26}, {31, 38}},
          interface_dof_ranks{{0, 1}, {0, 2}, {0, 1, 2, 3}, {1, 3}, {2, 3},
                              {0, 1}, {0, 2}, {1, 3},       {2, 3}}
    {
        auto rank = mtx->get_communicator().rank();
        auto local_data = generate_local_data(rank);
        auto partition = generate_partition();
        mtx->read_distributed(local_data, partition.get());
        expected_local_system = read_expected_local_system(rank);
        expected_phi = read_expected_phi(rank);
        expected_coarse_system = read_expected_coarse(rank);
        rhs = generate_rhs();
        gmres_factory = gko::share(
            Gmres::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                    gko::stop::ResidualNorm<real_type>::build()
                        .with_reduction_factor(real_type{1e-8})
                        .on(exec))
                .on(exec));
        /*gmres_factory = gko::share(
            Direct::build()
                .with_factorization(
                    Lu::build()
                        .with_symmetric_sparsity(true)
                        .on(exec))
                .on(exec));*/
        bddc = Bddc::build()
                   .with_local_solver_factory(gmres_factory)
                   .with_schur_complement_solver_factory(gmres_factory)
                   .with_coarse_solver_factory(gmres_factory)
                   .with_inner_solver_factory(gmres_factory)
                   .with_interface_dofs(interface_dofs)
                   .with_interface_dof_ranks(interface_dof_ranks)
                   .with_static_condensation(true)
                   .on(exec)
                   ->generate(mtx);
    }

    std::shared_ptr<part> generate_partition()
    {
        gko::array<comm_index_type> mapping{
            this->exec, {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                         0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3,
                         3, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3}};
        index_type num_parts = 4;
        auto partition = gko::share(
            part::build_from_mapping(this->exec, mapping, num_parts));

        return partition;
    }

    mat_data generate_local_data(int rank)
    {
        const char* input_name;
        if (rank == 0) input_name = gko::matrices::location_bddc_input_0;
        if (rank == 1) input_name = gko::matrices::location_bddc_input_1;
        if (rank == 2) input_name = gko::matrices::location_bddc_input_2;
        if (rank == 3) input_name = gko::matrices::location_bddc_input_3;

        std::ifstream in{input_name};
        mat_data local_data = gko::read_raw<value_type, index_type>(in);
        return local_data;
    }

    std::shared_ptr<LocalMtx> read_expected_local_system(int rank)
    {
        const char* input_name;
        if (rank == 0) input_name = gko::matrices::location_bddc_local_sys_0;
        if (rank == 1) input_name = gko::matrices::location_bddc_local_sys_1;
        if (rank == 2) input_name = gko::matrices::location_bddc_local_sys_2;
        if (rank == 3) input_name = gko::matrices::location_bddc_local_sys_3;

        std::ifstream in{input_name};
        auto expected = gko::share(gko::read<LocalMtx>(in, this->exec));
        return expected;
    }

    std::shared_ptr<LocalVec> read_expected_phi(int rank)
    {
        const char* input_name;
        if (rank == 0) input_name = gko::matrices::location_bddc_phi_0;
        if (rank == 1) input_name = gko::matrices::location_bddc_phi_1;
        if (rank == 2) input_name = gko::matrices::location_bddc_phi_2;
        if (rank == 3) input_name = gko::matrices::location_bddc_phi_3;

        std::ifstream in{input_name};
        auto expected = gko::share(gko::read<LocalVec>(in, this->exec));
        return expected;
    }

    std::shared_ptr<LocalVec> read_expected_coarse(int rank)
    {
        const char* input_name;
        if (rank == 0) input_name = gko::matrices::location_bddc_coarse_0;
        if (rank == 1) input_name = gko::matrices::location_bddc_coarse_1;
        if (rank == 2) input_name = gko::matrices::location_bddc_coarse_2;
        if (rank == 3) input_name = gko::matrices::location_bddc_coarse_3;

        std::ifstream in{input_name};
        auto expected = gko::share(gko::read<LocalVec>(in, this->exec));
        return expected;
    }

    std::shared_ptr<Vec> generate_rhs()
    {
        mat_data v_data{gko::dim<2>{49, 1}};
        for (auto i = 1; i < 6; i++) {
            for (auto j = 1; j < 6; j++) {
                v_data.nonzeros.emplace_back(i + j * 7, 0,
                                             gko::one<value_type>());
            }
        }

        auto v =
            gko::share(Vec::create(this->exec, this->mtx->get_communicator()));
        v->read_distributed(v_data, this->mtx->get_row_partition().get());

        return v;
    }

    void assert_correct_weights(const Diag* weights)
    {
        ASSERT_EQ(weights->get_size(), gko::dim<2>(16, 16));

        auto rank = this->mtx->get_communicator().rank();
        std::vector<value_type> w;
        if (rank == 0)
            w = std::vector<value_type>{1, 1, 1, .5, 1,  1,  1,  .5,
                                        1, 1, 1, .5, .5, .5, .5, .25};
        if (rank == 1)
            w = std::vector<value_type>{.5, 1, 1, 1, .5,  1,  1,  1,
                                        .5, 1, 1, 1, .25, .5, .5, .5};
        if (rank == 2)
            w = std::vector<value_type>{.5, .5, .5, .25, 1, 1, 1, .5,
                                        1,  1,  1,  .5,  1, 1, 1, .5};
        if (rank == 3)
            w = std::vector<value_type>{.25, .5, .5, .5, .5, 1, 1, 1,
                                        .5,  1,  1,  1,  .5, 1, 1, 1};

        for (auto i = 0; i < 16; i++) {
            EXPECT_EQ(w[i], weights->get_const_values()[i]);
        }
    }

    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        std::copy(std::begin(vals), std::end(vals), arr);
    }

    void assert_same_precond(const Bddc* a, const Bddc* b)
    {
        ASSERT_EQ(a->get_size()[0], b->get_size()[0]);
        ASSERT_EQ(a->get_size()[1], b->get_size()[1]);
        ASSERT_EQ(a->get_parameters().local_solver_factory,
                  b->get_parameters().local_solver_factory);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Bddc> bddc;
    std::shared_ptr<typename Jacobi::Factory> jacobi_factory;
    std::shared_ptr<typename Gmres::Factory> gmres_factory;
    std::shared_ptr<Mtx> mtx;
    std::vector<std::vector<index_type>> interface_dofs;
    std::vector<std::vector<index_type>> interface_dof_ranks;
    std::shared_ptr<LocalMtx> expected_local_system;
    std::shared_ptr<LocalVec> expected_phi;
    std::shared_ptr<LocalVec> expected_coarse_system;
    std::shared_ptr<Vec> rhs;
};

TYPED_TEST_SUITE(BddcFactory, gko::test::RealValueIndexTypes);


TYPED_TEST(BddcFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->bddc->get_executor(), this->exec);
}


TYPED_TEST(BddcFactory, CanSetLocalFactory)
{
    ASSERT_EQ(this->bddc->get_parameters().local_solver_factory,
              this->gmres_factory);
}


TYPED_TEST(BddcFactory, CanBeCloned)
{
    auto bddc_clone = clone(this->bddc);

    this->assert_same_precond(lend(bddc_clone), lend(this->bddc));
}


/*TYPED_TEST(BddcFactory, CanBeCopied)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Bddc = typename TestFixture::Bddc;
    using Mtx = typename TestFixture::Mtx;
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto copy = Bddc::build()
                    .with_local_solver_factory(bj)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(lend(this->bddc));

    this->assert_same_precond(lend(copy), lend(this->bddc));
}


TYPED_TEST(BddcFactory, CanBeMoved)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Bddc = typename TestFixture::Bddc;
    using Mtx = typename TestFixture::Mtx;
    auto tmp = clone(this->bddc);
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto copy = Bddc::build()
                    .with_local_solver_factory(bj)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(give(this->bddc));

    this->assert_same_precond(lend(copy), lend(tmp));
}*/


TYPED_TEST(BddcFactory, CanBeCleared)
{
    this->bddc->clear();

    ASSERT_EQ(this->bddc->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->bddc->get_parameters().local_solver_factory, nullptr);
}


TYPED_TEST(BddcFactory, GeneratesCorrectly)
{
    GKO_ASSERT_MTX_NEAR(this->expected_local_system,
                        this->bddc->get_local_system_matrix(), 1e-14);
    this->assert_correct_weights(this->bddc->get_weights().get());
    GKO_ASSERT_MTX_NEAR(this->expected_phi, this->bddc->get_phi(), 1e-5);
    GKO_ASSERT_MTX_NEAR(this->expected_coarse_system,
                        this->bddc->get_local_coarse_matrix(), 1e-5);
}


TYPED_TEST(BddcFactory, AppliesCorrectly)
{
    auto lhs = gko::clone(this->exec, this->rhs);
    this->bddc->apply(this->rhs.get(), lhs.get());
}


}  // namespace
