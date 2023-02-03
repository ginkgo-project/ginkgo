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

#include <array>
#include <memory>
#include <random>


#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/mpi/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class SchwarzPreconditioner : public CommonMpiTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<2, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_vec_type = gko::matrix::Dense<value_type>;
    using dist_prec_type =
        gko::experimental::distributed::preconditioner::Schwarz<
            value_type, local_index_type, global_index_type>;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    SchwarzPreconditioner()
        : size{8, 8},
          mat_input{size,
                    {{0, 0, 2},  {0, 1, -1}, {1, 0, -1}, {1, 1, 2},  {1, 2, -1},
                     {2, 1, -1}, {2, 2, 2},  {2, 3, -1}, {3, 2, -1}, {3, 3, 2},
                     {3, 4, -1}, {4, 3, -1}, {4, 4, 2},  {4, 5, -1}, {5, 4, -1},
                     {5, 5, 2},  {5, 6, -1}, {6, 5, -1}, {6, 6, 2},  {6, 7, -1},
                     {7, 6, -1}, {7, 7, 2}}},
          engine(42)
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part.get());

        auto nrhs = 1;
        auto global_size =
            gko::dim<2>{size[0], static_cast<gko::size_type>(nrhs)};
        auto local_size = gko::dim<2>{
            static_cast<gko::size_type>(row_part->get_part_size(comm.rank())),
            static_cast<gko::size_type>(nrhs)};
        auto dist_result =
            dist_vec_type::create(ref, comm, global_size, local_size, nrhs);
        dist_result->read_distributed(
            gen_dense_data<typename dist_vec_type::value_type,
                           global_index_type>(global_size),
            row_part.get());
        dist_b = gko::share(gko::clone(exec, dist_result));
        dist_x = gko::share(gko::clone(exec, dist_result));
        dist_x->fill(gko::zero<value_type>());

        local_solver_factory =
            local_prec_type::build().with_max_block_size(1u).on(exec);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }


    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            engine};
    }

    template <typename M1, typename DistVecType>
    void assert_residual_near(const std::shared_ptr<M1>& mtx,
                              const std::shared_ptr<DistVecType>& x,
                              const std::shared_ptr<DistVecType>& b,
                              double tolerance)
    {
        auto one =
            gko::initialize<local_vec_type>({gko::one<value_type>()}, exec);
        auto neg_one =
            gko::initialize<local_vec_type>({-gko::one<value_type>()}, exec);
        auto norm = DistVecType::local_vector_type::absolute_type::create(
            ref, gko::dim<2>{1, b->get_size()[1]});
        auto dist_res = gko::clone(b);
        auto b_norm = DistVecType::local_vector_type::absolute_type::create(
            ref, gko::dim<2>{1, b->get_size()[1]});
        b->compute_norm2(b_norm.get());
        mtx->apply(neg_one.get(), x.get(), one.get(), dist_res.get());
        dist_res->compute_norm2(norm.get());

        for (int i = 0; i < norm->get_num_stored_elements(); ++i) {
            EXPECT_LE(norm->at(i) / b_norm->at(i), tolerance);
        }
    }


    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
    std::shared_ptr<dist_vec_type> dist_b;
    std::shared_ptr<dist_vec_type> dist_x;
    std::shared_ptr<gko::LinOpFactory> solver_factory;
    std::shared_ptr<gko::LinOpFactory> local_solver_factory;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(SchwarzPreconditioner, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(SchwarzPreconditioner, CanApplyPreconditionedSolver)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using cg = typename TestFixture::solver_type;
    using prec = typename TestFixture::dist_prec_type;

    static constexpr double tolerance = 1e-3;
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(200u).on(this->exec));
    auto tol_stop = gko::share(
        gko::stop::ResidualNorm<value_type>::build()
            .with_reduction_factor(
                static_cast<gko::remove_complex<value_type>>(tolerance))
            .on(this->exec));

    this->solver_factory =
        cg::build()
            .with_preconditioner(
                prec::build()
                    .with_local_solver_factory(this->local_solver_factory)
                    .on(this->exec))
            .with_criteria(iter_stop, tol_stop)
            .on(this->exec);
    auto solver = this->solver_factory->generate(this->dist_mat);
    solver->apply(this->dist_b.get(), this->dist_x.get());

    this->assert_residual_near(this->dist_mat, this->dist_x, this->dist_b,
                               tolerance * 50);
}
