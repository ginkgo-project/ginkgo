// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include <ginkgo/core/distributed/preconditioner/bddc.hpp>
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
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class BddcPreconditioner : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_vec_type = gko::matrix::Dense<value_type>;
    using dist_prec_type = gko::experimental::distributed::preconditioner::Bddc<
        value_type, local_index_type, global_index_type>;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using non_dist_matrix_type =
        gko::matrix::Csr<value_type, global_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    BddcPreconditioner()
        : CommonMpiTestFixture(),
          size{8, 8},
          mat_input{size,
                    {{0, 0, 2},  {0, 1, -1}, {1, 0, -1}, {1, 1, 2},  {1, 2, -1},
                     {2, 1, -1}, {2, 2, 2},  {2, 3, -1}, {3, 2, -1}, {3, 3, 2},
                     {3, 4, -1}, {4, 3, -1}, {4, 4, 2},  {4, 5, -1}, {5, 4, -1},
                     {5, 5, 2},  {5, 6, -1}, {6, 5, -1}, {6, 6, 2},  {6, 7, -1},
                     {7, 6, -1}, {7, 7, 2}}}
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
        non_dist_mat = non_dist_matrix_type::create(exec);
        non_dist_mat->read(mat_input);

        auto nrhs = 1;
        auto global_size =
            gko::dim<2>{size[0], static_cast<gko::size_type>(nrhs)};
        auto local_size = gko::dim<2>{
            static_cast<gko::size_type>(row_part->get_part_size(comm.rank())),
            static_cast<gko::size_type>(nrhs)};
        auto dist_result =
            dist_vec_type::create(ref, comm, global_size, local_size, nrhs);
        dist_b = gko::share(gko::clone(exec, dist_result));
        dist_b->fill(-gko::one<value_type>());
        dist_x = gko::share(gko::clone(exec, dist_result));
        dist_x->fill(gko::zero<value_type>());
        auto non_dist_result = local_vec_type::create(ref, global_size, nrhs);
        non_dist_result->fill(-gko::one<value_type>());
        non_dist_b = gko::share(gko::clone(exec, non_dist_result));
        non_dist_x = gko::share(gko::clone(exec, non_dist_result));
        non_dist_x->fill(gko::zero<value_type>());

        local_solver_factory =
            local_prec_type::build().with_max_block_size(1u).on(exec);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
    std::shared_ptr<dist_vec_type> dist_b;
    std::shared_ptr<dist_vec_type> dist_x;
    std::shared_ptr<non_dist_matrix_type> non_dist_mat;
    std::shared_ptr<local_vec_type> non_dist_b;
    std::shared_ptr<local_vec_type> non_dist_x;
    std::shared_ptr<gko::LinOpFactory> non_dist_solver_factory;
    std::shared_ptr<gko::LinOpFactory> dist_solver_factory;
    std::shared_ptr<gko::LinOpFactory> local_solver_factory;

    void assert_equal_to_non_distributed_vector(
        std::shared_ptr<dist_vec_type> dist_vec,
        std::shared_ptr<local_vec_type> local_vec)
    {
        auto host_row_part = row_part->clone(ref);
        auto l_dist_vec = dist_vec->get_local_vector();
        auto vec_view = local_vec_type::create_const(
            exec, l_dist_vec->get_size(),
            gko::array<value_type>::const_view(
                exec, l_dist_vec->get_size()[0],
                local_vec->get_const_values() +
                    host_row_part->get_range_bounds()[comm.rank()]),
            l_dist_vec->get_size()[1]);
        GKO_ASSERT_MTX_NEAR(l_dist_vec, vec_view.get(), r<value_type>::value);
    }
};

TYPED_TEST_SUITE(BddcPreconditioner, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);

TYPED_TEST(BddcPreconditioner, GenerateFailsIfNotDdMatrix)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using local_prec_type =
        gko::preconditioner::Jacobi<value_type, local_index_type>;
    using prec = typename TestFixture::dist_prec_type;

    auto local_solver = gko::share(local_prec_type::build()
                                       .with_max_block_size(1u)
                                       .on(this->exec)
                                       ->generate(this->non_dist_mat));
    auto bddc = prec::build()
                    .with_local_solver(this->local_solver_factory)
                    .on(this->exec);

    ASSERT_THROW(bddc->generate(this->dist_mat), gko::NotSupported);
}