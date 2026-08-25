// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/rs.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


template <typename ValueLocalGlobalIndexType>
class Rs : public CommonMpiTestFixture {
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
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using local_vec_type = gko::matrix::Dense<value_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using rs = gko::multigrid::Rs<value_type, local_index_type>;

    Rs() : size{num_rows, num_rows}
    {
        // A 1D Laplacian is an M-matrix, and with four rows per rank every
        // rank keeps interior rows: forcing the interface rows into the coarse
        // set still leaves something to coarsen.
        mat_input = gko::matrix_data<value_type, global_index_type>{size};
        const auto n = static_cast<global_index_type>(size[0]);
        for (global_index_type i = 0; i < n; ++i) {
            if (i > 0) {
                mat_input.nonzeros.push_back(
                    {i, i - 1, -gko::one<value_type>()});
            }
            mat_input.nonzeros.push_back({i, i, value_type{2}});
            if (i < n - 1) {
                mat_input.nonzeros.push_back(
                    {i, i + 1, -gko::one<value_type>()});
            }
        }

        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 4, 8, 12}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::size_type local_rows_of(std::shared_ptr<const gko::LinOp> distributed)
    {
        return gko::as<local_matrix_type>(
                   gko::as<dist_mtx_type>(distributed)->get_diag_matrix())
            ->get_size()[0];
    }

    static constexpr gko::size_type num_rows = 12;

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;
    gko::matrix_data<value_type, global_index_type> mat_input;
    std::shared_ptr<dist_mtx_type> dist_mat;
};

TYPED_TEST_SUITE(Rs, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Rs, CoarseOperatorIsTheGalerkinProduct)
{
    using rs = typename TestFixture::rs;
    using value_type = typename TestFixture::value_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using local_vec_type = typename TestFixture::local_vec_type;
    auto level = rs::build().on(this->exec)->generate(this->dist_mat);
    auto coarse = level->get_coarse_op();
    auto prolong = level->get_prolong_op();
    auto restrict_op = level->get_restrict_op();
    const auto fine_global = this->dist_mat->get_size()[0];
    const auto fine_local = this->local_rows_of(this->dist_mat);
    const auto coarse_local = this->local_rows_of(coarse);

    // A deterministic, rank-dependent coarse vector
    auto host_x =
        local_vec_type::create(this->ref, gko::dim<2>{coarse_local, 1});
    for (gko::size_type i = 0; i < coarse_local; ++i) {
        host_x->at(i, 0) = static_cast<value_type>(
            1 + (i + 3 * static_cast<gko::size_type>(this->comm.rank())) % 5);
    }
    auto x = dist_vec_type::create(this->exec, this->comm,
                                   gko::clone(this->exec, host_x));
    const auto coarse_global = x->get_size()[0];
    auto direct = dist_vec_type::create(this->exec, this->comm,
                                        gko::dim<2>{coarse_global, 1},
                                        gko::dim<2>{coarse_local, 1});
    auto px = dist_vec_type::create(this->exec, this->comm,
                                    gko::dim<2>{fine_global, 1},
                                    gko::dim<2>{fine_local, 1});
    auto apx = dist_vec_type::create(this->exec, this->comm,
                                     gko::dim<2>{fine_global, 1},
                                     gko::dim<2>{fine_local, 1});
    auto galerkin = dist_vec_type::create(this->exec, this->comm,
                                          gko::dim<2>{coarse_global, 1},
                                          gko::dim<2>{coarse_local, 1});

    // Ac * x, against R * (A * (P * x)). The right hand side pulls the
    // neighbors' prolongated values through A's off-diagonal block, so it is
    // the true Galerkin product. It agrees with the assembled coarse operator
    // only if replacing the neighbors' prolongation rows by unit vectors was
    // legitimate, i.e. only if every halo row really is a C-point.
    coarse->apply(x, direct);
    prolong->apply(x, px);
    this->dist_mat->apply(px, apx);
    restrict_op->apply(apx, galerkin);

    // the two sides sum the same terms in a different order
    GKO_ASSERT_MTX_NEAR(direct->get_local_vector(),
                        galerkin->get_local_vector(),
                        10 * r<value_type>::value);
}


TYPED_TEST(Rs, InterfaceRowsAreForcedToCPoints)
{
    using rs = typename TestFixture::rs;
    using value_type = typename TestFixture::value_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto level = rs::build().on(this->exec)->generate(this->dist_mat);

    auto prolong = gko::as<dist_mtx_type>(level->get_prolong_op());
    auto p_local = gko::clone(
        this->ref, gko::as<local_matrix_type>(prolong->get_diag_matrix()));
    auto off_diag = gko::clone(
        this->ref,
        gko::as<local_matrix_type>(this->dist_mat->get_off_diag_matrix()));
    const auto* od_row_ptrs = off_diag->get_const_row_ptrs();
    const auto* p_row_ptrs = p_local->get_const_row_ptrs();
    const auto* p_vals = p_local->get_const_values();

    // The prolongation must have no off-diagonal block at all, otherwise it
    // would not be representable as a purely local operator.
    ASSERT_EQ(gko::as<local_matrix_type>(prolong->get_off_diag_matrix())
                  ->get_num_stored_elements(),
              0);
    for (gko::size_type i = 0; i < p_local->get_size()[0]; ++i) {
        if (od_row_ptrs[i + 1] > od_row_ptrs[i]) {
            // This row couples to a remote row, and the test matrix has a
            // symmetric pattern, so the remote rank couples back to it: it is
            // one of our send indices and must have become a C-point, i.e. its
            // prolongation row is a unit vector.
            ASSERT_EQ(p_row_ptrs[i + 1] - p_row_ptrs[i], 1);
            ASSERT_EQ(p_vals[p_row_ptrs[i]], gko::one<value_type>());
        }
    }
}


TYPED_TEST(Rs, GeneratesConsistentlySizedOperators)
{
    using rs = typename TestFixture::rs;
    auto level = rs::build().on(this->exec)->generate(this->dist_mat);
    auto coarse = level->get_coarse_op();
    auto prolong = level->get_prolong_op();
    auto restrict_op = level->get_restrict_op();
    const auto fine_global = this->dist_mat->get_size()[0];

    const auto coarse_global = coarse->get_size()[0];
    ASSERT_EQ(coarse->get_size()[1], coarse_global);
    ASSERT_EQ(prolong->get_size()[0], fine_global);
    ASSERT_EQ(prolong->get_size()[1], coarse_global);
    ASSERT_EQ(restrict_op->get_size()[0], coarse_global);
    ASSERT_EQ(restrict_op->get_size()[1], fine_global);
    // the interior of every rank is coarsened, so the coarse grid is smaller
    ASSERT_LT(coarse_global, fine_global);
    ASSERT_GT(coarse_global, 0);
}
