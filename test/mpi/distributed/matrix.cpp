// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>
#include <random>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


template <typename ValueLocalGlobalIndexType>
class MatrixCreation : public CommonMpiTestFixture {
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
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    MatrixCreation()
        : size{5, 5},
          mat_input{size,
                    {{0, 1, 1},
                     {0, 3, 2},
                     {1, 1, 3},
                     {1, 2, 4},
                     {2, 1, 5},
                     {2, 2, 6},
                     {3, 3, 8},
                     {3, 4, 7},
                     {4, 0, 9},
                     {4, 4, 10}}},
          dist_input{
              {{size,
                {{0, 1, 1},
                 {0, 3, 2},
                 {1, 1, 3},
                 {1, 2, 4},
                 {2, 0, 1},
                 {2, 3, 1}}},
               {size, {{0, 0, 1}, {2, 1, 5}, {2, 2, 6}, {3, 3, 8}, {3, 4, 7}}},
               {size, {{2, 2, 1}, {3, 3, -1}, {4, 0, 9}, {4, 4, 10}}}}},
          engine(42)
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 5}));
        col_part = Partition::build_from_mapping(
            exec,
            gko::array<gko::experimental::distributed::comm_index_type>(
                exec,
                I<gko::experimental::distributed::comm_index_type>{1, 1, 2, 0,
                                                                   0}),
            3);

        dist_mat = dist_mtx_type::create(exec, comm);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }


    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;
    std::shared_ptr<Partition> col_part;

    gko::matrix_data<value_type, global_index_type> mat_input;
    std::array<matrix_data, 3> dist_input;

    std::unique_ptr<dist_mtx_type> dist_mat;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(MatrixCreation, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(MatrixCreation, ReadsDistributedGlobalData)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    I<I<value_type>> res_local[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_non_local[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = this->dist_mat->get_communicator().rank();

    this->dist_mat->read_distributed(this->mat_input, this->row_part);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
}


TYPED_TEST(MatrixCreation, ReadsDistributedLocalData)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    I<I<value_type>> res_local[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_non_local[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = this->dist_mat->get_communicator().rank();

    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
}


TYPED_TEST(MatrixCreation, ReadsDistributedLocalDataWithCommunicate)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    I<I<value_type>> res_local[] = {{{1, 1}, {0, 3}}, {{7, 1}, {0, 7}}, {{10}}};
    I<I<value_type>> res_non_local[] = {
        {{0, 2}, {4, 0}}, {{1, 5, 0}, {0, 0, 7}}, {{9}}};
    auto rank = this->dist_mat->get_communicator().rank();

    this->dist_mat->read_distributed(
        this->dist_input[rank], this->row_part,
        gko::experimental::distributed::assembly_mode::communicate);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
}


TYPED_TEST(MatrixCreation, ReadsDistributedWithColPartition)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    I<I<value_type>> res_local[] = {{{2, 0}, {0, 0}}, {{0, 5}, {0, 0}}, {{0}}};
    I<I<value_type>> res_non_local[] = {
        {{1, 0}, {3, 4}}, {{0, 0, 6}, {8, 7, 0}}, {{10, 9}}};
    auto rank = this->dist_mat->get_communicator().rank();

    this->dist_mat->read_distributed(this->mat_input, this->row_part,
                                     this->col_part);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
}


TYPED_TEST(MatrixCreation, ReadsDistributedWithColPartitionAndCommunicate)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    I<I<value_type>> res_local[] = {{{2, 0}, {0, 0}}, {{1, 5}, {0, 0}}, {{0}}};
    I<I<value_type>> res_non_local[] = {
        {{1, 1, 0}, {0, 3, 4}}, {{1, 0, 7}, {7, 7, 0}}, {{10, 9}}};
    auto rank = this->dist_mat->get_communicator().rank();

    this->dist_mat->read_distributed(
        this->dist_input[rank], this->row_part, this->col_part,
        gko::experimental::distributed::assembly_mode::communicate);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
}


TYPED_TEST(MatrixCreation, BuildOnlyLocal)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using global_index_type = typename TestFixture::global_index_type;
    using Partition = typename TestFixture::Partition;
    using local_index_type = typename TestFixture::local_index_type;
    using matrix_data = gko::matrix_data<value_type, local_index_type>;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    I<I<value_type>> res_local[] = {
        {{1, 2}, {0, 3}}, {{0, 1}, {-1, 0}}, {{1, 0}, {0, 1}}};
    auto rank = this->comm.rank();
    gko::dim<2> size(2, 2);
    std::array<matrix_data, 3> dist_input{
        {{size, {{0, 0, 1}, {0, 1, 2}, {1, 1, 3}}},
         {size, {{0, 1, 1}, {1, 0, -1}}},
         {size, {{0, 0, 1}, {1, 1, 1}}}}};
    auto local = gko::share(csr::create(this->exec));
    local->read(dist_input[rank]);
    // create vector
    auto x = dist_vec_type::create(this->ref, this->comm);
    auto y = dist_vec_type::create(this->ref, this->comm);
    auto vec_md = gko::matrix_data<value_type, global_index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}, {6}}};
    auto part = Partition::build_from_mapping(
        this->exec,
        gko::array<gko::experimental::distributed::comm_index_type>(
            this->exec,
            I<gko::experimental::distributed::comm_index_type>{1, 1, 2, 0, 0,
                                                               2}),
        3);
    I<I<value_type>> result[3] = {{{14}, {15}}, {{2}, {-1}}, {{3}, {6}}};
    x->read_distributed(vec_md, part);
    y->read_distributed(vec_md, part);

    auto mat =
        dist_mtx_type::create(this->exec, this->comm, gko::dim<2>{6, 6}, local);
    mat->apply(x, y);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(mat->get_local_matrix()), res_local[rank],
                        0);
    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), result[rank], 0);
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(MatrixCreation, BuildFromExistingDataDeprecated)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using global_index_type = typename TestFixture::global_index_type;
    using Partition = typename TestFixture::Partition;
    using local_index_type = typename TestFixture::local_index_type;
    using matrix_data = gko::matrix_data<value_type, local_index_type>;
    using input_triple =
        gko::detail::input_triple<value_type, local_index_type>;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using comm_index_type = gko::experimental::distributed::comm_index_type;
    auto rank = this->comm.rank();
    I<I<value_type>> res_local[] = {{{2, 0}, {0, 0}}, {{0, 5}, {0, 0}}, {{0}}};
    std::array<gko::dim<2>, 3> size_local{{{2, 2}, {2, 2}, {1, 1}}};
    std::array<matrix_data, 3> dist_input_local{
        {{size_local[0], I<input_triple>{{0, 0, 2}}},
         {size_local[1], I<input_triple>{{0, 1, 5}}},
         {size_local[2]}}};
    I<I<value_type>> res_non_local[] = {
        {{1, 0}, {3, 4}}, {{0, 0, 6}, {8, 7, 0}}, {{10, 9}}};
    std::array<gko::dim<2>, 3> size_non_local{{{2, 2}, {2, 3}, {1, 2}}};
    std::array<matrix_data, 3> dist_input_non_local{
        {{size_non_local[0], I<input_triple>{{0, 0, 1}, {1, 0, 3}, {1, 1, 4}}},
         {size_non_local[1], I<input_triple>{{0, 2, 6}, {1, 0, 8}, {1, 1, 7}}},
         {size_non_local[2], I<input_triple>{{0, 0, 10}, {0, 1, 9}}}}};
    std::array<std::vector<comm_index_type>, 3> recv_sizes{
        {{0, 1, 1}, {2, 0, 1}, {1, 1, 0}}};
    std::array<std::vector<comm_index_type>, 3> recv_offsets{
        {{0, 0, 1, 2}, {0, 2, 2, 3}, {0, 1, 2, 2}}};
    std::array<gko::array<local_index_type>, 3> recv_gather_index{
        {{this->exec, {1, 0}}, {this->exec, {0, 1, 0}}, {this->exec, {1, 0}}}};
    auto local = gko::share(csr::create(this->exec));
    local->read(dist_input_local[rank]);
    auto non_local = gko::share(csr::create(this->exec));
    non_local->read(dist_input_non_local[rank]);
    // create vector
    auto vec_md = gko::matrix_data<value_type, global_index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{10}, {18}}, {{28}, {67}}, {{59}}};
    auto row_part = Partition::build_from_contiguous(
        this->exec, gko::array<global_index_type>(
                        this->exec, I<global_index_type>{0, 2, 4, 5}));
    auto col_part = Partition::build_from_mapping(
        this->exec,
        gko::array<comm_index_type>(this->exec,
                                    I<comm_index_type>{1, 1, 2, 0, 0}),
        3);
    auto x = dist_vec_type::create(this->ref, this->comm);
    auto y = dist_vec_type::create(this->ref, this->comm);
    x->read_distributed(vec_md, col_part);
    y->read_distributed(vec_md, row_part);

    auto mat = dist_mtx_type::create(
        this->exec, this->comm, gko::dim<2>{5, 5}, local, non_local,
        recv_sizes[rank], recv_offsets[rank], recv_gather_index[rank]);
    mat->apply(x, y);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(mat->get_local_matrix()), res_local[rank],
                        0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), result[rank], 0);
}


GKO_END_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(MatrixCreation, BuildFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using global_index_type = typename TestFixture::global_index_type;
    using local_index_type = typename TestFixture::local_index_type;
    using Partition = typename TestFixture::Partition;
    using index_map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, local_index_type>;
    using input_triple =
        gko::detail::input_triple<value_type, local_index_type>;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using comm_index_type = gko::experimental::distributed::comm_index_type;
    auto rank = this->comm.rank();
    auto row_part = gko::share(Partition::build_from_contiguous(
        this->exec, gko::array<global_index_type>(
                        this->exec, I<global_index_type>{0, 2, 4, 5})));
    auto col_part = gko::share(Partition::build_from_mapping(
        this->exec,
        gko::array<comm_index_type>(this->exec,
                                    I<comm_index_type>{1, 1, 2, 0, 0}),
        3));
    std::array<gko::array<global_index_type>, 3> recv_connections = {
        gko::array<global_index_type>{this->exec, {1, 2}},
        gko::array<global_index_type>{this->exec, {3, 4, 2}},
        gko::array<global_index_type>{this->exec, {4, 0}}};
    index_map_type imap(this->exec, col_part, rank, recv_connections[rank]);
    std::array<gko::dim<2>, 3> size_local{{{2, 2}, {2, 2}, {1, 1}}};
    std::array<matrix_data, 3> dist_input_local{
        {{size_local[0], I<input_triple>{{0, 0, 2}}},
         {size_local[1], I<input_triple>{{0, 1, 5}}},
         {size_local[2]}}};
    std::array<gko::dim<2>, 3> size_non_local{{{2, 2}, {2, 3}, {1, 2}}};
    std::array<matrix_data, 3> dist_input_non_local{
        {{size_non_local[0], I<input_triple>{{0, 0, 1}, {1, 0, 3}, {1, 1, 4}}},
         {size_non_local[1], I<input_triple>{{0, 2, 6}, {1, 0, 8}, {1, 1, 7}}},
         {size_non_local[2], I<input_triple>{{0, 0, 10}, {0, 1, 9}}}}};
    I<I<value_type>> res_local[] = {{{2, 0}, {0, 0}}, {{0, 5}, {0, 0}}, {{0}}};
    I<I<value_type>> res_non_local[] = {
        {{1, 0}, {3, 4}}, {{0, 0, 6}, {8, 7, 0}}, {{10, 9}}};
    auto local = gko::share(csr::create(this->exec));
    local->read(dist_input_local[rank]);
    auto non_local = gko::share(csr::create(this->exec));
    non_local->read(dist_input_non_local[rank]);
    // create vector
    auto vec_md = gko::matrix_data<value_type, global_index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{10}, {18}}, {{28}, {67}}, {{59}}};
    auto x = dist_vec_type::create(this->ref, this->comm);
    auto y = dist_vec_type::create(this->ref, this->comm);
    x->read_distributed(vec_md, col_part);
    y->read_distributed(vec_md, row_part);

    auto mat =
        dist_mtx_type::create(this->exec, this->comm, imap, local, non_local);
    mat->apply(x, y);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(mat->get_local_matrix()), res_local[rank],
                        0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(mat->get_non_local_matrix()),
                        res_non_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), result[rank], 0);
}


#endif


template <typename ValueType>
class Matrix : public CommonMpiTestFixture {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using csr_mtx_type = gko::matrix::Csr<value_type, global_index_type>;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using dense_vec_type = gko::matrix::Dense<value_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;

    Matrix() : size{5, 5}, engine()
    {
        row_part = part_type::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 5}));
        col_part = part_type::build_from_mapping(
            exec,
            gko::array<gko::experimental::distributed::comm_index_type>(
                exec,
                I<gko::experimental::distributed::comm_index_type>{1, 1, 2, 0,
                                                                   0}),
            3);

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat_large = dist_mtx_type::create(exec, comm);
        x = dist_vec_type::create(ref, comm);
        y = dist_vec_type::create(ref, comm);

        csr_mat = csr_mtx_type::create(exec);
        dense_x = dense_vec_type::create(exec);
        dense_y = dense_vec_type::create(exec);

        mat_input = matrix_data{
            size,
            // clang-format off
            {{0, 1, 1}, {0, 3, 2}, {1, 1, 3}, {1, 2, 4}, {2, 1, 5},
             {2, 2, 6}, {3, 3, 8}, {3, 4, 7}, {4, 0, 9}, {4, 4, 10}}
            // clang-format on
        };
        dist_mat->read_distributed(mat_input, this->row_part, this->col_part);
        csr_mat->read(mat_input);

        alpha = gko::test::generate_random_matrix<dense_vec_type>(
            1, 1, std::uniform_int_distribution<gko::size_type>(1, 1),
            std::normal_distribution<>(), this->engine, this->exec);
        beta = gko::test::generate_random_matrix<dense_vec_type>(
            1, 1, std::uniform_int_distribution<gko::size_type>(1, 1),
            std::normal_distribution<>(), this->engine, this->exec);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    void assert_local_vector_equal_to_global_vector(const dist_vec_type* dist,
                                                    const dense_vec_type* dense,
                                                    const part_type* part,
                                                    int rank)
    {
        auto host_part = gko::clone(this->ref, part);
        auto range_bounds = host_part->get_range_bounds();
        auto part_ids = host_part->get_part_ids();
        std::vector<global_index_type> gather_idxs;
        for (gko::size_type range_id = 0;
             range_id < host_part->get_num_ranges(); ++range_id) {
            if (part_ids[range_id] == rank) {
                for (global_index_type global_row = range_bounds[range_id];
                     global_row < range_bounds[range_id + 1]; ++global_row) {
                    gather_idxs.push_back(global_row);
                }
            }
        }
        gko::array<global_index_type> gather_idxs_view(
            this->exec, gather_idxs.begin(), gather_idxs.end());
        auto gathered_local = dense->row_gather(&gather_idxs_view);

        GKO_ASSERT_MTX_NEAR(dist->get_local_vector(), gathered_local,
                            r<value_type>::value);
    }

    void init_large(gko::size_type num_rows, gko::size_type num_cols)
    {
        auto rank = comm.rank();
        int num_parts = comm.size();
        auto vec_md = gko::test::generate_random_matrix_data<value_type,
                                                             global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(static_cast<int>(num_cols),
                                               static_cast<int>(num_cols)),
            std::normal_distribution<>(), engine);
        auto mat_md = gko::test::generate_random_matrix_data<value_type,
                                                             global_index_type>(
            num_rows, num_rows,
            std::uniform_int_distribution<int>(0, static_cast<int>(num_rows)),
            std::normal_distribution<>(), engine);

        auto row_mapping = gko::test::generate_random_array<
            gko::experimental::distributed::comm_index_type>(
            num_rows, std::uniform_int_distribution<int>(0, num_parts - 1),
            engine, exec);
        auto col_mapping = gko::test::generate_random_array<
            gko::experimental::distributed::comm_index_type>(
            num_rows, std::uniform_int_distribution<int>(0, num_parts - 1),
            engine, exec);
        row_part_large =
            part_type::build_from_mapping(exec, row_mapping, num_parts);
        col_part_large =
            part_type::build_from_mapping(exec, col_mapping, num_parts);

        dist_mat_large->read_distributed(mat_md, row_part_large,
                                         col_part_large);
        csr_mat->read(mat_md);

        x->read_distributed(vec_md, col_part_large);
        dense_x->read(vec_md);

        y->read_distributed(vec_md, row_part_large);
        dense_y->read(vec_md);
    }

    gko::dim<2> size;

    std::shared_ptr<part_type> row_part;
    std::shared_ptr<part_type> col_part;
    std::shared_ptr<part_type> row_part_large;
    std::shared_ptr<part_type> col_part_large;

    std::unique_ptr<dist_mtx_type> dist_mat;
    std::unique_ptr<dist_mtx_type> dist_mat_large;
    std::unique_ptr<csr_mtx_type> csr_mat;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dense_vec_type> dense_x;
    std::unique_ptr<dense_vec_type> dense_y;

    std::unique_ptr<dense_vec_type> alpha;
    std::unique_ptr<dense_vec_type> beta;

    std::default_random_engine engine;

    matrix_data mat_input;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Matrix, CanApplyToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{10}, {18}}, {{28}, {67}}, {{59}}};
    auto rank = this->comm.rank();
    this->x->read_distributed(vec_md, this->col_part);
    this->y->read_distributed(vec_md, this->row_part);

    this->dist_mat->apply(this->x, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(Matrix, CanApplyToMultipleVectors)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1, 11}, {2, 22}, {3, 33}, {4, 44}, {5, 55}}};
    I<I<value_type>> result[3] = {
        {{10, 110}, {18, 198}}, {{28, 308}, {67, 737}}, {{59, 649}}};
    auto rank = this->comm.rank();
    this->x->read_distributed(vec_md, this->col_part);
    this->y->read_distributed(vec_md, this->row_part);

    this->dist_mat->apply(this->x, this->y);

    auto eps = std::is_same_v<value_type, gko::bfloat16> ||
                       std::is_same_v<value_type, std::complex<gko::bfloat16>>
                   ? r<value_type>::value
                   : gko::remove_complex<value_type>{0.0};
    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], eps);
}


TYPED_TEST(Matrix, CanAdvancedApplyToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using dense_vec_type = typename TestFixture::dense_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{17}, {30}}, {{47}, {122}}, {{103}}};
    auto rank = this->comm.rank();
    this->alpha = gko::initialize<dense_vec_type>({2.0}, this->exec);
    this->beta = gko::initialize<dense_vec_type>({-3.0}, this->exec);
    this->x->read_distributed(vec_md, this->col_part);
    this->y->read_distributed(vec_md, this->row_part);

    this->dist_mat->apply(this->alpha, this->x, this->beta, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(Matrix, CanApplyToSingleVectorByNonLocalApply2)
{
    using value_type = typename TestFixture::value_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    // Coo does not support 16-bit floating point precision in some situations.
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    auto vec_md = gko::matrix_data<value_type, global_index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{10}, {18}}, {{28}, {67}}, {{59}}};
    auto rank = this->comm.rank();
    this->x->read_distributed(vec_md, this->col_part);
    this->y->read_distributed(vec_md, this->row_part);
    // default setup with csr should use normal advanced apply, so use coo as
    // non_local to check apply2
    auto dist_mat_coo = dist_mtx_type::create(
        this->exec, this->comm, gko::with_matrix_type<gko::matrix::Csr>(),
        gko::with_matrix_type<gko::matrix::Coo>());
    dist_mat_coo->read_distributed(this->mat_input, this->row_part,
                                   this->col_part);

    dist_mat_coo->apply(this->x, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(Matrix, CanAdvancedApplyToSingleVectorByNonLocalApply2)
{
    using value_type = typename TestFixture::value_type;
    using global_index_type = typename TestFixture::global_index_type;
    using dense_vec_type = typename TestFixture::dense_vec_type;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    // Coo does not support 16-bit floating point precision in some situations.
    SKIP_IF_HALF(value_type);
    SKIP_IF_BFLOAT16(value_type);
    auto vec_md = gko::matrix_data<value_type, global_index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> result[3] = {{{17}, {30}}, {{47}, {122}}, {{103}}};
    auto rank = this->comm.rank();
    this->alpha = gko::initialize<dense_vec_type>({2.0}, this->exec);
    this->beta = gko::initialize<dense_vec_type>({-3.0}, this->exec);
    this->x->read_distributed(vec_md, this->col_part);
    this->y->read_distributed(vec_md, this->row_part);
    // default setup with csr should use normal advanced apply, so use coo as
    // non_local to check apply2
    auto dist_mat_coo = dist_mtx_type::create(
        this->exec, this->comm, gko::with_matrix_type<gko::matrix::Csr>(),
        gko::with_matrix_type<gko::matrix::Coo>());
    dist_mat_coo->read_distributed(this->mat_input, this->row_part,
                                   this->col_part);

    dist_mat_coo->apply(this->alpha, this->x, this->beta, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(Matrix, CanApplyToSingleVectorLarge)
{
    this->init_large(100, 1);

    this->dist_mat_large->apply(this->x, this->y);
    this->csr_mat->apply(this->dense_x, this->dense_y);

    this->assert_local_vector_equal_to_global_vector(
        this->y.get(), this->dense_y.get(), this->row_part_large.get(),
        this->comm.rank());
}


TYPED_TEST(Matrix, CanApplyToMultipleVectorsLarge)
{
    this->init_large(100, 17);

    this->dist_mat_large->apply(this->x, this->y);
    this->csr_mat->apply(this->dense_x, this->dense_y);

    this->assert_local_vector_equal_to_global_vector(
        this->y.get(), this->dense_y.get(), this->row_part_large.get(),
        this->comm.rank());
}


TYPED_TEST(Matrix, CanAdvancedApplyToMultipleVectorsLarge)
{
    this->init_large(100, 17);

    this->dist_mat_large->apply(this->alpha, this->x, this->beta, this->y);
    this->csr_mat->apply(this->alpha, this->dense_x, this->beta, this->dense_y);

    this->assert_local_vector_equal_to_global_vector(
        this->y.get(), this->dense_y.get(), this->row_part_large.get(),
        this->comm.rank());
}


TYPED_TEST(Matrix, CanColScale)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using csr = typename TestFixture::local_matrix_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> res_col_scale_local[] = {
        {{8, 0}, {0, 0}}, {{0, 10}, {0, 0}}, {{0}}};
    I<I<value_type>> res_col_scale_non_local[] = {
        {{2, 0}, {6, 12}}, {{0, 0, 18}, {32, 35, 0}}, {{50, 9}}};
    auto rank = this->comm.rank();
    auto col_scaling_factors = dist_vec_type::create(this->exec, this->comm);
    col_scaling_factors->read_distributed(vec_md, this->col_part);

    this->dist_mat->col_scale(col_scaling_factors);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_col_scale_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_col_scale_non_local[rank], 0);
}


TYPED_TEST(Matrix, CanRowScale)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using csr = typename TestFixture::local_matrix_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> res_row_scale_local[] = {
        {{2, 0}, {0, 0}}, {{0, 15}, {0, 0}}, {{0}}};
    I<I<value_type>> res_row_scale_non_local[] = {
        {{1, 0}, {6, 8}}, {{0, 0, 18}, {32, 28, 0}}, {{50, 45}}};
    auto rank = this->comm.rank();
    auto row_scaling_factors = dist_vec_type::create(this->exec, this->comm);
    row_scaling_factors->read_distributed(vec_md, this->row_part);

    this->dist_mat->row_scale(row_scaling_factors);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_row_scale_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_row_scale_non_local[rank], 0);
}


TYPED_TEST(Matrix, CanColScaleWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using csr = typename TestFixture::local_matrix_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> res_col_scale_local[] = {
        {{8, 0}, {0, 0}}, {{0, 10}, {0, 0}}, {{0}}};
    I<I<value_type>> res_col_scale_non_local[] = {
        {{2, 0}, {6, 12}}, {{0, 0, 18}, {32, 35, 0}}, {{50, 9}}};
    gko::dim<2> local_sizes[] = {{2, 1}, {2, 1}, {1, 1}};
    auto rank = this->comm.rank();
    auto col_scaling_factors = dist_vec_type::create(
        this->exec, this->comm, gko::dim<2>{5, 1}, local_sizes[rank], 2);
    col_scaling_factors->read_distributed(vec_md, this->col_part);

    this->dist_mat->col_scale(col_scaling_factors);

    ASSERT_EQ(col_scaling_factors->get_stride(), 2);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_col_scale_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_col_scale_non_local[rank], 0);
}


TYPED_TEST(Matrix, CanRowScaleWithStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using csr = typename TestFixture::local_matrix_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    I<I<value_type>> res_row_scale_local[] = {
        {{2, 0}, {0, 0}}, {{0, 15}, {0, 0}}, {{0}}};
    I<I<value_type>> res_row_scale_non_local[] = {
        {{1, 0}, {6, 8}}, {{0, 0, 18}, {32, 28, 0}}, {{50, 45}}};
    gko::dim<2> local_sizes[] = {{2, 1}, {2, 1}, {1, 1}};
    auto rank = this->comm.rank();
    auto row_scaling_factors = dist_vec_type::create(
        this->exec, this->comm, gko::dim<2>{5, 1}, local_sizes[rank], 2);
    row_scaling_factors->read_distributed(vec_md, this->row_part);

    this->dist_mat->row_scale(row_scaling_factors);

    ASSERT_EQ(row_scaling_factors->get_stride(), 2);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_row_scale_local[rank], 0);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        res_row_scale_non_local[rank], 0);
}


TYPED_TEST(Matrix, ColScaleThrowsOnWrongDimension)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using part_type = typename TestFixture::part_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}}};
    auto two_vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}};
    auto rank = this->comm.rank();
    auto col_part = part_type::build_from_mapping(
        this->exec,
        gko::array<gko::experimental::distributed::comm_index_type>(
            this->exec,
            I<gko::experimental::distributed::comm_index_type>{1, 2, 0, 0}),
        3);
    auto col_scaling_factors = dist_vec_type::create(this->exec, this->comm);
    col_scaling_factors->read_distributed(vec_md, col_part);
    auto two_col_scaling_factors =
        dist_vec_type::create(this->exec, this->comm);
    two_col_scaling_factors->read_distributed(two_vec_md, this->col_part);

    ASSERT_THROW(this->dist_mat->col_scale(col_scaling_factors),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->dist_mat->col_scale(two_col_scaling_factors),
                 gko::ValueMismatch);
}


TYPED_TEST(Matrix, RowScaleThrowsOnWrongDimension)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using part_type = typename TestFixture::part_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}}};
    auto two_vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}};
    auto rank = this->comm.rank();
    auto row_part = part_type::build_from_contiguous(
        this->exec,
        gko::array<index_type>(this->exec, I<index_type>{0, 2, 3, 4}));
    auto row_scaling_factors = dist_vec_type::create(this->exec, this->comm);
    row_scaling_factors->read_distributed(vec_md, row_part);
    auto two_row_scaling_factors =
        dist_vec_type::create(this->exec, this->comm);
    two_row_scaling_factors->read_distributed(two_vec_md, this->col_part);

    ASSERT_THROW(this->dist_mat->row_scale(row_scaling_factors),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->dist_mat->row_scale(two_row_scaling_factors),
                 gko::ValueMismatch);
}


TYPED_TEST(Matrix, CanConvertToNextPrecision)
{
    using T = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDist = typename gko::experimental::distributed::Matrix<
        OtherT, local_index_type, global_index_type>;
    auto tmp = OtherDist::create(this->ref, this->comm);
    auto res = TestFixture::dist_mtx_type::create(this->ref, this->comm);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : static_cast<gko::remove_complex<T>>(r<OtherT>::value);

    this->dist_mat->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        gko::as<csr>(res->get_local_matrix()), residual);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_non_local_matrix()),
                        gko::as<csr>(res->get_non_local_matrix()), residual);
}


TYPED_TEST(Matrix, CanMoveToNextPrecision)
{
    using T = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDist = typename gko::experimental::distributed::Matrix<
        OtherT, local_index_type, global_index_type>;
    auto tmp = OtherDist::create(this->ref, this->comm);
    auto res = TestFixture::dist_mtx_type::create(this->ref, this->comm);
    auto clone_dist_mat = gko::clone(this->dist_mat);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : static_cast<gko::remove_complex<T>>(r<OtherT>::value);

    this->dist_mat->move_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(clone_dist_mat->get_local_matrix()),
                        gko::as<csr>(res->get_local_matrix()), residual);
    GKO_ASSERT_MTX_NEAR(gko::as<csr>(clone_dist_mat->get_non_local_matrix()),
                        gko::as<csr>(res->get_non_local_matrix()), residual);
}


bool needs_transfers(std::shared_ptr<const gko::Executor> exec)
{
    return exec->get_master() != exec &&
           !gko::experimental::mpi::is_gpu_aware();
}


class HostToDeviceLogger : public gko::log::Logger {
public:
    mutable int transfer_count = 0;

    void on_copy_started(const gko::Executor* exec_from,
                         const gko::Executor* exec_to,
                         const gko::uintptr& loc_from,
                         const gko::uintptr& loc_to,
                         const gko::size_type& num_bytes) const override
    {
        if (exec_from != exec_to) {
            transfer_count++;
        }
    }
};


class AllocationLogger : public gko::log::Logger {
public:
    mutable int count = 0;

protected:
    void on_allocation_completed(const gko::Executor* exec,
                                 const gko::size_type& num_bytes,
                                 const gko::uintptr& location) const override
    {
        ++count;
    }
};


class MatrixInternalBuffers : public CommonMpiTestFixture {
public:
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using dense_vec_type = gko::matrix::Dense<value_type>;

    MatrixInternalBuffers()
    {
        exec->add_logger(copy_logger);
        exec->add_logger(alloc_logger);

        mat = dist_mtx_type::create(exec, comm);
        x = dist_vec_type::create(exec, comm);
        y = dist_vec_type::create(exec, comm);
        factor = dist_vec_type::create(exec, comm, gko::dim<2>{0, 1},
                                       gko::dim<2>{0, 1});

        alpha = dense_vec_type::create(exec, gko::dim<2>{1, 1});
        beta = dense_vec_type::create(exec, gko::dim<2>{1, 1});
    }


    std::unique_ptr<dist_mtx_type> mat;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;
    std::unique_ptr<dist_vec_type> factor;

    std::unique_ptr<dense_vec_type> alpha;
    std::unique_ptr<dense_vec_type> beta;

    std::shared_ptr<HostToDeviceLogger> copy_logger =
        std::make_shared<HostToDeviceLogger>();
    std::shared_ptr<AllocationLogger> alloc_logger =
        std::make_shared<AllocationLogger>();

    std::default_random_engine engine{42};
};


TEST_F(MatrixInternalBuffers, ApplyCopiesToHostOnlyIfNecessary)
{
    auto transfer_count_before = copy_logger->transfer_count;

    mat->apply(x, y);

    ASSERT_EQ(copy_logger->transfer_count > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(MatrixInternalBuffers, AdvancedApplyCopiesToHostOnlyIfNecessary)
{
    auto transfer_count_before = copy_logger->transfer_count;

    mat->apply(alpha, x, beta, y);

    ASSERT_EQ(copy_logger->transfer_count > transfer_count_before,
              needs_transfers(exec));
}


TEST_F(MatrixInternalBuffers, ApplyAllocatesBuffersOnlyOnce)
{
    mat->apply(x, y);

    auto alloc_count_before = alloc_logger->count;
    mat->apply(x, y);

    ASSERT_EQ(alloc_logger->count, alloc_count_before);
}


TEST_F(MatrixInternalBuffers, AdvancedApplyAllocatesBuffersOnlyOnce)
{
    mat->apply(alpha, x, beta, y);

    auto alloc_count_before = alloc_logger->count;
    mat->apply(alpha, x, beta, y);

    ASSERT_EQ(alloc_logger->count, alloc_count_before);
}


TEST_F(MatrixInternalBuffers, ColScaleAllocatesBuffersOnlyOnce)
{
    mat->col_scale(factor);

    auto alloc_count_before = alloc_logger->count;
    mat->col_scale(factor);

    ASSERT_EQ(alloc_logger->count, alloc_count_before);
}
