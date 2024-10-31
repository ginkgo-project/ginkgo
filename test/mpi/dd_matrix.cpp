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
#include <ginkgo/core/distributed/dd_matrix.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"
#include "ginkgo/core/base/exception.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


template <typename ValueLocalGlobalIndexType>
class DdMatrix : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dd_mtx_type =
        gko::experimental::distributed::DdMatrix<value_type, local_index_type,
                                                 global_index_type>;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;
    using local_matrix_data = gko::matrix_data<value_type, local_index_type>;
    using dense_vec_type = gko::matrix::Dense<value_type>;


    DdMatrix()
        : size{12, 12},
          dist_input{
              {{size, {{0, 0, 2},    {0, 1, -1}, {0, 3, -1},   {1, 0, -1},
                       {1, 1, 3},    {1, 2, -1}, {1, 4, -1},   {2, 1, -1},
                       {2, 2, 2},    {2, 5, -1}, {3, 0, -1},   {3, 3, 1.5},
                       {3, 4, -0.5}, {4, 1, -1}, {4, 3, -0.5}, {4, 4, 2},
                       {4, 5, -0.5}, {5, 2, -1}, {5, 4, -0.5}, {5, 5, 1.5}}},
               {size, {{3, 3, 1.5},  {3, 4, -0.5}, {3, 6, -1},   {4, 3, -0.5},
                       {4, 4, 2},    {4, 5, -0.5}, {4, 7, -1},   {5, 4, -0.5},
                       {5, 5, 1.5},  {5, 8, -1},   {6, 3, -1},   {6, 6, 1.5},
                       {6, 7, -0.5}, {7, 4, -1},   {7, 6, -0.5}, {7, 7, 2},
                       {7, 8, -0.5}, {8, 5, -1},   {8, 7, -0.5}, {8, 8, 1.5}}},
               {size,
                {{6, 6, 1.5},  {6, 7, -0.5}, {6, 9, -1},   {7, 6, -0.5},
                 {7, 7, 2},    {7, 8, -0.5}, {7, 10, -1},  {8, 7, -0.5},
                 {8, 8, 1.5},  {8, 11, -1},  {9, 6, -1},   {9, 9, 2},
                 {9, 10, -1},  {10, 7, -1},  {10, 9, -1},  {10, 10, 3},
                 {10, 11, -1}, {11, 8, -1},  {11, 10, -1}, {11, 11, 2}}}}},
          local_size{6, 6},
          local_result{
              {{local_size,
                {{0, 0, 2},    {0, 1, -1}, {0, 3, -1},   {1, 0, -1},
                 {1, 1, 3},    {1, 2, -1}, {1, 4, -1},   {2, 1, -1},
                 {2, 2, 2},    {2, 5, -1}, {3, 0, -1},   {3, 3, 1.5},
                 {3, 4, -0.5}, {4, 1, -1}, {4, 3, -0.5}, {4, 4, 2},
                 {4, 5, -0.5}, {5, 2, -1}, {5, 4, -0.5}, {5, 5, 1.5}}},
               {local_size,
                {{0, 0, 2},    {0, 1, -0.5}, {0, 3, -1},   {0, 4, -0.5},
                 {1, 0, -0.5}, {1, 1, 1.5},  {1, 5, -1},   {2, 2, 1.5},
                 {2, 3, -0.5}, {2, 4, -1},   {3, 0, -1},   {3, 2, -0.5},
                 {3, 3, 2},    {3, 5, -0.5}, {4, 0, -0.5}, {4, 2, -1},
                 {4, 4, 1.5},  {5, 1, -1},   {5, 3, -0.5}, {5, 5, 1.5}}},
               {local_size,
                {{0, 0, 1.5},  {0, 3, -1}, {0, 5, -0.5}, {1, 1, 2},
                 {1, 2, -1},   {1, 4, -1}, {2, 1, -1},   {2, 2, 3},
                 {2, 3, -1},   {2, 5, -1}, {3, 0, -1},   {3, 2, -1},
                 {3, 3, 2},    {4, 1, -1}, {4, 4, 1.5},  {4, 5, -0.5},
                 {5, 0, -0.5}, {5, 2, -1}, {5, 4, -0.5}, {5, 5, 2}}}}},
          engine(42)
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 4, 8, 12}));

        dist_mat = dd_mtx_type::create(exec, comm);
        x = dist_vec_type::create(ref, comm);
        y = dist_vec_type::create(ref, comm);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }


    gko::dim<2> size;
    gko::dim<2> local_size;
    std::shared_ptr<Partition> row_part;
    std::shared_ptr<Partition> col_part;

    gko::matrix_data<value_type, global_index_type> mat_input;
    std::array<matrix_data, 3> dist_input;
    std::array<local_matrix_data, 3> local_result;

    std::unique_ptr<dd_mtx_type> dist_mat;

    std::unique_ptr<dist_vec_type> x;
    std::unique_ptr<dist_vec_type> y;

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(DdMatrix, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(DdMatrix, ReadsDistributed)
{
    using value_type = typename TestFixture::value_type;
    using csr = typename TestFixture::local_matrix_type;
    auto rank = this->comm.rank();
    auto res_local = csr::create(this->exec);
    res_local->read(this->local_result[rank]);
    I<I<value_type>> local_restriction = {{1, 0, 0, 0}, {0, 1, 0, 0},
                                          {0, 0, 1, 0}, {0, 0, 0, 1},
                                          {0, 0, 0, 0}, {0, 0, 0, 0}};
    I<I<value_type>> non_local_restriction = {{0, 0}, {0, 0}, {0, 0},
                                              {0, 0}, {1, 0}, {0, 1}};
    I<I<value_type>> local_prolongation = {{1, 0, 0, 0, 0, 0},
                                           {0, 1, 0, 0, 0, 0},
                                           {0, 0, 1, 0, 0, 0},
                                           {0, 0, 0, 1, 0, 0}};
    I<I<value_type>> non_local_prolongation[] = {
        {{0}, {0}, {0}, {1}},
        {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
        {{1}, {0}, {0}, {0}}};

    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local, 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<csr>(this->dist_mat->get_restriction()->get_local_matrix()),
        local_restriction, 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<csr>(this->dist_mat->get_restriction()->get_non_local_matrix()),
        non_local_restriction, 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<csr>(this->dist_mat->get_prolongation()->get_local_matrix()),
        local_prolongation, 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<csr>(
            this->dist_mat->get_prolongation()->get_non_local_matrix()),
        non_local_prolongation[rank], 0);
}


TYPED_TEST(DdMatrix, CanApplyToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{I<I<value_type>>{
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}};
    I<I<value_type>> result[3] = {
        {{-4}, {-3}, {-2}, {-1}}, {{0}, {1}, {-1}, {0}}, {{1}, {2}, {3}, {4}}};
    auto rank = this->comm.rank();
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);
    this->y->read_distributed(vec_md, this->row_part);

    this->dist_mat->apply(this->x, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(DdMatrix, CanApplyToMultipleVectors)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    auto vec_md =
        gko::matrix_data<value_type, index_type>{I<I<value_type>>{{1, 2},
                                                                  {2, 4},
                                                                  {3, 6},
                                                                  {4, 8},
                                                                  {5, 10},
                                                                  {6, 12},
                                                                  {7, 14},
                                                                  {8, 16},
                                                                  {9, 18},
                                                                  {10, 20},
                                                                  {11, 22},
                                                                  {12, 24}}};
    I<I<value_type>> result[3] = {{{-4, -8}, {-3, -6}, {-2, -4}, {-1, -2}},
                                  {{0, 0}, {1, 2}, {-1, -2}, {0, 0}},
                                  {{1, 2}, {2, 4}, {3, 6}, {4, 8}}};
    auto rank = this->comm.rank();
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);
    this->y->read_distributed(vec_md, this->row_part);

    this->dist_mat->apply(this->x, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(DdMatrix, CanAdvancedApplyToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using dense_vec_type = typename TestFixture::dense_vec_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{I<I<value_type>>{
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}};
    I<I<value_type>> result[3] = {
        {{-3}, {-1}, {1}, {3}}, {{5}, {7}, {6}, {8}}, {{10}, {12}, {14}, {16}}};
    auto rank = this->comm.rank();
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);
    this->y->read_distributed(vec_md, this->row_part);
    auto alpha = gko::initialize<dense_vec_type>({1.0}, this->exec);
    auto beta = gko::initialize<dense_vec_type>({1.0}, this->exec);

    this->dist_mat->apply(alpha, this->x, beta, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(DdMatrix, CanAdvancedApplyToMultipleVectors)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using dense_vec_type = typename TestFixture::dense_vec_type;
    auto vec_md =
        gko::matrix_data<value_type, index_type>{I<I<value_type>>{{1, 2},
                                                                  {2, 4},
                                                                  {3, 6},
                                                                  {4, 8},
                                                                  {5, 10},
                                                                  {6, 12},
                                                                  {7, 14},
                                                                  {8, 16},
                                                                  {9, 18},
                                                                  {10, 20},
                                                                  {11, 22},
                                                                  {12, 24}}};
    I<I<value_type>> result[3] = {{{-3, -6}, {-1, -2}, {1, 2}, {3, 6}},
                                  {{5, 10}, {7, 14}, {6, 12}, {8, 16}},
                                  {{10, 20}, {12, 24}, {14, 28}, {16, 32}}};
    auto rank = this->comm.rank();
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);
    this->y->read_distributed(vec_md, this->row_part);
    auto alpha = gko::initialize<dense_vec_type>({1.0}, this->exec);
    auto beta = gko::initialize<dense_vec_type>({1.0}, this->exec);

    this->dist_mat->apply(alpha, this->x, beta, this->y);

    GKO_ASSERT_MTX_NEAR(this->y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(DdMatrix, CanColScale)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using matrix_data = typename TestFixture::local_matrix_data;
    using csr = typename TestFixture::local_matrix_type;
    auto local_size = gko::dim<2>{6, 6};
    auto vec_md = gko::matrix_data<value_type, index_type>{I<I<value_type>>{
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}};
    std::array<matrix_data, 3> scaled_result{
        {{local_size,
          {{0, 0, 2},  {0, 1, -2}, {0, 3, -4},   {1, 0, -1},   {1, 1, 6},
           {1, 2, -3}, {1, 4, -5}, {2, 1, -2},   {2, 2, 6},    {2, 5, -6},
           {3, 0, -1}, {3, 3, 6},  {3, 4, -2.5}, {4, 1, -2},   {4, 3, -2},
           {4, 4, 10}, {4, 5, -3}, {5, 2, -3},   {5, 4, -2.5}, {5, 5, 9}}},
         {local_size,
          {{0, 0, 10}, {0, 1, -3},   {0, 3, -8},   {0, 4, -2},   {1, 0, -2.5},
           {1, 1, 9},  {1, 5, -9},   {2, 2, 10.5}, {2, 3, -4},   {2, 4, -4},
           {3, 0, -5}, {3, 2, -3.5}, {3, 3, 16},   {3, 5, -4.5}, {4, 0, -2.5},
           {4, 2, -7}, {4, 4, 6},    {5, 1, -6},   {5, 3, -4},   {5, 5, 13.5}}},
         {local_size, {{0, 0, 13.5}, {0, 3, -12}, {0, 5, -4},   {1, 1, 20},
                       {1, 2, -11},  {1, 4, -7},  {2, 1, -10},  {2, 2, 33},
                       {2, 3, -12},  {2, 5, -8},  {3, 0, -9},   {3, 2, -11},
                       {3, 3, 24},   {4, 1, -10}, {4, 4, 10.5}, {4, 5, -4},
                       {5, 0, -4.5}, {5, 2, -11}, {5, 4, -3.5}, {5, 5, 16}}}}};
    auto rank = this->comm.rank();
    auto res_local = csr::create(this->exec);
    res_local->read(scaled_result[rank]);
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);

    this->dist_mat->col_scale(this->x);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local, 0);
}


TYPED_TEST(DdMatrix, CanRowScale)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using matrix_data = typename TestFixture::local_matrix_data;
    using csr = typename TestFixture::local_matrix_type;
    auto local_size = gko::dim<2>{6, 6};
    auto vec_md = gko::matrix_data<value_type, index_type>{I<I<value_type>>{
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}}};
    std::array<matrix_data, 3> scaled_result{
        {{local_size,
          {{0, 0, 2},  {0, 1, -1},   {0, 3, -1}, {1, 0, -2}, {1, 1, 6},
           {1, 2, -2}, {1, 4, -2},   {2, 1, -3}, {2, 2, 6},  {2, 5, -3},
           {3, 0, -4}, {3, 3, 6},    {3, 4, -2}, {4, 1, -5}, {4, 3, -2.5},
           {4, 4, 10}, {4, 5, -2.5}, {5, 2, -6}, {5, 4, -3}, {5, 5, 9}}},
         {local_size,
          {{0, 0, 10}, {0, 1, -2.5}, {0, 3, -5},   {0, 4, -2.5}, {1, 0, -3},
           {1, 1, 9},  {1, 5, -6},   {2, 2, 10.5}, {2, 3, -3.5}, {2, 4, -7},
           {3, 0, -8}, {3, 2, -4},   {3, 3, 16},   {3, 5, -4},   {4, 0, -2},
           {4, 2, -4}, {4, 4, 6},    {5, 1, -9},   {5, 3, -4.5}, {5, 5, 13.5}}},
         {local_size,
          {{0, 0, 13.5}, {0, 3, -9},  {0, 5, -4.5}, {1, 1, 20},  {1, 2, -10},
           {1, 4, -10},  {2, 1, -11}, {2, 2, 33},   {2, 3, -11}, {2, 5, -11},
           {3, 0, -12},  {3, 2, -12}, {3, 3, 24},   {4, 1, -7},  {4, 4, 10.5},
           {4, 5, -3.5}, {5, 0, -4},  {5, 2, -8},   {5, 4, -4},  {5, 5, 16}}}}};
    auto rank = this->comm.rank();
    auto res_local = csr::create(this->exec);
    res_local->read(scaled_result[rank]);
    this->dist_mat->read_distributed(this->dist_input[rank], this->row_part);
    this->x->read_distributed(vec_md, this->row_part);

    this->dist_mat->row_scale(this->x);

    GKO_ASSERT_MTX_NEAR(gko::as<csr>(this->dist_mat->get_local_matrix()),
                        res_local, 0);
}


#endif