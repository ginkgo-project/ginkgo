/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>


#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Matrix : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Mtx = gko::distributed::Matrix<value_type, local_index_type,
                                         global_index_type>;
    using Vec = gko::distributed::Vector<value_type>;
    using Partition =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          size{5, 5},
          comm(gko::mpi::communicator(MPI_COMM_WORLD)),
          part{Partition::build_from_contiguous(
              ref, gko::Array<global_index_type>(
                       ref, I<global_index_type>{0, 2, 4, 5}))},
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
          dist_input{{{size, {{0, 1, 1}, {0, 3, 2}, {1, 1, 3}, {1, 2, 4}}},
                      {size, {{2, 1, 5}, {2, 2, 6}, {3, 3, 8}, {3, 4, 7}}},
                      {size, {{4, 0, 9}, {4, 4, 10}}}}},
          engine(42)
    {}


    void assert_local_vector_equal_to_global_vector(
        const Vec* dist, const typename Vec::local_vector_type* dense,
        const Partition* part, int rank)
    {
        auto range_bounds = part->get_range_bounds();
        auto range_starting_indices = part->get_range_starting_indices();
        auto part_ids = part->get_part_ids();
        std::vector<global_index_type> gather_idxs;
        for (gko::size_type range_id = 0; range_id < part->get_num_ranges();
             ++range_id) {
            if (part_ids[range_id] == rank) {
                for (global_index_type global_row = range_bounds[range_id];
                     global_row < range_bounds[range_id + 1]; ++global_row) {
                    gather_idxs.push_back(global_row);
                }
            }
        }
        auto gather_idxs_view = gko::Array<global_index_type>::view(
            this->ref, gather_idxs.size(), gather_idxs.data());
        auto gathered_local = dense->row_gather(&gather_idxs_view);

        GKO_ASSERT_MTX_NEAR(dist->get_local_vector(), gathered_local.get(),
                            r<value_type>::value);
    }


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::dim<2> size;
    gko::mpi::communicator comm;
    std::shared_ptr<Partition> part;

    gko::matrix_data<value_type, global_index_type> mat_input;
    std::array<matrix_data, 3> dist_input;
    std::default_random_engine engine;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Matrix, ReadsDistributedGlobalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm);
    I<I<value_type>> res_diag[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_offdiag[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = dist_mat->get_communicator().rank();

    dist_mat->read_distributed(this->mat_input, this->part.get());

    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_diag(), res_diag[rank], 0);
    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_offdiag(), res_offdiag[rank], 0);
}


TYPED_TEST(Matrix, ReadsDistributedLocalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm);
    I<I<value_type>> res_diag[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_offdiag[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = dist_mat->get_communicator().rank();

    dist_mat->read_distributed(this->dist_input[rank], this->part.get());

    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_diag(), res_diag[rank], 0);
    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_offdiag(), res_offdiag[rank], 0);
}

TYPED_TEST(Matrix, CanApplyToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    auto vec_md = gko::matrix_data<value_type, index_type>{
        I<I<value_type>>{{1}, {2}, {3}, {4}, {5}}};
    auto dist_mat = TestFixture::Mtx ::create(this->ref, this->comm);
    auto x = TestFixture::Vec ::create(this->ref, this->comm);
    auto y = TestFixture::Vec ::create(this->ref, this->comm);
    I<I<value_type>> result[3] = {{{10}, {18}}, {{28}, {67}}, {{59}}};
    auto rank = this->comm.rank();
    dist_mat->read_distributed(this->mat_input, this->part.get());
    x->read_distributed(vec_md, this->part.get());
    y->read_distributed(vec_md, this->part.get());
    y->fill(gko::zero<value_type>());

    dist_mat->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y->get_local_vector(), result[rank], 0);
}


TYPED_TEST(Matrix, CanApplyToSingleVectorLarge)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    auto rank = this->comm.rank();
    gko::size_type num_rows = 100;
    int num_parts = this->comm.size();
    auto vec_md =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            num_rows, 1, std::uniform_int_distribution<int>(1, 1),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            this->engine);
    auto mat_md =
        gko::test::generate_random_matrix_data<value_type, global_index_type>(
            num_rows, num_rows,
            std::uniform_int_distribution<int>(0,
                                               static_cast<int>(num_rows) - 1),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            this->engine);
    auto mapping = gko::test::generate_random_array<comm_index_type>(
        num_rows, std::uniform_int_distribution<int>(0, num_parts - 1),
        this->engine, this->ref);
    auto part = gko::share(
        gko::distributed::Partition<local_index_type, global_index_type>::
            build_from_mapping(this->ref, mapping, num_parts));
    auto dist_mat = TestFixture::Mtx ::create(this->ref, this->comm);
    auto csr_mat =
        gko::matrix::Csr<value_type, global_index_type>::create(this->ref);
    auto x = TestFixture::Vec ::create(this->ref, this->comm);
    auto y = TestFixture::Vec ::create(
        this->ref, this->comm, gko::dim<2>{num_rows, 1},
        gko::dim<2>{static_cast<gko::size_type>(part->get_part_size(rank)), 1});
    auto dense_x = gko::matrix::Dense<value_type>::create(this->ref);
    auto dense_y = gko::matrix::Dense<value_type>::create(
        this->ref, gko::dim<2>{num_rows, 1});
    dist_mat->read_distributed(mat_md, part.get());
    csr_mat->read(mat_md);
    x->read_distributed(vec_md, part.get());
    dense_x->read(vec_md);

    dist_mat->apply(x.get(), y.get());
    csr_mat->apply(dense_x.get(), dense_y.get());

    this->assert_local_vector_equal_to_global_vector(y.get(), dense_y.get(),
                                                     part.get(), rank);
}


}  // namespace
