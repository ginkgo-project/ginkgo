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
    using Vec = gko::distributed::Vector<value_type, local_index_type,
                                         global_index_type>;
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
                      {size, {{4, 0, 9}, {4, 4, 10}}}}}
    {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::dim<2> size;
    gko::mpi::communicator comm;
    std::shared_ptr<Partition> part;

    gko::matrix_data<value_type, global_index_type> mat_input;
    std::array<matrix_data, 3> dist_input;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Matrix, ReadsDistributedGlobalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm, this->part);
    I<I<value_type>> res_diag[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_offdiag[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = dist_mat->get_communicator().rank();

    dist_mat->read_distributed(this->mat_input);

    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_diag(), res_diag[rank], 0);
    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_offdiag(), res_offdiag[rank], 0);
}


TYPED_TEST(Matrix, ReadsDistributedLocalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm, this->part);
    I<I<value_type>> res_diag[] = {{{0, 1}, {0, 3}}, {{6, 0}, {0, 8}}, {{10}}};
    I<I<value_type>> res_offdiag[] = {
        {{0, 2}, {4, 0}}, {{5, 0}, {0, 7}}, {{9}}};
    auto rank = dist_mat->get_communicator().rank();

    dist_mat->read_distributed(this->dist_input[rank]);

    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_diag(), res_diag[rank], 0);
    GKO_ASSERT_MTX_NEAR(dist_mat->get_local_offdiag(), res_offdiag[rank], 0);
}


}  // namespace
