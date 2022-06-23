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
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/repartition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include <algorithm>
#include <memory>
#include <vector>

#include "core/test/utils.hpp"


namespace {


using global_index_type = gko::int64;
using comm_index_type = gko::distributed::comm_index_type;


template <typename LocalIndexType>
class Repartitioner : public ::testing::Test {
protected:
    using local_index_type = LocalIndexType;
    using Partition = gko::distributed::Partition<local_index_type>;
    using Vector = gko::distributed::Vector<double>;
    Repartitioner()
        : ref(gko::ReferenceExecutor::create()),
          from_comm(gko::mpi::communicator(MPI_COMM_WORLD)),
          from_part(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {0, 0, 1, 1, 2, 2, 3}),
                  4))),
          to1_part(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {0, 0, 0, 0, 0, 0, 0}),
                  1))),
          to2_part(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {0, 0, 0, 0, 1, 1, 1}),
                  2))),
          to3_part(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {0, 0, 0, 1, 1, 1, 2}),
                  3))),
          from_part_unordered(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {1, 1, 0, 2, 3, 0, 2}),
                  4))),
          to_part_unordered(gko::share(
              gko::distributed::Partition<local_index_type>::build_from_mapping(
                  this->ref,
                  gko::Array<comm_index_type>(this->ref, {1, 1, 0, 1, 0, 0, 0}),
                  2))),
          input{gko::dim<2>{7, 1},
                {{0, 0, 0},
                 {1, 0, 1},
                 {2, 0, 2},
                 {3, 0, 3},
                 {4, 0, 4},
                 {5, 0, 5},
                 {6, 0, 6}}},
          mtx_input{gko::dim<2>{7, 7},
                    {{0, 1, 1},
                     {0, 3, 2},
                     {1, 1, 3},
                     {1, 2, 4},
                     {2, 2, 5},
                     {2, 4, 6},
                     {3, 1, 7},
                     {3, 3, 8},
                     {4, 0, 9},
                     {4, 4, 10},
                     {5, 1, 11},
                     {5, 6, 12},
                     {6, 5, 13},
                     {6, 6, 14}}},
          from_vec(Vector::create(ref, from_comm))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::mpi::communicator from_comm;

    std::shared_ptr<Partition> from_part;
    std::shared_ptr<Partition> to1_part;
    std::shared_ptr<Partition> to2_part;
    std::shared_ptr<Partition> to3_part;

    std::shared_ptr<Partition> from_part_unordered;
    std::shared_ptr<Partition> to_part_unordered;

    gko::matrix_data<double, global_index_type> input;
    gko::matrix_data<double, global_index_type> mtx_input;
    std::shared_ptr<Vector> from_vec;

    std::shared_ptr<gko::distributed::repartitioner<local_index_type>>
        repartitioner;
};

TYPED_TEST_SUITE(Repartitioner, gko::test::IndexTypes);


TYPED_TEST(Repartitioner, CanCreateWithDifferentPartitions)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto rank = this->from_comm.rank();

    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);

    auto to_comm = repartitioner->get_to_communicator();
    GKO_ASSERT(this->from_comm.get() != to_comm.get());
    GKO_ASSERT(to_comm.size() == 2);
    GKO_ASSERT(to_comm.size() - to_comm.node_local_rank() >= 0);
    if (rank < 2) {
        ASSERT_TRUE(repartitioner->to_has_data());
        GKO_ASSERT(rank == to_comm.node_local_rank());
    } else {
        ASSERT_FALSE(repartitioner->to_has_data());
        GKO_ASSERT(rank - 2 == to_comm.node_local_rank());
    }
}


TYPED_TEST(Repartitioner, CanCreateWithSameNumberOfParts)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto to_part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref,
            gko::Array<comm_index_type>(this->ref, {3, 2, 2, 1, 1, 0, 0}), 4));

    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, to_part);

    auto to_comm = repartitioner->get_to_communicator();
    GKO_ASSERT(this->from_comm.get() == to_comm.get());
}


TYPED_TEST(Repartitioner, GatherToSmallerPartition_4_1)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to1_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref);
    auto to_vec_clone = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec_clone->read_distributed(this->input, this->to1_part.get());
    }
    this->from_vec->read_distributed(this->input, this->from_part.get());

    repartitioner->gather(this->from_vec.get(), to_vec.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(to_vec->get_local_vector(),
                            to_vec_clone->get_local_vector(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_vec, gko::dim<2>(0, 0));
    }
}


TYPED_TEST(Repartitioner, GatherToSmallerPartition_4_2)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref);
    auto to_vec_clone = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec_clone->read_distributed(this->input, this->to2_part.get());
    }
    this->from_vec->read_distributed(this->input, this->from_part.get());

    repartitioner->gather(this->from_vec.get(), to_vec.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(to_vec->get_local_vector(),
                            to_vec_clone->get_local_vector(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_vec, gko::dim<2>(0, 0));
    }
}


TYPED_TEST(Repartitioner, GatherToSmallerPartition_4_3)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to3_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref);
    auto to_vec_clone = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec_clone->read_distributed(this->input, this->to3_part.get());
    }
    this->from_vec->read_distributed(this->input, this->from_part.get());

    repartitioner->gather(this->from_vec.get(), to_vec.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(to_vec->get_local_vector(),
                            to_vec_clone->get_local_vector(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_vec, gko::dim<2>(0, 0));
    }
}


TYPED_TEST(Repartitioner, GatherToSmallerPartitionUnordered)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part_unordered,
            this->to_part_unordered);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    auto from_vec = Vector::create(this->ref, this->from_comm);

    ASSERT_THROW(repartitioner->gather(from_vec.get(), to_vec.get()),
                 gko::NotImplemented);
}


TYPED_TEST(Repartitioner, ThrowGatherToLargerPartition)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    to_vec->read_distributed(this->input, this->to2_part.get());
    this->from_vec->read_distributed(this->input, this->from_part.get());

    ASSERT_THROW(repartitioner->gather(to_vec.get(), this->from_vec.get()),
                 gko::MpiError);
}


TYPED_TEST(Repartitioner, ScatterToLargerPartition_4_1)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to1_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec->read_distributed(this->input, this->to1_part.get());
    }
    auto from_vec = Vector::create(this->ref);
    auto from_vec_clone = Vector::create(this->ref, this->from_comm);
    from_vec_clone->read_distributed(this->input, this->from_part.get());

    repartitioner->scatter(to_vec.get(), from_vec.get());

    GKO_ASSERT_MTX_NEAR(from_vec->get_local_vector(),
                        from_vec_clone->get_local_vector(), 0);
}


TYPED_TEST(Repartitioner, ScatterToLargerPartition_4_2)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec->read_distributed(this->input, this->to2_part.get());
    }
    auto from_vec = Vector::create(this->ref);
    auto from_vec_clone = Vector::create(this->ref, this->from_comm);
    from_vec_clone->read_distributed(this->input, this->from_part.get());

    repartitioner->scatter(to_vec.get(), from_vec.get());

    GKO_ASSERT_MTX_NEAR(from_vec->get_local_vector(),
                        from_vec_clone->get_local_vector(), 0);
}


TYPED_TEST(Repartitioner, ScatterToLargerPartition_4_3)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to3_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        to_vec->read_distributed(this->input, this->to3_part.get());
    }
    auto from_vec = Vector::create(this->ref);
    auto from_vec_clone = Vector::create(this->ref, this->from_comm);
    from_vec_clone->read_distributed(this->input, this->from_part.get());

    repartitioner->scatter(to_vec.get(), from_vec.get());

    GKO_ASSERT_MTX_NEAR(from_vec->get_local_vector(),
                        from_vec_clone->get_local_vector(), 0);
}


TYPED_TEST(Repartitioner, ThrowScatterToSmallerPartition)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    auto from_vec = Vector::create(this->ref, this->from_comm);

    ASSERT_THROW(repartitioner->scatter(from_vec.get(), to_vec.get()),
                 gko::MpiError);
}


TYPED_TEST(Repartitioner, GatherScatterIdentity)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Vector = typename TestFixture::Vector;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto to_vec = Vector::create(this->ref, to_comm);
    auto from_vec = Vector::create(this->ref, this->from_comm);
    from_vec->read_distributed(this->input, this->from_part.get());
    auto ref_vec = gko::clone(from_vec);

    repartitioner->gather(from_vec.get(), to_vec.get());
    repartitioner->scatter(to_vec.get(), from_vec.get());

    GKO_ASSERT_MTX_NEAR(ref_vec->get_local_vector(),
                        from_vec->get_local_vector(), 0);
}


TYPED_TEST(Repartitioner, GatherMatrix_4_1)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Partition = typename TestFixture::Partition;
    using Mtx = gko::distributed::Matrix<double, local_index_type>;
    using Csr = gko::matrix::Csr<double, local_index_type>;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to1_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto from_mat = Mtx::create(this->ref, this->from_comm);
    from_mat->read_distributed(this->mtx_input, this->from_part.get());
    auto result_mat = Mtx::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        result_mat->read_distributed(this->mtx_input, this->to1_part.get());
    }
    auto to_mat = Mtx::create(this->ref, to_comm);

    repartitioner->gather(from_mat.get(), to_mat.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_diag()),
                            gko::as<Csr>(result_mat->get_const_local_diag()),
                            0);
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_offdiag()),
                            gko::as<Csr>(result_mat->get_const_local_offdiag()),
                            0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_mat->get_size(), gko::dim<2>{});
    }
}


TYPED_TEST(Repartitioner, GatherMatrix_4_2)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Partition = typename TestFixture::Partition;
    using Mtx = gko::distributed::Matrix<double, local_index_type>;
    using Csr = gko::matrix::Csr<double, local_index_type>;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to2_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto from_mat = Mtx::create(this->ref, this->from_comm);
    from_mat->read_distributed(this->mtx_input, this->from_part.get());
    auto result_mat = Mtx::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        result_mat->read_distributed(this->mtx_input, this->to2_part.get());
    }
    auto to_mat = Mtx::create(this->ref, to_comm);

    repartitioner->gather(from_mat.get(), to_mat.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_diag()),
                            gko::as<Csr>(result_mat->get_const_local_diag()),
                            0);
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_offdiag()),
                            gko::as<Csr>(result_mat->get_const_local_offdiag()),
                            0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_mat->get_size(), gko::dim<2>{});
    }
}


TYPED_TEST(Repartitioner, GatherMatrix_4_3)
{
    using local_index_type = typename TestFixture::local_index_type;
    using Partition = typename TestFixture::Partition;
    using Mtx = gko::distributed::Matrix<double, local_index_type>;
    using Csr = gko::matrix::Csr<double, local_index_type>;
    auto repartitioner =
        gko::distributed::repartitioner<local_index_type>::create(
            this->from_comm, this->from_part, this->to3_part);
    auto to_comm = repartitioner->get_to_communicator();
    auto from_mat = Mtx::create(this->ref, this->from_comm);
    from_mat->read_distributed(this->mtx_input, this->from_part.get());
    auto result_mat = Mtx::create(this->ref, to_comm);
    if (repartitioner->to_has_data()) {
        result_mat->read_distributed(this->mtx_input, this->to3_part.get());
    }
    auto to_mat = Mtx::create(this->ref, to_comm);

    repartitioner->gather(from_mat.get(), to_mat.get());

    if (repartitioner->to_has_data()) {
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_diag()),
                            gko::as<Csr>(result_mat->get_const_local_diag()),
                            0);
        GKO_ASSERT_MTX_NEAR(gko::as<Csr>(to_mat->get_const_local_offdiag()),
                            gko::as<Csr>(result_mat->get_const_local_offdiag()),
                            0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(to_mat->get_size(), gko::dim<2>{});
    }
}


}  // namespace
