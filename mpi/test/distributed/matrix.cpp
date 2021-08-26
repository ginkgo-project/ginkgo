/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <memory>
#include <tuple>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"


namespace {


using global_index_type = gko::distributed::global_index_type;
using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalIndexType>
class Matrix : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueLocalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(ValueLocalIndexType())>::type;
    using local_entry = gko::matrix_data_entry<value_type, local_index_type>;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;
    using Mtx = gko::distributed::Matrix<value_type, local_index_type>;
    using GMtx = gko::matrix::Csr<value_type, local_index_type>;
    using Vec = gko::distributed::Vector<value_type, local_index_type>;
    using GVec = gko::matrix::Dense<value_type>;
    using Partition = gko::distributed::Partition<local_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;


    Matrix()
        : ref(gko::ReferenceExecutor::create()),
          comm(gko::mpi::communicator::create_world()),
          size{5, 5},

          mat_input{size,
                    {{0, 1, 1},
                     {0, 3, 2},
                     {1, 1, 3},
                     {1, 2, 4},
                     {2, 2, 5},
                     {2, 4, 6},
                     {3, 1, 7},
                     {3, 3, 8},
                     {4, 0, 9},
                     {4, 4, 10}}},
          global_mat_input{size,
                           {{0, 1, 1},
                            {0, 3, 2},
                            {1, 1, 3},
                            {1, 2, 4},
                            {2, 2, 5},
                            {2, 4, 6},
                            {3, 1, 7},
                            {3, 3, 8},
                            {4, 0, 9},
                            {4, 4, 10}}},
          x_input{gko::dim<2>{size[0], 1},
                  {{0, 0, 1}, {1, 0, 1}, {2, 0, 1}, {3, 0, 1}, {4, 0, 1}}},
          dist_input{{{size, {{0, 1, 1}, {0, 3, 2}, {1, 1, 3}, {1, 2, 4}}},
                      {size, {{2, 2, 5}, {2, 4, 6}, {3, 1, 7}, {3, 3, 8}}},
                      {size, {{4, 0, 9}, {4, 4, 10}}}}},
          part{Partition::build_from_contiguous(
              ref, gko::Array<global_index_type>(ref, {0, 2, 4, 5}))},
          dist_x{Vec::create(ref, comm)},
          dist_y{Vec::create(ref, comm)},
          global_x{GVec::create(ref)},
          global_y{GVec::create(ref)}
    {
        dist_x->read_distributed(x_input, part);
        dist_y->read_distributed(x_input, part);
        global_x->read(x_input);
        global_y->read(x_input);
    }

    void compare_local_with_global(const Vec* dist, const GVec* global,
                                   const Partition* part)
    {
        auto p_id = dist->get_communicator()->rank();
        auto global_idx = [&](const auto idx) {
            auto start = part->get_const_range_bounds()[p_id];
            return start + idx;
        };

        auto local = dist->get_local();

        for (int i = 0; i < local->get_size()[0]; ++i) {
            ASSERT_EQ(local->at(i), global->at(global_idx(i)));
        }
    }

    void compare_nnz_per_row(const Mtx* dist, const GMtx* global,
                             const Partition* part)
    {
        auto p_id = dist->get_communicator()->rank();
        auto global_idx = [&](const auto idx) {
            auto start = part->get_const_range_bounds()[p_id];
            return start + idx;
        };

        auto diag_row_ptrs = dist->get_local_diag()->get_const_row_ptrs();
        auto offdiag_row_ptrs = dist->get_local_offdiag()->get_const_row_ptrs();

        auto global_row_ptrs = global->get_const_row_ptrs();

        auto local_size = dist->get_local_diag()->get_size();
        for (int row = 0; row < local_size[0]; ++row) {
            auto diag_nnz = diag_row_ptrs[row + 1] - diag_row_ptrs[row];
            auto offdiag_nnz =
                offdiag_row_ptrs[row + 1] - offdiag_row_ptrs[row];
            auto global_nnz = global_row_ptrs[global_idx(row + 1)] -
                              global_row_ptrs[global_idx(row)];

            ASSERT_EQ(diag_nnz + offdiag_nnz, global_nnz);
        }
    }


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::mpi::communicator> comm;
    gko::dim<2> size;
    gko::matrix_data<value_type, global_index_type> mat_input;
    gko::matrix_data<value_type, local_index_type> global_mat_input;
    gko::matrix_data<value_type, global_index_type> x_input;
    std::shared_ptr<Partition> part;

    std::array<matrix_data, 3> dist_input;

    std::unique_ptr<Vec> dist_x;
    std::unique_ptr<Vec> dist_y;
    std::unique_ptr<GVec> global_x;
    std::unique_ptr<GVec> global_y;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueIndexTypes);


TYPED_TEST(Matrix, ReadsDistributedGlobalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm);
    auto global_mat = TestFixture::GMtx::create(this->ref);
    this->global_y->fill(gko::zero<value_type>());
    this->dist_y->fill(gko::zero<value_type>());

    dist_mat->read_distributed(this->mat_input, this->part);
    global_mat->read(this->global_mat_input);
    dist_mat->apply(this->dist_x.get(), this->dist_y.get());
    global_mat->apply(this->global_x.get(), this->global_y.get());

    this->compare_nnz_per_row(dist_mat.get(), global_mat.get(),
                              this->part.get());
    this->compare_local_with_global(this->dist_y.get(), this->global_y.get(),
                                    this->part.get());
}

TYPED_TEST(Matrix, ReadsDistributedLocalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref, this->comm);
    auto global_mat = TestFixture::GMtx::create(this->ref);
    this->global_y->fill(gko::zero<value_type>());
    this->dist_y->fill(gko::zero<value_type>());
    auto p_id = dist_mat->get_communicator()->rank();

    dist_mat->read_distributed(this->dist_input[p_id], this->part);
    global_mat->read(this->global_mat_input);
    dist_mat->apply(this->dist_x.get(), this->dist_y.get());
    global_mat->apply(this->global_x.get(), this->global_y.get());

    this->compare_nnz_per_row(dist_mat.get(), global_mat.get(),
                              this->part.get());
    this->compare_local_with_global(this->dist_y.get(), this->global_y.get(),
                                    this->part.get());
}

TYPED_TEST(Matrix, ConvertToCsrContiguousRanges)
{
    using value_type = typename TestFixture::value_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref);
    auto global_mat = TestFixture::GMtx::create(this->ref);
    auto converted = TestFixture::GMtx::create(this->ref);
    this->global_y->fill(gko::zero<value_type>());
    this->dist_y->fill(gko::zero<value_type>());
    dist_mat->read_distributed(this->mat_input, this->part);
    global_mat->read(this->global_mat_input);

    dist_mat->convert_to(converted.get());

    if (dist_mat->get_communicator()->rank() == 0) {
        GKO_ASSERT_MTX_NEAR(global_mat.get(), converted.get(), 0);
    } else {
        GKO_ASSERT(converted->get_num_stored_elements() == 0);
    }
}

TYPED_TEST(Matrix, ConvertToCsrContiguousRangesPermuted)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref);
    auto global_mat = TestFixture::GMtx::create(this->ref);
    auto converted = TestFixture::GMtx::create(this->ref);
    this->global_y->fill(gko::zero<value_type>());
    this->dist_y->fill(gko::zero<value_type>());
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {2, 1, 1, 0, 0}},
            3));
    dist_mat->read_distributed(this->mat_input, part);
    global_mat->read(this->global_mat_input);

    dist_mat->convert_to(converted.get());

    if (dist_mat->get_communicator()->rank() == 0) {
        GKO_ASSERT_MTX_NEAR(global_mat.get(), converted.get(), 0);
    } else {
        GKO_ASSERT(converted->get_num_stored_elements() == 0);
    }
}

TYPED_TEST(Matrix, ConvertToCsrScatteredRanges)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    auto dist_mat = TestFixture::Mtx::create(this->ref);
    auto converted = TestFixture::GMtx::create(this->ref);
    auto global_mat = TestFixture::GMtx::create(this->ref);
    this->global_y->fill(gko::zero<value_type>());
    this->dist_y->fill(gko::zero<value_type>());
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 1, 2, 0, 1}},
            3));
    dist_mat->read_distributed(this->mat_input, part);                               gko::distributed::data_placement::local);
    global_mat->read(this->global_mat_input);

    dist_mat->convert_to(converted.get());

    if (dist_mat->get_communicator()->rank() == 0) {
        GKO_ASSERT_MTX_NEAR(global_mat.get(), converted.get(), 0);
    } else {
        GKO_ASSERT(converted->get_num_stored_elements() == 0);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
