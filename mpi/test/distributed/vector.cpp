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
class Vector : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueLocalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(ValueLocalIndexType())>::type;
    using local_entry = gko::matrix_data_entry<value_type, local_index_type>;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;
    using Mtx = gko::distributed::Matrix<value_type, local_index_type>;
    using Vec = gko::distributed::Vector<value_type, local_index_type>;
    using GVec = gko::matrix::Dense<value_type>;
    using Partition = gko::distributed::Partition<local_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;

    Vector()
        : ref(gko::ReferenceExecutor::create()),
          comm(gko::mpi::communicator::create_world()),
          size{5, 5},
          vec_input{gko::dim<2>{size[0], 1},
                    {{0, 0, 1}, {1, 0, 2}, {2, 0, 3}, {3, 0, 4}, {4, 0, 5}}},
          part{Partition::build_from_mapping(
              ref, gko::Array<comm_index_type>(ref, {0, 0, 1, 1, 2}), 3)}
    {}

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

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::mpi::communicator> comm;
    gko::dim<2> size;
    gko::matrix_data<value_type, global_index_type> vec_input;
    std::shared_ptr<Partition> part;
};

TYPED_TEST_SUITE(Vector, gko::test::ValueIndexTypes);


TYPED_TEST(Vector, ReadsDistributedGlobalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist = TestFixture::Vec::create(this->ref, this->comm);
    auto global = TestFixture::GVec::create(this->ref);

    dist->read_distributed(this->vec_input, this->part);
    global->read(this->vec_input);

    this->compare_local_with_global(dist.get(), global.get(), this->part.get());
}

TYPED_TEST(Vector, ReadsDistributedLocalData)
{
    using value_type = typename TestFixture::value_type;
    auto dist = TestFixture::Vec::create(this->ref, this->comm);
    auto global = TestFixture::GVec::create(this->ref);
    gko::matrix_data<value_type, global_index_type> local_input[3] = {
        {gko::dim<2>{2, 1}, {{0, 0, 1}, {1, 0, 2}}},
        {gko::dim<2>{2, 1}, {{2, 0, 3}, {3, 0, 4}}},
        {gko::dim<2>{1, 1}, {{4, 0, 5}}}};
    auto rank = dist->get_communicator()->rank();

    dist->read_distributed(local_input[rank], this->part);
    global->read(this->vec_input);

    this->compare_local_with_global(dist.get(), global.get(), this->part.get());
}

TYPED_TEST(Vector, ConvertsToDense)
{
    using value_type = typename TestFixture::value_type;
    auto dist = TestFixture::Vec::create(this->ref);
    auto global = TestFixture::GVec::create(this->ref);
    auto gathered = TestFixture::GVec::create(this->ref);
    dist->read_distributed(this->vec_input, this->part);
    global->read(this->vec_input);

    dist->convert_to(gathered.get());

    auto rank = dist->get_communicator()->rank();
    if (rank == 0) {
        GKO_ASSERT_MTX_NEAR(gathered.get(), global.get(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(gathered->get_size(), gko::dim<2>(0, 0));
    }
}

TYPED_TEST(Vector, ConvertsToDenseReversePartition)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    auto dist = TestFixture::Vec::create(this->ref);
    auto global = TestFixture::GVec::create(this->ref);
    auto gathered = TestFixture::GVec::create(this->ref);
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, {this->ref, {2, 1, 1, 0, 0}}, 3));
    dist->read_distributed(this->vec_input, part);
    global->read(this->vec_input);

    dist->convert_to(gathered.get());

    auto rank = dist->get_communicator()->rank();
    if (rank == 0) {
        GKO_ASSERT_MTX_NEAR(gathered.get(), global.get(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(gathered->get_size(), gko::dim<2>(0, 0));
    }
}

TYPED_TEST(Vector, ConvertsToDenseScatteredPartition)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    auto dist = TestFixture::Vec::create(this->ref);
    auto global = TestFixture::GVec::create(this->ref);
    auto gathered = TestFixture::GVec::create(this->ref);
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, {this->ref, {0, 1, 2, 0, 1}}, 3));
    dist->read_distributed(this->vec_input, part);
    global->read(this->vec_input);

    dist->convert_to(gathered.get());

    auto rank = dist->get_communicator()->rank();
    if (rank == 0) {
        GKO_ASSERT_MTX_NEAR(gathered.get(), global.get(), 0);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(gathered->get_size(), gko::dim<2>(0, 0));
    }
}

}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
