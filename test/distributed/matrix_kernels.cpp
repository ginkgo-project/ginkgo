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

#include "core/distributed/matrix_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Matrix : public ::testing::Test {
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
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;

    Matrix() : engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    void validate(
        const gko::distributed::Partition<local_index_type, global_index_type>*
            row_partition,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            col_partition,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            d_row_partition,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            d_col_partition,
        gko::device_matrix_data<value_type, global_index_type> input)
    {
        gko::device_matrix_data<value_type, global_index_type> d_input{exec,
                                                                       input};
        for (comm_index_type part = 0; part < row_partition->get_num_parts();
             ++part) {
            gko::array<local_index_type> diag_row_idxs{ref};
            gko::array<local_index_type> diag_col_idxs{ref};
            gko::array<value_type> diag_values{ref};
            gko::array<local_index_type> d_diag_row_idxs{exec};
            gko::array<local_index_type> d_diag_col_idxs{exec};
            gko::array<value_type> d_diag_values{exec};
            gko::array<local_index_type> offdiag_row_idxs{ref};
            gko::array<local_index_type> offdiag_col_idxs{ref};
            gko::array<value_type> offdiag_values{ref};
            gko::array<local_index_type> d_offdiag_row_idxs{exec};
            gko::array<local_index_type> d_offdiag_col_idxs{exec};
            gko::array<value_type> d_offdiag_values{exec};
            gko::array<local_index_type> gather_idxs{ref};
            gko::array<local_index_type> d_gather_idxs{exec};
            gko::array<comm_index_type> recv_sizes{
                ref,
                static_cast<gko::size_type>(row_partition->get_num_parts())};
            gko::array<comm_index_type> d_recv_sizes{
                exec,
                static_cast<gko::size_type>(row_partition->get_num_parts())};
            gko::array<global_index_type> local_to_global_col{ref};
            gko::array<global_index_type> d_local_to_global_col{exec};

            gko::kernels::reference::distributed_matrix::build_diag_offdiag(
                ref, input, row_partition, col_partition, part, diag_row_idxs,
                diag_col_idxs, diag_values, offdiag_row_idxs, offdiag_col_idxs,
                offdiag_values, gather_idxs, recv_sizes.get_data(),
                local_to_global_col);
            gko::kernels::EXEC_NAMESPACE::distributed_matrix::
                build_diag_offdiag(
                    exec, d_input, d_row_partition, d_col_partition, part,
                    d_diag_row_idxs, d_diag_col_idxs, d_diag_values,
                    d_offdiag_row_idxs, d_offdiag_col_idxs, d_offdiag_values,
                    d_gather_idxs, d_recv_sizes.get_data(),
                    d_local_to_global_col);

            assert_device_matrix_data_equal(diag_row_idxs, diag_col_idxs,
                                            diag_values, d_diag_row_idxs,
                                            d_diag_col_idxs, d_diag_values);
            assert_device_matrix_data_equal(
                offdiag_row_idxs, offdiag_col_idxs, offdiag_values,
                d_offdiag_row_idxs, d_offdiag_col_idxs, d_offdiag_values);
            GKO_ASSERT_ARRAY_EQ(gather_idxs, d_gather_idxs);
            GKO_ASSERT_ARRAY_EQ(recv_sizes, d_recv_sizes);
            GKO_ASSERT_ARRAY_EQ(local_to_global_col, d_local_to_global_col);
        }
    }

    template <typename AI, typename AV>
    void assert_device_matrix_data_equal(AI& row_idxs, AI& col_idxs, AV& values,
                                         AI& d_row_idxs, AI& d_col_idxs,
                                         AV& d_values)
    {
        GKO_ASSERT_ARRAY_EQ(row_idxs, d_row_idxs);
        GKO_ASSERT_ARRAY_EQ(col_idxs, d_col_idxs);
        GKO_ASSERT_ARRAY_EQ(values, d_values);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::default_random_engine engine;
};

TYPED_TEST_SUITE(Matrix, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Matrix, BuildsDiagOffdiagEmptyIsSameAsRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> mapping{this->ref, {1, 0, 2, 2, 0, 1, 1, 2}};
    comm_index_type num_parts = 3;

    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(
        partition.get(), partition.get(), d_partition.get(), d_partition.get(),
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(Matrix, BuildsLocalSmallIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 10;
    auto mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(0, static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition.get(), partition.get(), d_partition.get(),
                   d_partition.get(), input);
}


TYPED_TEST(Matrix, BuildsLocalIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 67;
    gko::size_type num_cols = 67;
    auto mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(static_cast<int>(num_cols),
                                           static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition.get(), partition.get(), d_partition.get(),
                   d_partition.get(), input);
}


TYPED_TEST(Matrix, BuildsDiagOffdiagEmptyWithColPartitionIsSameAsRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<comm_index_type> row_mapping{this->ref,
                                            {1, 0, 2, 2, 0, 1, 1, 2}};
    gko::array<comm_index_type> col_mapping{this->ref,
                                            {0, 0, 2, 2, 2, 1, 1, 1}};
    comm_index_type num_parts = 3;

    auto row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(
        row_partition.get(), col_partition.get(), d_row_partition.get(),
        d_col_partition.get(),
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(Matrix, BuildsLocalSmallWithColPartitionIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 10;
    auto row_mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto col_mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(0, static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(row_partition.get(), col_partition.get(),
                   d_row_partition.get(), d_col_partition.get(), input);
}


TYPED_TEST(Matrix, BuildsLocalWithColPartitionIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 67;
    gko::size_type num_cols = 67;
    auto row_mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto col_mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input = gko::test::generate_random_device_matrix_data<
        value_type, global_index_type>(
        num_rows, num_cols,
        std::uniform_int_distribution<int>(static_cast<int>(num_cols),
                                           static_cast<int>(num_cols - 1)),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 1),
        this->engine, this->ref);

    auto row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 row_mapping,
                                                                 num_parts);
    auto d_row_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 row_mapping,
                                                                 num_parts);
    auto col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 col_mapping,
                                                                 num_parts);
    auto d_col_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 col_mapping,
                                                                 num_parts);

    this->validate(row_partition.get(), col_partition.get(),
                   d_row_partition.get(), d_col_partition.get(), input);
}

}  // namespace
