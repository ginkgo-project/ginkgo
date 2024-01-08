// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/vector_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Vector : public CommonTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;
    using mtx = gko::matrix::Dense<value_type>;

    Vector() : engine(42) {}

    void validate(
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            partition,
        gko::ptr_param<const gko::experimental::distributed::Partition<
            local_index_type, global_index_type>>
            d_partition,
        const gko::device_matrix_data<value_type, global_index_type>& input)
    {
        gko::device_matrix_data<value_type, global_index_type> d_input{exec,
                                                                       input};
        for (comm_index_type part = 0; part < partition->get_num_parts();
             ++part) {
            auto num_rows =
                static_cast<gko::size_type>(partition->get_part_size(part));
            auto output =
                mtx::create(ref, gko::dim<2>{num_rows, input.get_size()[1]});
            output->fill(gko::zero<value_type>());
            auto d_output = gko::clone(exec, output);

            gko::kernels::reference::distributed_vector::build_local(
                ref, input, partition.get(), part, output.get());
            gko::kernels::EXEC_NAMESPACE::distributed_vector::build_local(
                exec, d_input, d_partition.get(), part, d_output.get());

            GKO_ASSERT_MTX_NEAR(output, d_output, 0);
        }
    }

    std::default_random_engine engine;
};

TYPED_TEST_SUITE(Vector, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


template <typename ValueType, typename IndexType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine>
gko::device_matrix_data<ValueType, IndexType> generate_random_matrix_data_array(
    gko::size_type num_rows, gko::size_type num_cols,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const gko::Executor> exec)
{
    auto md = gko::test::generate_random_matrix_data<ValueType, IndexType>(
        num_rows, num_cols, std::forward<NonzeroDistribution>(nonzero_dist),
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine));
    md.sort_row_major();
    return gko::device_matrix_data<ValueType, IndexType>::create_from_host(exec,
                                                                           md);
}


TYPED_TEST(Vector, BuildsLocalEmptyIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 10;
    auto mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        100,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(
        partition, d_partition,
        gko::device_matrix_data<value_type, global_index_type>{this->ref});
}


TYPED_TEST(Vector, BuildsLocalSmallIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 2;
    auto mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input =
        generate_random_matrix_data_array<value_type, global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(0,
                                               static_cast<int>(num_cols - 1)),
            std::uniform_real_distribution<gko::remove_complex<value_type>>(0,
                                                                            1),
            this->engine, this->ref);
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, d_partition, input);
}


TYPED_TEST(Vector, BuildsLocalIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::experimental::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 40;
    gko::size_type num_cols = 67;
    auto mapping = gko::test::generate_random_array<
        gko::experimental::distributed::comm_index_type>(
        num_rows,
        std::uniform_int_distribution<
            gko::experimental::distributed::comm_index_type>(0, num_parts - 1),
        this->engine, this->ref);
    auto input =
        generate_random_matrix_data_array<value_type, global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(0,
                                               static_cast<int>(num_cols - 1)),
            std::uniform_real_distribution<gko::remove_complex<value_type>>(0,
                                                                            1),
            this->engine, this->ref);
    auto partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::experimental::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(partition, d_partition, input);
}
