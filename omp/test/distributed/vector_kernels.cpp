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

#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/distributed/vector_kernels.hpp"
#include "core/test/utils.hpp"


namespace {

using comm_index_type = gko::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Vector : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using global_entry = gko::matrix_data_entry<value_type, global_index_type>;
    using mtx = gko::matrix::Dense<value_type>;

    Vector()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::OmpExecutor::create()),
          engine(42)
    {}

    void validate(
        const gko::size_type num_cols,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            partition,
        const gko::distributed::Partition<local_index_type, global_index_type>*
            d_partition,
        gko::Array<global_entry> input)
    {
        gko::Array<global_entry> d_input{exec, input};
        for (comm_index_type part = 0; part < partition->get_num_parts();
             ++part) {
            auto num_rows =
                static_cast<gko::size_type>(partition->get_part_size(part));
            auto output = mtx::create(ref, gko::dim<2>{num_rows, num_cols});
            output->fill(gko::zero<value_type>());
            auto d_output = gko::clone(exec, output);

            gko::kernels::reference::distributed_vector::build_local(
                ref, input, partition, part, output.get(), value_type{});
            gko::kernels::omp::distributed_vector::build_local(
                exec, d_input, d_partition, part, d_output.get(), value_type{});

            GKO_ASSERT_MTX_NEAR(output, d_output, 0);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> exec;
    std::default_random_engine engine;
};
template <typename ValueType, typename IndexType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine>

gko::Array<gko::matrix_data_entry<ValueType, IndexType>>
generate_random_matrix_data_array(gko::size_type num_rows,
                                  gko::size_type num_cols,
                                  NonzeroDistribution&& nonzero_dist,
                                  ValueDistribution&& value_dist,
                                  Engine&& engine,
                                  std::shared_ptr<const gko::Executor> exec)
{
    auto md = gko::test::generate_random_matrix_data<ValueType, IndexType>(
        num_rows, num_cols, std::forward<NonzeroDistribution>(nonzero_dist),
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine));
    md.ensure_row_major_order();
    return gko::Array<gko::matrix_data_entry<ValueType, IndexType>>(
        exec, md.nonzeros.begin(), md.nonzeros.end());
}


TYPED_TEST_SUITE(Vector, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(Vector, BuildsLocalEmptyIsEquivalentToRef)
{
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    using global_entry = typename TestFixture::global_entry;
    gko::distributed::comm_index_type num_parts = 10;
    auto mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            100,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(0, partition.get(), d_partition.get(),
                   gko::Array<global_entry>{this->ref});
}


TYPED_TEST(Vector, BuildsLocalSmallIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 3;
    gko::size_type num_rows = 10;
    gko::size_type num_cols = 2;
    auto mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input =
        generate_random_matrix_data_array<value_type, global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(0,
                                               static_cast<int>(num_cols - 1)),
            std::uniform_real_distribution<gko::remove_complex<value_type>>(0,
                                                                            1),
            this->engine, this->ref);
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(num_cols, partition.get(), d_partition.get(), input);
}


TYPED_TEST(Vector, BuildsLocalIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using local_index_type = typename TestFixture::local_index_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::distributed::comm_index_type num_parts = 13;
    gko::size_type num_rows = 40;
    gko::size_type num_cols = 67;
    auto mapping =
        gko::test::generate_random_array<gko::distributed::comm_index_type>(
            num_rows,
            std::uniform_int_distribution<gko::distributed::comm_index_type>(
                0, num_parts - 1),
            this->engine, this->ref);
    auto input =
        generate_random_matrix_data_array<value_type, global_index_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<int>(0,
                                               static_cast<int>(num_cols - 1)),
            std::uniform_real_distribution<gko::remove_complex<value_type>>(0,
                                                                            1),
            this->engine, this->ref);
    auto partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->ref,
                                                                 mapping,
                                                                 num_parts);
    auto d_partition = gko::distributed::Partition<
        local_index_type, global_index_type>::build_from_mapping(this->exec,
                                                                 mapping,
                                                                 num_parts);

    this->validate(num_cols, partition.get(), d_partition.get(), input);
}


}  // namespace
