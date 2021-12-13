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

Copyright(c) 2017 - 2021,
    the Ginkgo authors All rights reserved.

    Redistribution and use in source and binary forms,
    with or without modification,
    are permitted provided that the following conditions are met :

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the
            documentation and /
        or
        other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse
        or
        promote products derived from
        this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL
    DAMAGES(INCLUDING, BUT NOT LIMITED TO,
            PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
            LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
                INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT(INCLUDING NEGLIGENCE OR
                OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*
            *****************************<
                GINKGO LICENSE>******************************* /

#include <ginkgo/core/distributed/matrix.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/distributed/matrix_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


        namespace
{
    using global_index_type = gko::distributed::global_index_type;
    using comm_index_type = gko::distributed::comm_index_type;


    template <typename ValueLocalIndexType>
    class Matrix : public ::testing::Test {
    protected:
        using value_type =
            typename std::tuple_element<0,
                                        decltype(ValueLocalIndexType())>::type;
        using local_index_type =
            typename std::tuple_element<1,
                                        decltype(ValueLocalIndexType())>::type;
        using local_entry =
            gko::matrix_data_entry<value_type, local_index_type>;
        using global_entry =
            gko::matrix_data_entry<value_type, global_index_type>;
        using Mtx = gko::matrix::Csr<value_type, local_index_type>;
        using GMtx = gko::matrix::Csr<value_type, local_index_type>;

        Matrix() : rand_engine(42) {}

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

        std::shared_ptr<gko::ReferenceExecutor> ref;
        std::shared_ptr<gko::EXEC_TYPE> exec;

        std::ranlux48 rand_engine;
    };

    TYPED_TEST_SUITE(Matrix, gko::test::ValueIndexTypes);


    template <typename Engine>
    gko::Array<global_index_type> generate_index_map(
        std::shared_ptr<gko::Executor> exec, gko::size_type num_local,
        gko::size_type num_global, Engine && engine)
    {
        std::vector<global_index_type> indices(num_global);
        std::iota(begin(indices), end(indices), global_index_type{0});
        std::shuffle(begin(indices), end(indices), engine);

        return {exec, begin(indices), begin(indices) + num_local};
    }


    TYPED_TEST(Matrix, CombineToGlobalDataAndFilterIsEquivalentToRef)
    {
        using value_type = typename TestFixture::value_type;
        using local_index_type = typename TestFixture::local_index_type;
        using local_entry = typename TestFixture::local_entry;
        using global_entry = typename TestFixture::global_entry;
        using local_dmd = gko::device_matrix_data<value_type, local_index_type>;
        gko::dim<2> result_size(20, 53);
        gko::span rows(4, 17);
        gko::span cols(11, 39);
        auto diag_md = gko::test::generate_random_matrix_data<value_type,
                                                              local_index_type>(
            20, 20, std::uniform_int_distribution<local_index_type>(0, 20),
            std::normal_distribution<gko::remove_complex<value_type>>(0., 1.),
            this->rand_engine);
        auto offdiag_md = gko::test::generate_random_matrix_data<
            value_type, local_index_type>(
            20, 33, std::uniform_int_distribution<local_index_type>(0, 33),
            std::normal_distribution<gko::remove_complex<value_type>>(0., 1.),
            this->rand_engine);
        gko::device_matrix_data<value_type, global_index_type> result{
            this->ref};
        gko::device_matrix_data<value_type, global_index_type> d_result{
            this->exec};
        auto map_row = generate_index_map(this->ref, result_size[0], 60,
                                          this->rand_engine);
        auto map_col = generate_index_map(
            this->ref, result_size[1] - result_size[0], 60, this->rand_engine);
        gko::Array<global_index_type> d_map_row{this->exec, map_row};
        gko::Array<global_index_type> d_map_col{this->exec, map_col};

        gko::kernels::reference::distributed_matrix::
            combine_to_global_data_and_filter(
                this->ref, result_size,
                local_dmd::create_view_from_host(this->ref, diag_md),
                local_dmd::create_view_from_host(this->ref, offdiag_md),
                map_row.get_const_data(), map_col.get_const_data(), rows, cols,
                result);
        gko::kernels::EXEC_NAMESPACE::distributed_matrix::
            combine_to_global_data_and_filter(
                this->exec, result_size,
                local_dmd::create_view_from_host(this->exec, diag_md),
                local_dmd::create_view_from_host(this->exec, offdiag_md),
                d_map_row.get_const_data(), d_map_col.get_const_data(), rows,
                cols, d_result);

        GKO_ASSERT_ARRAY_EQ(result.nonzeros, d_result.nonzeros);
    }


    TYPED_TEST(Matrix, CheckIndicesWithinSpanIsEquivalentToRef)
    {
        using value_type = typename TestFixture::value_type;
        using local_index_type = typename TestFixture::local_index_type;
        auto indices = gko::test::generate_random_array<local_index_type>(
            100, std::uniform_int_distribution<local_index_type>(0, 100),
            this->rand_engine, this->ref);
        gko::Array<local_index_type> d_indices{this->exec, indices};
        auto map = generate_index_map(this->ref, 100, 1000, this->rand_engine);
        gko::Array<global_index_type> d_map{this->exec, map};
        gko::span valid_span{33, 59};
        gko::Array<bool> result{this->ref};
        gko::Array<bool> d_result{this->exec};


        gko::kernels::reference::distributed_matrix::check_indices_within_span(
            this->ref, indices, map, valid_span, result);
        gko::kernels::EXEC_NAMESPACE::distributed_matrix::
            check_indices_within_span(this->exec, d_indices, d_map, valid_span,
                                      d_result);

        GKO_ASSERT_ARRAY_EQ(result, d_result);
    }


}  // namespace
