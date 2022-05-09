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
#include <random>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/components/sparse_bitset.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "ginkgo/core/base/array.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename LocalGlobalIndexType>
class SparseBitset : public ::testing::Test {
protected:
    using local_index_type =
        typename std::tuple_element<0, LocalGlobalIndexType>::type;
    using global_index_type =
        typename std::tuple_element<1, LocalGlobalIndexType>::type;
    using flat_type =
        gko::sparse_bitset<0, local_index_type, global_index_type>;
    using hierarchical_type1 =
        gko::sparse_bitset<1, local_index_type, global_index_type>;
    using hierarchical_type2 =
        gko::sparse_bitset<2, local_index_type, global_index_type>;

    SparseBitset() : rand_engine(7653), size{12483}, universe_size{150343} {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        std::vector<global_index_type> local_indices;
        std::uniform_int_distribution<global_index_type> dist{
            0, universe_size - 1};
        for (int i = 0; i < size; i++) {
            local_indices.push_back(dist(rand_engine));
        }
        std::sort(local_indices.begin(), local_indices.end());
        local_indices.erase(
            std::unique(local_indices.begin(), local_indices.end()),
            local_indices.end());
        indices = gko::array<global_index_type>{ref, local_indices.begin(),
                                                local_indices.end()};
        dindices = gko::array<global_index_type>{exec, indices};
        size = indices.get_num_elems();
    }

    template <int depth>
    void assert_eq(const gko::sparse_bitset<depth, local_index_type,
                                            global_index_type>& expected,
                   const gko::sparse_bitset<depth, local_index_type,
                                            global_index_type>& result)
    {
        const auto dev_expected = expected.to_device();
        const auto dev_result = result.to_device();
        const auto num_blocks =
            gko::ceildiv(universe_size, gko::sparse_bitset_word_size);
        ASSERT_EQ(dev_expected.offsets, dev_result.offsets);
        const auto expected_bitmap = gko::make_array_view(
            ref, num_blocks, const_cast<gko::uint32*>(dev_expected.bitmaps));
        const auto result_bitmap = gko::make_array_view(
            exec, num_blocks, const_cast<gko::uint32*>(dev_result.bitmaps));
        const auto expected_ranks = gko::make_array_view(
            ref, num_blocks, const_cast<local_index_type*>(dev_expected.ranks));
        const auto result_ranks = gko::make_array_view(
            exec, num_blocks, const_cast<local_index_type*>(dev_result.ranks));
        GKO_ASSERT_ARRAY_EQ(expected_bitmap, result_bitmap);
        GKO_ASSERT_ARRAY_EQ(expected_ranks, result_ranks);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::default_random_engine rand_engine;
    int size;
    int universe_size;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    gko::array<global_index_type> indices;
    gko::array<global_index_type> dindices;
};

TYPED_TEST_SUITE(SparseBitset, gko::test::LocalGlobalIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SparseBitset, FlatFromSortedIndicesIsEquivalentToRef)
{
    using flat_type = typename TestFixture::flat_type;

    auto set =
        flat_type::from_indices_sorted(this->indices, this->universe_size);
    auto dset =
        flat_type::from_indices_sorted(this->dindices, this->universe_size);

    this->assert_eq(set, dset);
}


}  // namespace
