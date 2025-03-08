// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector_kernels.hpp"

#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/index_range.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


// workaround for cudafe 11.0 bug
using gko::irange;


template <typename T>
class Bitvector : public CommonTestFixture {
protected:
    using index_type = T;
    using device_type = gko::bitvector<index_type>;
    using storage_type = typename device_type::storage_type;
    constexpr static auto block_size = device_type::block_size;

    Bitvector()
        : rng{67193}, sizes{0,    1,    2,    16,    31,    32,  33,
                            40,   63,   64,   65,    127,   128, 129,
                            1000, 1024, 2000, 10000, 100000}
    {}

    gko::array<index_type> create_random_values(index_type num_values,
                                                index_type size)
    {
        std::vector<index_type> values(num_values);
        std::uniform_int_distribution<index_type> dist(
            0, std::max(size - 1, index_type{}));
        for (auto& value : values) {
            value = dist(this->rng);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return gko::array<index_type>{this->ref, values.begin(), values.end()};
    }

    std::default_random_engine rng;
    std::vector<index_type> sizes;
};

TYPED_TEST_SUITE(Bitvector, gko::test::IndexTypes, TypenameNameGenerator);


// nvcc doesn't like device lambdas inside class member functions

template <typename IndexType>
void run_device(std::shared_ptr<const gko::EXEC_TYPE> exec,
                const gko::device_bitvector<IndexType> bv,
                const gko::device_bitvector<IndexType> dbv,
                gko::array<bool>& output_bools,
                gko::array<IndexType>& output_ranks,
                gko::array<bool>& doutput_bools,
                gko::array<IndexType>& doutput_ranks)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto bv, auto output_bool, auto output_rank) {
            output_bool[i] = bv.get(i);
            output_rank[i] = bv.rank(i);
        },
        dbv.size(), dbv, doutput_bools, doutput_ranks);
    for (auto i : gko::irange{bv.size()}) {
        output_bools.get_data()[i] = bv.get(i);
        output_ranks.get_data()[i] = bv.rank(i);
    }
}


TYPED_TEST(Bitvector, ComputeBitsAndRanksIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using storage_type = typename TestFixture::storage_type;
    constexpr auto block_size = TestFixture::block_size;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        for (auto num_values :
             {index_type{}, size / 10, size / 4, size / 2, size}) {
            SCOPED_TRACE(num_values);
            auto values = this->create_random_values(num_values, size);
            num_values = values.get_size();
            gko::array<index_type> dvalues{this->exec, values};
            const auto num_blocks =
                static_cast<gko::size_type>(gko::ceildiv(size, block_size));
            gko::array<index_type> ranks{this->ref, num_blocks};
            gko::array<index_type> dranks{this->exec, num_blocks};
            gko::array<storage_type> bits{this->ref, num_blocks};
            gko::array<storage_type> dbits{this->exec, num_blocks};
            dranks.fill(-1);
            dbits.fill(~storage_type{});

            gko::kernels::reference::bitvector::compute_bits_and_ranks(
                this->ref, values.get_const_data(), num_values, size,
                bits.get_data(), ranks.get_data());
            gko::kernels::GKO_DEVICE_NAMESPACE::bitvector::
                compute_bits_and_ranks(this->exec, dvalues.get_const_data(),
                                       num_values, size, dbits.get_data(),
                                       dranks.get_data());

            GKO_ASSERT_ARRAY_EQ(bits, dbits);
            GKO_ASSERT_ARRAY_EQ(ranks, dranks);
        }
    }
}


TYPED_TEST(Bitvector, AccessIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using storage_type = typename TestFixture::storage_type;
    constexpr auto block_size = TestFixture::block_size;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        for (auto num_values :
             {index_type{}, size / 10, size / 4, size / 2, size}) {
            SCOPED_TRACE(num_values);
            auto values = this->create_random_values(num_values, size);
            num_values = values.get_size();
            gko::array<index_type> dvalues{this->exec, values};

            auto bv =
                gko::bitvector<index_type>::from_sorted_indices(values, size);
            auto dbv =
                gko::bitvector<index_type>::from_sorted_indices(dvalues, size);

            const auto usize = static_cast<gko::size_type>(size);
            gko::array<bool> output_bools{this->ref, usize};
            gko::array<index_type> output_ranks{this->ref, usize};
            gko::array<bool> doutput_bools{this->exec, usize};
            gko::array<index_type> doutput_ranks{this->exec, usize};
            doutput_bools.fill(true);
            doutput_ranks.fill(-1);
            run_device(this->exec, bv.device_view(), dbv.device_view(),
                       output_bools, output_ranks, doutput_bools,
                       doutput_ranks);
        }
    }
}
