// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/index_range.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


template <typename T>
class RangeMinimumQuery : public CommonTestFixture {
protected:
    using index_type = T;
    using storage_type = std::make_unsigned_t<index_type>;
    using block_argmin_storage_type =
        gko::kernels::reference::range_minimum_query::block_argmin_storage_type<
            index_type>;
    using superblock_storage_type =
        gko::range_minimum_query_superblocks<index_type>;
    constexpr static auto block_size =
        gko::device_range_minimum_query<index_type>::block_size;

    RangeMinimumQuery()
        : rng{19654}, sizes{0, 1, 7, 8, 9, 1023, 1024, 1025, 10000, 100000}
    {}

    gko::array<index_type> create_random_values(index_type size)
    {
        gko::array<index_type> data{this->ref,
                                    static_cast<gko::size_type>(size)};
        std::uniform_int_distribution<index_type> dist{
            0, std::numeric_limits<index_type>::max()};
        for (auto i : gko::irange{size}) {
            data.get_data()[i] = dist(rng);
        }
        return data;
    }

    std::pair<gko::array<index_type>, gko::array<index_type>>
    create_random_queries(index_type size)
    {
        std::vector<index_type> begins;
        std::vector<index_type> ends;
        std::uniform_int_distribution<index_type> dist{0, size - 1};
        const auto add_query = [&](index_type begin, index_type end) {
            assert(begin <= end);
            assert(begin >= 0);
            assert(end < size);
            begins.push_back(begin);
            ends.push_back(end);
        };
        if (size > 0) {
            for (const auto i : gko::irange{size}) {
                // singleton queries
                add_query(i, i);
            }
            for (const auto i : gko::irange{100}) {
                // random block-local queries
                const auto begin = dist(rng);
                std::uniform_int_distribution<index_type> end_dist{
                    begin, std::min(size - 1,
                                    (begin / block_size + 1) * block_size - 1)};
                add_query(begin, end_dist(rng));
            }
            for (const auto i : gko::irange{100}) {
                // random two-block queries
                const auto begin = dist(rng);
                std::uniform_int_distribution<index_type> end_dist{
                    std::min(size - 1,
                             (begin / block_size + 1) * block_size - 1),
                    std::min(size - 1,
                             (begin / block_size + 2) * block_size - 1)};
                add_query(begin, end_dist(rng));
            }
            for (const auto i : gko::irange{100}) {
                // random arbitrary queries
                const auto begin = dist(rng);
                std::uniform_int_distribution<index_type> end_dist{begin,
                                                                   size - 1};
                add_query(begin, end_dist(rng));
            }
        }
        return std::make_pair(
            gko::array<index_type>{this->ref, begins.begin(), begins.end()},
            gko::array<index_type>{this->ref, ends.begin(), ends.end()});
    }

    std::default_random_engine rng;
    std::vector<index_type> sizes;
};

TYPED_TEST_SUITE(RangeMinimumQuery, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(RangeMinimumQuery, ComputeLookupSmallAndLargeIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using storage_type = typename TestFixture::storage_type;
    using block_argmin_storage_type =
        typename TestFixture::block_argmin_storage_type;
    using superblock_storage_type =
        typename TestFixture::superblock_storage_type;
    constexpr auto block_size = TestFixture::block_size;
    constexpr auto block_argmin_num_bits = gko::ceil_log2_constexpr(block_size);
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        auto values = this->create_random_values(size);
        auto dvalues = gko::array<index_type>{this->exec, values};
        const auto num_blocks =
            static_cast<gko::size_type>(gko::ceildiv(size, block_size));
        gko::array<index_type> block_min{this->ref, num_blocks};
        gko::array<index_type> dblock_min{this->exec, num_blocks};
        const auto block_argmin_storage_size =
            static_cast<gko::size_type>(block_argmin_storage_type::storage_size(
                num_blocks, block_argmin_num_bits));
        gko::array<gko::uint32> block_argmin_storage{this->ref,
                                                     block_argmin_storage_size};
        gko::array<gko::uint32> dblock_argmin_storage{
            this->exec, block_argmin_storage_size};
        block_argmin_storage.fill(0);
        dblock_argmin_storage.fill(0);
        block_argmin_storage_type block_argmin{
            block_argmin_storage.get_data(), block_argmin_num_bits,
            static_cast<index_type>(num_blocks)};
        block_argmin_storage_type dblock_argmin{
            dblock_argmin_storage.get_data(), block_argmin_num_bits,
            static_cast<index_type>(num_blocks)};
        gko::array<gko::uint16> block_type{this->ref, num_blocks};
        gko::array<gko::uint16> dblock_type{this->exec, num_blocks};

        gko::kernels::reference::range_minimum_query::compute_lookup_small(
            this->ref, values.get_const_data(), size, block_argmin,
            block_min.get_data(), block_type.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::range_minimum_query::
            compute_lookup_small(this->exec, dvalues.get_const_data(), size,
                                 dblock_argmin, dblock_min.get_data(),
                                 dblock_type.get_data());

        GKO_ASSERT_ARRAY_EQ(block_min, dblock_min);
        GKO_ASSERT_ARRAY_EQ(block_type, dblock_type);
        GKO_ASSERT_ARRAY_EQ(block_argmin_storage, dblock_argmin_storage);

        if (num_blocks > 1) {
            const auto superblock_storage_size = static_cast<gko::size_type>(
                superblock_storage_type::compute_storage_size(num_blocks));
            gko::array<storage_type> superblock_storage{
                this->ref, superblock_storage_size};
            gko::array<storage_type> dsuperblock_storage{
                this->exec, superblock_storage_size};
            superblock_storage.fill(0);
            dsuperblock_storage.fill(0);
            superblock_storage_type superblocks{
                block_min.get_const_data(), superblock_storage.get_data(),
                static_cast<index_type>(num_blocks)};
            superblock_storage_type dsuperblocks{
                dblock_min.get_const_data(), dsuperblock_storage.get_data(),
                static_cast<index_type>(num_blocks)};

            gko::kernels::reference::range_minimum_query::compute_lookup_large(
                this->ref, block_min.get_const_data(),
                static_cast<index_type>(num_blocks), superblocks);
            gko::kernels::GKO_DEVICE_NAMESPACE::range_minimum_query::
                compute_lookup_large(this->exec, dblock_min.get_const_data(),
                                     static_cast<index_type>(num_blocks),
                                     dsuperblocks);

            GKO_ASSERT_ARRAY_EQ(superblock_storage, dsuperblock_storage);
        }
    }
}


// nvcc doesn't like device lambdas inside class member functions

template <typename IndexType>
void run_rmq_device(std::shared_ptr<const gko::EXEC_TYPE> exec,
                    const gko::array<IndexType>& dbegins,
                    const gko::array<IndexType>& dends,
                    const gko::device_range_minimum_query<IndexType>& drmq,
                    gko::array<IndexType>& doutput_min,
                    gko::array<IndexType>& doutput_argmin)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto begins, auto ends, auto rmq, auto output_min,
                      auto output_argmin) {
            const auto begin = begins[i];
            const auto end = ends[i];
            const auto result = rmq.query(begin, end);
            output_min[i] = result.min;
            output_argmin[i] = result.argmin;
        },
        dbegins.get_size(), dbegins, dends, drmq.get(), doutput_min,
        doutput_argmin);
}


TYPED_TEST(RangeMinimumQuery, QueryIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        auto values = this->create_random_values(size);
        const auto [begins, ends] = this->create_random_queries(size);
        gko::array<index_type> output_argmin{this->ref, begins.get_size()};
        gko::array<index_type> output_min{this->ref, begins.get_size()};
        gko::array<index_type> dvalues{this->exec, values};
        gko::array<index_type> dbegins{this->exec, begins};
        gko::array<index_type> dends{this->exec, ends};
        gko::array<index_type> doutput_min{this->exec, begins.get_size()};
        gko::array<index_type> doutput_argmin{this->exec, begins.get_size()};
        gko::device_range_minimum_query<index_type> rmq{std::move(values)};
        gko::device_range_minimum_query<index_type> drmq{std::move(dvalues)};

        for (const auto i :
             gko::irange{static_cast<index_type>(begins.get_size())}) {
            const auto result = rmq.get().query(begins.get_const_data()[i],
                                                ends.get_const_data()[i]);
            output_min.get_data()[i] = result.min;
            output_argmin.get_data()[i] = result.argmin;
        }
        run_rmq_device(this->exec, dbegins, dends, drmq, doutput_min,
                       doutput_argmin);

        GKO_ASSERT_ARRAY_EQ(output_min, doutput_min);
        GKO_ASSERT_ARRAY_EQ(output_argmin, doutput_argmin);
    }
}
