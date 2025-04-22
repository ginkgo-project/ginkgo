// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include "core/components/bitvector.hpp"

#include <cassert>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/components/bitvector.hpp"
#include "core/base/index_range.hpp"
#include "core/test/utils.hpp"
#include "reference/components/bitvector.hpp"
#include "test/utils/common_fixture.hpp"


// workaround for cudafe 11.0 bug
using gko::irange;


template <typename T>
class Bitvector : public CommonTestFixture {
protected:
    using index_type = T;
    using bitvector = gko::bitvector<index_type>;
    using device_bitvector = gko::device_bitvector<index_type>;
    using storage_type = typename bitvector::storage_type;
    constexpr static auto block_size = bitvector::block_size;

    Bitvector()
        : rng{67193}, sizes{0,    1,    2,    16,    31,    32,  33,
                            40,   63,   64,   65,    127,   128, 129,
                            1000, 1024, 2000, 10000, 100000}
    {}

    gko::array<index_type> create_random_values(index_type num_values,
                                                index_type size)
    {
        assert(num_values <= size);
        std::vector<index_type> values(size);
        std::iota(values.begin(), values.end(), index_type{});
        std::shuffle(values.begin(), values.end(), rng);
        values.resize(num_values);
        std::sort(values.begin(), values.end());
        return gko::array<index_type>{this->ref, values.begin(), values.end()};
    }

    void assert_bitvector_equal(const bitvector& bv, const bitvector& dbv)
    {
        ASSERT_EQ(bv.get_size(), dbv.get_size());
        const auto num_blocks =
            static_cast<gko::size_type>(bv.get_num_blocks());
        const auto bits =
            gko::detail::array_const_cast(gko::make_const_array_view(
                bv.get_executor(), num_blocks, bv.get_bits()));
        const auto dbits =
            gko::detail::array_const_cast(gko::make_const_array_view(
                dbv.get_executor(), num_blocks, dbv.get_bits()));
        const auto ranks =
            gko::detail::array_const_cast(gko::make_const_array_view(
                bv.get_executor(), num_blocks, bv.get_ranks()));
        const auto dranks =
            gko::detail::array_const_cast(gko::make_const_array_view(
                dbv.get_executor(), num_blocks, dbv.get_ranks()));
        GKO_ASSERT_ARRAY_EQ(bits, dbits);
        GKO_ASSERT_ARRAY_EQ(ranks, dranks);
    }

    std::default_random_engine rng;
    std::vector<index_type> sizes;
};

TYPED_TEST_SUITE(Bitvector, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(Bitvector, BuildFromIndicesIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using bitvector = typename TestFixture::bitvector;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        for (auto num_values :
             {index_type{}, size / 10, size / 4, size / 2, size}) {
            SCOPED_TRACE(num_values);
            auto values = this->create_random_values(num_values, size);
            gko::array<index_type> dvalues{this->exec, values};

            auto bv = gko::kernels::reference::bitvector::from_sorted_indices(
                this->ref, values.get_data(), values.get_size(), size);
            auto dbv = gko::kernels::GKO_DEVICE_NAMESPACE::bitvector::
                from_sorted_indices(this->exec, dvalues.get_data(),
                                    dvalues.get_size(), size);

            this->assert_bitvector_equal(bv, dbv);
        }
    }
}


// nvcc doesn't like device lambdas inside class member functions
template <typename IndexType>
std::pair<gko::bitvector<IndexType>, gko::bitvector<IndexType>> run_predicate(
    std::shared_ptr<const gko::ReferenceExecutor> ref,
    std::shared_ptr<const gko::EXEC_TYPE> exec, IndexType size, int stride)
{
    return std::make_pair(
        gko::kernels::reference::bitvector::from_predicate(
            ref, size, [stride](int i) { return i % stride == 0; }),
        gko::kernels::GKO_DEVICE_NAMESPACE::bitvector::from_predicate(
            exec, size,
            [stride] GKO_KERNEL(int i) { return i % stride == 0; }));
}


TYPED_TEST(Bitvector, BuildFromPredicateIsEquivalentToFromIndices)
{
    using index_type = typename TestFixture::index_type;
    using bitvector = typename TestFixture::bitvector;
    for (auto size : this->sizes) {
        SCOPED_TRACE(size);
        for (auto stride : {1, 2, 3, 4, 5, 65}) {
            SCOPED_TRACE(stride);
            std::vector<index_type> indices;
            for (index_type i = 0; i < size; i += stride) {
                indices.push_back(i);
            }
            gko::array<index_type> values{this->ref, indices.begin(),
                                          indices.end()};

            auto [bv, dbv] = run_predicate(this->ref, this->exec, size, stride);

            auto ref_bv =
                gko::kernels::reference::bitvector::from_sorted_indices(
                    this->ref, values.get_data(), values.get_size(), size);
            this->assert_bitvector_equal(bv, dbv);
            this->assert_bitvector_equal(ref_bv, dbv);
        }
    }
}


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

            auto bv = gko::kernels::reference::bitvector::from_sorted_indices(
                this->ref, values.get_const_data(), values.get_size(), size);
            auto dbv = gko::kernels::GKO_DEVICE_NAMESPACE::bitvector::
                from_sorted_indices(this->exec, dvalues.get_const_data(),
                                    dvalues.get_size(), size);

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

            GKO_ASSERT_ARRAY_EQ(doutput_bools, output_bools);
            GKO_ASSERT_ARRAY_EQ(doutput_ranks, output_ranks);
        }
    }
}
