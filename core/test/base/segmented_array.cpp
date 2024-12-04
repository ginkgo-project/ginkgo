// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/segmented_array.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/segmented_array.hpp>

#include "core/test/utils.hpp"


template <typename T>
gko::array<T> get_flat_array(gko::segmented_array<T>& arr)
{
    return gko::make_array_view(arr.get_executor(), arr.get_size(),
                                arr.get_flat_data());
}


template <typename T>
class SegmentedArray : public ::testing::Test {
public:
    using value_type = T;

    std::shared_ptr<gko::Executor> exec = gko::ReferenceExecutor::create();
};

TYPED_TEST_SUITE(SegmentedArray, gko::test::PODTypes, TypenameNameGenerator);


TYPED_TEST(SegmentedArray, CanConstructFromExecutor)
{
    using value_type = typename TestFixture::value_type;

    gko::segmented_array<value_type> arr{this->exec};

    ASSERT_EQ(arr.get_executor(), this->exec);
    ASSERT_EQ(arr.get_segment_count(), 0);
    ASSERT_EQ(arr.get_size(), 0);
    ASSERT_EQ(arr.get_flat_data(), nullptr);
    ASSERT_NE(arr.get_offsets().get_const_data(), nullptr);
}


TYPED_TEST(SegmentedArray, CanConstructFromSizes)
{
    using value_type = typename TestFixture::value_type;

    auto arr = gko::segmented_array<value_type>::create_from_sizes(
        {this->exec, {1, 2, 3}});

    auto expected_offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});
    ASSERT_EQ(arr.get_executor(), this->exec);
    ASSERT_EQ(arr.get_segment_count(), 3);
    ASSERT_EQ(arr.get_size(), 6);
    ASSERT_NE(arr.get_flat_data(), nullptr);
    GKO_ASSERT_ARRAY_EQ(arr.get_offsets(), expected_offsets);
}


TYPED_TEST(SegmentedArray, CanConstructFromBufferAndSizes)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});

    auto arr = gko::segmented_array<value_type>::create_from_sizes(
        buffer, {this->exec, {1, 2, 3}});

    auto expected_offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});
    ASSERT_EQ(arr.get_executor(), this->exec);
    ASSERT_EQ(arr.get_segment_count(), 3);
    ASSERT_EQ(arr.get_size(), 6);
    GKO_ASSERT_ARRAY_EQ(arr.get_offsets(), expected_offsets);
    GKO_ASSERT_ARRAY_EQ(get_flat_array(arr), buffer);
}


TYPED_TEST(SegmentedArray, CanConstructFromOffsets)
{
    using value_type = typename TestFixture::value_type;
    auto offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});

    auto arr = gko::segmented_array<value_type>::create_from_offsets(offsets);

    ASSERT_EQ(arr.get_executor(), this->exec);
    ASSERT_EQ(arr.get_segment_count(), 3);
    ASSERT_EQ(arr.get_size(), 6);
    ASSERT_NE(arr.get_flat_data(), nullptr);
    GKO_ASSERT_ARRAY_EQ(arr.get_offsets(), offsets);
}


TYPED_TEST(SegmentedArray, CanConstructFromBufferAndOffsets)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});
    auto offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});

    auto arr =
        gko::segmented_array<value_type>::create_from_offsets(buffer, offsets);

    ASSERT_EQ(arr.get_executor(), this->exec);
    ASSERT_EQ(arr.get_segment_count(), 3);
    ASSERT_EQ(arr.get_size(), 6);
    GKO_ASSERT_ARRAY_EQ(arr.get_offsets(), offsets);
    GKO_ASSERT_ARRAY_EQ(get_flat_array(arr), buffer);
}


TYPED_TEST(SegmentedArray, CanBeCopied)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});
    auto offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});
    auto arr =
        gko::segmented_array<value_type>::create_from_offsets(buffer, offsets);

    auto copy = arr;

    GKO_ASSERT_ARRAY_EQ(copy.get_offsets(), arr.get_offsets());
    GKO_ASSERT_ARRAY_EQ(get_flat_array(copy), get_flat_array(arr));
}


TYPED_TEST(SegmentedArray, CanBeMoved)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});
    auto offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 6});
    auto arr =
        gko::segmented_array<value_type>::create_from_offsets(buffer, offsets);

    auto move = std::move(arr);

    GKO_ASSERT_ARRAY_EQ(move.get_offsets(), offsets);
    GKO_ASSERT_ARRAY_EQ(get_flat_array(move), buffer);
    ASSERT_EQ(arr.get_size(), 0);
    ASSERT_EQ(arr.get_segment_count(), 0);
    ASSERT_EQ(arr.get_flat_data(), nullptr);
    ASSERT_NE(arr.get_offsets().get_const_data(), nullptr);
}


TYPED_TEST(SegmentedArray, ThrowsIfBufferSizeDoesntMatchSizes)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});

    auto construct_with_size_mismatch = [&] {
        auto arr = gko::segmented_array<value_type>::create_from_offsets(
            buffer, {this->exec, {1, 2, 1}});
    };
    ASSERT_THROW(construct_with_size_mismatch(), gko::ValueMismatch);
}


TYPED_TEST(SegmentedArray, ThrowsIfBufferSizeDoesntMatchOffsets)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = gko::array<value_type>(this->exec, {1, 2, 2, 3, 3, 3});
    auto offsets = gko::array<gko::int64>(this->exec, {0, 1, 3, 4});

    auto construct_with_size_mismatch = [&] {
        auto arr = gko::segmented_array<value_type>::create_from_offsets(
            buffer, offsets);
    };
    ASSERT_THROW(construct_with_size_mismatch(), gko::ValueMismatch);
}


TYPED_TEST(SegmentedArray, ThrowsOnEmptyOffsets)
{
    using value_type = typename TestFixture::value_type;
    auto offsets = gko::array<gko::int64>(this->exec);

    auto construct_with_empty_offsets = [&] {
        auto arr =
            gko::segmented_array<value_type>::create_from_offsets(offsets);
    };
    ASSERT_THROW(construct_with_empty_offsets(), gko::InvalidStateError);
}
