// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/vector_collection.hpp"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


class VectorCollection : public ::testing::Test {
protected:
    using value_type = double;
    using Vec = gko::matrix::MultiVector<value_type>;

    VectorCollection()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Vec>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, exec)),
          vectors{gko::initialize<Vec>({{1.0, 2.0}, {3.0, 4.0}}, exec),
                  gko::initialize<Vec>({{5.0, 6.0}, {7.0, 8.0}}, exec),
                  gko::initialize<Vec>({{9.0, 10.0}, {11.0, 12.0}}, exec)}
    {}

    template <typename T>
    std::unique_ptr<gko::VectorCollection> create_collection(T&& vectors)
    {
        return gko::VectorCollection::create(vectors.begin(), vectors.end());
    }

    static void assert_equal(const gko::AbstractMultiVector* result,
                             const gko::AbstractMultiVector* expected)
    {
        auto result_vec = dynamic_cast<const Vec*>(result);
        auto expected_vec = dynamic_cast<const Vec*>(expected);
        ASSERT_NE(result_vec, nullptr);
        ASSERT_NE(expected_vec, nullptr);
        GKO_EXPECT_MTX_NEAR(result_vec->as_const_dense_view(),
                            expected_vec->as_const_dense_view(), 0.0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Vec> mtx;
    std::vector<std::shared_ptr<const Vec>> vectors;
};


TEST_F(VectorCollection, CanCreateFromBlueprint)
{
    auto collection = gko::VectorCollection::create_with_type_of(
        mtx, exec, 4, gko::dim<2>{5, 2}, gko::dim<2>{5, 2});

    ASSERT_EQ(collection->get_executor(), exec);
    ASSERT_EQ(collection->get_size(), 4);
    ASSERT_EQ(collection->get_element_size(), gko::dim<2>(5, 2));
    for (gko::size_type i = 0; i < 4; ++i) {
        auto element = collection->at(i);
        ASSERT_NE(dynamic_cast<Vec*>(element.get()), nullptr);
        ASSERT_EQ(element->get_size(), gko::dim<2>(5, 2));
        ASSERT_EQ(element->get_local_size(), gko::dim<2>(5, 2));
        ASSERT_EQ(element->get_executor(), exec);
    }
}


TEST_F(VectorCollection, CanCreateFromMultiVector)
{
    auto collection = gko::VectorCollection::create(mtx);

    ASSERT_EQ(collection->get_size(), 3);
    ASSERT_EQ(collection->get_element_size(), gko::dim<2>(2, 1));
    for (gko::size_type i = 0; i < 3; ++i) {
        auto expected =
            mtx->create_subview(gko::local_span(0, 2), gko::local_span(i));
        assert_equal(collection->at(i).get(), expected.get());
    }
}


TEST_F(VectorCollection, CanCreateFromVectorOfVectors)
{
    auto collection = create_collection(vectors);

    ASSERT_EQ(collection->get_executor(), exec);
    ASSERT_EQ(collection->get_size(), 3);
    ASSERT_EQ(collection->get_element_size(), gko::dim<2>(2, 2));
    for (gko::size_type i = 0; i < 3; ++i) {
        assert_equal(collection->at(i).get(), vectors[i].get());
    }
}


TEST_F(VectorCollection, CreateFromEmptyRangeThrows)
{
    std::vector<std::shared_ptr<const gko::AbstractMultiVector>> empty;

    ASSERT_THROW(gko::VectorCollection::create(empty.begin(), empty.end()),
                 gko::InvalidStateError);
}


TEST_F(VectorCollection, CreateFromDifferentSizesThrows)
{
    vectors.push_back(gko::initialize<Vec>({{1.0, 2.0, 3.0}}, exec));

    ASSERT_THROW(create_collection(vectors), gko::InvalidStateError);
}


TEST_F(VectorCollection, CreateFromDifferentTypesThrows)
{
    std::vector<std::shared_ptr<const gko::AbstractMultiVector>>
        generic_vectors(vectors.begin(), vectors.end());
    generic_vectors.push_back(gko::initialize<gko::matrix::MultiVector<float>>(
        {{1.0, 2.0}, {3.0, 4.0}}, exec));

    ASSERT_THROW(create_collection(vectors), gko::InvalidStateError);
}


TEST_F(VectorCollection, CanAccessVector)
{
    auto collection = create_collection(vectors);

    auto element = collection->at(1);
    auto element_vec = dynamic_cast<Vec*>(element.get());
    ASSERT_NE(element_vec, nullptr);
    element_vec->at(1, 0) = -1.0;

    auto modified = collection->at(1);
    auto modified_vec = dynamic_cast<Vec*>(modified.get());
    ASSERT_NE(modified_vec, nullptr);
    ASSERT_EQ(modified_vec->get_size(), gko::dim<2>(2, 2));
    EXPECT_EQ(modified_vec->at(0, 0), 5.0);
    EXPECT_EQ(modified_vec->at(0, 1), 6.0);
    EXPECT_EQ(modified_vec->at(1, 0), -1.0);
    EXPECT_EQ(modified_vec->at(1, 1), 8.0);
    assert_equal(collection->at(0).get(), vectors[0].get());
    assert_equal(collection->at(2).get(), vectors[2].get());
}


TEST_F(VectorCollection, CanAccessVectorConst)
{
    std::unique_ptr<const gko::VectorCollection> collection =
        create_collection(vectors);

    for (gko::size_type i = 0; i < 3; ++i) {
        std::unique_ptr<const gko::AbstractMultiVector> element =
            collection->at(i);
        assert_equal(element.get(), vectors[i].get());
    }
}


TEST_F(VectorCollection, CanGetDeviceView)
{
    auto collection = create_collection(vectors);

    auto view = collection->get_device_view<value_type>();
    view(2, 0, 1) = -1.0;

    ASSERT_EQ(view.size, 3);
    ASSERT_EQ(view.element_size, gko::dim<2>(2, 2));
    EXPECT_EQ(view(0, 0, 0), 1.0);
    EXPECT_EQ(view(0, 1, 1), 4.0);
    EXPECT_EQ(view(1, 0, 1), 6.0);
    EXPECT_EQ(view(1, 1, 0), 7.0);
    EXPECT_EQ(view(2, 1, 1), 12.0);
    auto element = view.at(2);
    ASSERT_EQ(element.size, gko::dim<2>(2, 2));
    EXPECT_EQ(element(0, 0), 9.0);
    EXPECT_EQ(element(0, 1), -1.0);
    EXPECT_EQ(dynamic_cast<Vec*>(collection->at(2).get())->at(0, 1), -1.0);
}


TEST_F(VectorCollection, CanGetDeviceViewConst)
{
    std::unique_ptr<const gko::VectorCollection> collection =
        create_collection(vectors);

    auto view = collection->get_device_view<value_type>();

    static_assert(
        std::is_same_v<decltype(view),
                       gko::matrix::view::vector_collection<const value_type>>);
    ASSERT_EQ(view.size, 3);
    ASSERT_EQ(view.element_size, gko::dim<2>(2, 2));
    for (gko::size_type i = 0; i < 3; ++i) {
        auto expected = dynamic_cast<const Vec*>(vectors[i].get());
        for (gko::size_type row = 0; row < 2; ++row) {
            for (gko::size_type col = 0; col < 2; ++col) {
                EXPECT_EQ(view(i, row, col), expected->at(row, col));
            }
        }
    }
}
