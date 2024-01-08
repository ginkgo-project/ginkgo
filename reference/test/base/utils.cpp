// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/utils.hpp"


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace {


class ConvertToWithSorting : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;

    ConvertToWithSorting()
        : ref{gko::ReferenceExecutor::create()},
          mtx{gko::initialize<Dense>({{1, 2, 3}, {6, 0, 7}, {-1, 8, 0}}, ref)},
          unsorted_coo{Coo::create(ref, gko::dim<2>{3, 3},
                                   I<value_type>{1, 3, 2, 7, 6, -1, 8},
                                   I<index_type>{0, 2, 1, 2, 0, 0, 1},
                                   I<index_type>{0, 0, 0, 1, 1, 2, 2})},
          unsorted_csr{Csr::create(
              ref, gko::dim<2>{3, 3}, I<value_type>{1, 3, 2, 7, 6, -1, 8},
              I<index_type>{0, 2, 1, 2, 0, 0, 1}, I<index_type>{0, 3, 5, 7})}

    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::unique_ptr<Dense> mtx;
    std::unique_ptr<Coo> unsorted_coo;
    std::unique_ptr<Csr> unsorted_csr;
};


TEST_F(ConvertToWithSorting, SortWithUniquePtr)
{
    auto result = gko::convert_to_with_sorting<Csr>(ref, unsorted_coo, false);

    ASSERT_TRUE(result->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, DontSortWithUniquePtr)
{
    auto result = gko::convert_to_with_sorting<Csr>(ref, unsorted_csr, true);

    ASSERT_EQ(result.get(), unsorted_csr.get());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, SortWithSharedPtr)
{
    std::shared_ptr<Csr> shared = gko::share(unsorted_csr->clone());

    auto result = gko::convert_to_with_sorting<Csr>(ref, shared, false);

    ASSERT_TRUE(result->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, DontSortWithSharedPtr)
{
    std::shared_ptr<Csr> shared = gko::share(unsorted_csr->clone());

    auto result = gko::convert_to_with_sorting<Csr>(ref, shared, true);

    ASSERT_EQ(result.get(), shared.get());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, SortWithSharedConstPtr)
{
    std::shared_ptr<const Coo> shared = gko::share(unsorted_coo->clone());

    auto result = gko::convert_to_with_sorting<Csr>(ref, shared, false);

    ASSERT_TRUE(result->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, DontSortWithSharedConstPtr)
{
    std::shared_ptr<const Coo> shared = gko::share(unsorted_coo->clone());

    auto result = gko::convert_to_with_sorting<Csr>(ref, shared, true);

    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, SortWithRawPtr)
{
    auto result =
        gko::convert_to_with_sorting<Csr>(ref, unsorted_coo.get(), false);

    ASSERT_TRUE(result->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, DontSortWithRawPtr)
{
    auto result =
        gko::convert_to_with_sorting<Csr>(ref, unsorted_coo.get(), true);

    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, SortWithConstRawPtr)
{
    const Coo* cptr = unsorted_coo.get();

    auto result = gko::convert_to_with_sorting<Csr>(ref, cptr, false);

    ASSERT_TRUE(result->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


TEST_F(ConvertToWithSorting, DontSortWithConstRawPtr)
{
    const auto cptr = mtx.get();

    auto result = gko::convert_to_with_sorting<Csr>(ref, cptr, true);

    GKO_ASSERT_MTX_NEAR(result, mtx, 0.);
}


}  // namespace
