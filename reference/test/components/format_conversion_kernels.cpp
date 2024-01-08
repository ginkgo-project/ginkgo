// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"


#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename IndexType>
class FormatConversion : public ::testing::Test {
protected:
    FormatConversion() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(FormatConversion, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FormatConversion, ConvertsEmptyPtrsToIdxs)
{
    std::vector<TypeParam> ptrs(10);
    TypeParam* out = nullptr;

    gko::kernels::reference::components::convert_ptrs_to_idxs(
        this->ref, ptrs.data(), 9, out);

    // mustn't segfault
}


TYPED_TEST(FormatConversion, ConvertsPtrsToIdxs)
{
    std::vector<TypeParam> ptrs{0, 1, 3, 5, 5};
    std::vector<TypeParam> idxs(5);
    std::vector<TypeParam> reference{0, 1, 1, 2, 2};

    gko::kernels::reference::components::convert_ptrs_to_idxs(
        this->ref, ptrs.data(), 4, idxs.data());

    ASSERT_EQ(idxs, reference);
}


TYPED_TEST(FormatConversion, ConvertsEmptyIdxsToPtrs)
{
    std::vector<TypeParam> idxs;
    std::vector<TypeParam> ptrs(10);
    std::vector<TypeParam> reference(10);

    gko::kernels::reference::components::convert_idxs_to_ptrs(
        this->ref, idxs.data(), 0, 9, ptrs.data());

    ASSERT_EQ(ptrs, reference);
}


TYPED_TEST(FormatConversion, ConvertsIdxsToPtrs)
{
    std::vector<TypeParam> idxs{1, 1, 1, 2, 2, 4};
    std::vector<TypeParam> ptrs(6);
    std::vector<TypeParam> reference{0, 0, 3, 5, 5, 6};

    gko::kernels::reference::components::convert_idxs_to_ptrs(
        this->ref, idxs.data(), 6, 5, ptrs.data());

    ASSERT_EQ(ptrs, reference);
}


TYPED_TEST(FormatConversion, ConvertsPtrsToSizes)
{
    std::vector<TypeParam> ptrs{0, 1, 3, 5, 5};
    std::vector<gko::size_type> sizes(4);
    std::vector<gko::size_type> reference{1, 2, 2, 0};

    gko::kernels::reference::components::convert_ptrs_to_sizes(
        this->ref, ptrs.data(), 4, sizes.data());

    ASSERT_EQ(sizes, reference);
}


}  // namespace
