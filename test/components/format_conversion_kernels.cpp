// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


template <typename IndexType>
class FormatConversion : public CommonTestFixture {
protected:
    FormatConversion()
        : rand(293),
          size(42793),
          sizes{ref, size},
          ptrs{ref, size + 1},
          idxs{ref}
    {
        std::uniform_int_distribution<int> row_dist{0, 10};
        ptrs.get_data()[0] = 0;
        for (gko::size_type i = 0; i < size; i++) {
            sizes.get_data()[i] = row_dist(rand);
            ptrs.get_data()[i + 1] = ptrs.get_data()[i] + sizes.get_data()[i];
        }
        idxs.resize_and_reset(ptrs.get_const_data()[size]);
        for (gko::size_type i = 0; i < size; i++) {
            auto begin = ptrs.get_const_data()[i];
            auto end = ptrs.get_const_data()[i + 1];
            for (auto j = begin; j < end; j++) {
                idxs.get_data()[j] = i;
            }
        }
        sizes.set_executor(exec);
        ptrs.set_executor(exec);
        idxs.set_executor(exec);
    }

    gko::size_type size;
    std::default_random_engine rand;
    gko::array<gko::size_type> sizes;
    gko::array<IndexType> ptrs;
    gko::array<IndexType> idxs;
};

TYPED_TEST_SUITE(FormatConversion, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FormatConversion, ConvertsEmptyPtrsToIdxs)
{
    gko::array<TypeParam> ptrs(this->exec, this->size + 1);
    ptrs.fill(0);
    TypeParam* output = nullptr;

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_idxs(
        this->exec, ptrs.get_const_data(), this->size, output);

    // mustn't segfault
}


TYPED_TEST(FormatConversion, ConvertPtrsToIdxs)
{
    auto ref_idxs = this->idxs;
    this->idxs.fill(-1);

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_idxs(
        this->exec, this->ptrs.get_const_data(), this->size,
        this->idxs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->idxs, ref_idxs);
}


TYPED_TEST(FormatConversion, ConvertsEmptyIdxsToPtrs)
{
    this->ptrs.fill(0);
    auto ref_ptrs = this->ptrs;
    this->ptrs.fill(-1);
    TypeParam* input = nullptr;

    gko::kernels::EXEC_NAMESPACE::components::convert_idxs_to_ptrs(
        this->exec, input, 0, this->size, this->ptrs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->ptrs, ref_ptrs);
}


TYPED_TEST(FormatConversion, ConvertIdxsToPtrsIsEquivalentToRef)
{
    auto ref_ptrs = this->ptrs;
    this->ptrs.fill(-1);

    gko::kernels::EXEC_NAMESPACE::components::convert_idxs_to_ptrs(
        this->exec, this->idxs.get_const_data(), this->idxs.get_size(),
        this->size, this->ptrs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->ptrs, ref_ptrs);
}


TYPED_TEST(FormatConversion, ConvertPtrsToSizesIsEquivalentToRef)
{
    auto ref_sizes = this->sizes;
    this->sizes.fill(12345);

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_sizes(
        this->exec, this->ptrs.get_const_data(), this->size,
        this->sizes.get_data());

    GKO_ASSERT_ARRAY_EQ(this->sizes, ref_sizes);
}
