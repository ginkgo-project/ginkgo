// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/fill_array_kernels.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class FillArray : public ::testing::Test {
protected:
    using value_type = T;
    FillArray()
        : ref(gko::ReferenceExecutor::create()),
          total_size(6344),
          expected(ref, total_size),
          vals(ref, total_size),
          seqs(ref, total_size)
    {
        std::fill_n(expected.get_data(), total_size, T(6453));
        std::iota(seqs.get_data(), seqs.get_data() + total_size, 0);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::size_type total_size;
    gko::array<value_type> expected;
    gko::array<value_type> vals;
    gko::array<value_type> seqs;
};

TYPED_TEST_SUITE(FillArray, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FillArray, EqualsReference)
{
    using T = typename TestFixture::value_type;
    gko::kernels::reference::components::fill_array(
        this->ref, this->vals.get_data(), this->total_size, T(6453));

    GKO_ASSERT_ARRAY_EQ(this->vals, this->expected);
}


TYPED_TEST(FillArray, FillSeqEqualsReference)
{
    using T = typename TestFixture::value_type;
    gko::kernels::reference::components::fill_seq_array(
        this->ref, this->vals.get_data(), this->total_size);

    GKO_ASSERT_ARRAY_EQ(this->vals, this->seqs);
}


}  // namespace
