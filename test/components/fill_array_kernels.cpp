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
#include "test/utils/executor.hpp"


template <typename T>
class FillArray : public CommonTestFixture {
protected:
    using value_type = T;
    FillArray()
        : total_size(63531),
          vals{ref, total_size},
          dvals{exec, total_size},
          seqs{ref, total_size}
    {
        std::fill_n(vals.get_data(), total_size, T(1523));
        std::iota(seqs.get_data(), seqs.get_data() + total_size, 0);
    }

    gko::size_type total_size;
    gko::array<value_type> vals;
    gko::array<value_type> dvals;
    gko::array<value_type> seqs;
};

TYPED_TEST_SUITE(FillArray, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FillArray, EqualsReference)
{
    using T = typename TestFixture::value_type;
    gko::kernels::EXEC_NAMESPACE::components::fill_array(
        this->exec, this->dvals.get_data(), this->total_size, T(1523));

    GKO_ASSERT_ARRAY_EQ(this->vals, this->dvals);
}


TYPED_TEST(FillArray, FillSeqEqualsReference)
{
    using T = typename TestFixture::value_type;
    gko::kernels::EXEC_NAMESPACE::components::fill_seq_array(
        this->exec, this->dvals.get_data(), this->total_size);

    GKO_ASSERT_ARRAY_EQ(this->seqs, this->dvals);
}
