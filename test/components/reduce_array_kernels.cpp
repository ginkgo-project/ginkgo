// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/reduce_array_kernels.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


template <typename T>
class ReduceArray : public CommonTestFixture {
protected:
    using value_type = T;
    ReduceArray()
        : total_size(6355),
          out{ref, I<T>{2}},
          dout{exec, out},
          vals{ref, total_size},
          dvals{exec}
    {
        std::fill_n(vals.get_data(), total_size, 3);
        dvals = vals;
    }

    gko::size_type total_size;
    gko::array<value_type> out;
    gko::array<value_type> dout;
    gko::array<value_type> vals;
    gko::array<value_type> dvals;
};

TYPED_TEST_SUITE(ReduceArray, gko::test::ValueAndIndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(ReduceArray, EqualsReference)
{
    gko::kernels::reference::components::reduce_add_array(this->ref, this->vals,
                                                          this->out);
    gko::kernels::EXEC_NAMESPACE::components::reduce_add_array(
        this->exec, this->dvals, this->dout);

    GKO_ASSERT_ARRAY_EQ(this->out, this->dout);
}
