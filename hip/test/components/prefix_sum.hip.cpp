/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/components/prefix_sum.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "hip/test/utils.hip.hpp"


namespace {


class PrefixSum : public ::testing::Test {
protected:
    using index_type = gko::int32;
    PrefixSum()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::HipExecutor::create(0, ref)),
          rand(293),
          total_size(42793),
          vals(ref, total_size),
          dvals(exec)
    {
        std::uniform_int_distribution<index_type> dist(0, 1000);
        for (gko::size_type i = 0; i < total_size; ++i) {
            vals.get_data()[i] = dist(rand);
        }
        dvals = vals;
    }

    void test(gko::size_type size)
    {
        gko::kernels::reference::components::prefix_sum(ref, vals.get_data(),
                                                        size);
        gko::kernels::hip::components::prefix_sum(exec, dvals.get_data(), size);

        GKO_ASSERT_ARRAY_EQ(vals, dvals);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> exec;
    std::default_random_engine rand;
    gko::size_type total_size;
    gko::Array<index_type> vals;
    gko::Array<index_type> dvals;
};


TEST_F(PrefixSum, SmallEqualsReference) { test(100); }


TEST_F(PrefixSum, BigEqualsReference) { test(total_size); }


}  // namespace
