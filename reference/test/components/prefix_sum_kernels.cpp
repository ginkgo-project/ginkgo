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

#include "core/components/prefix_sum_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class PrefixSum : public ::testing::Test {
protected:
    using index_type = T;
    PrefixSum()
        : exec(gko::ReferenceExecutor::create()),
          vals{3, 5, 6, 7, 1, 5, 9, 7, 2, 0, 5},
          expected{0, 3, 8, 14, 21, 22, 27, 36, 43, 45, 45}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::vector<index_type> vals;
    std::vector<index_type> expected;
};

TYPED_TEST_CASE(PrefixSum, gko::test::IndexTypes);


TYPED_TEST(PrefixSum, Works)
{
    gko::kernels::reference::components::prefix_sum(
        this->exec, this->vals.data(), this->vals.size());

    ASSERT_EQ(this->vals, this->expected);
}


}  // namespace
