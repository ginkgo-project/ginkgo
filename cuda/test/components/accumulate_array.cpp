/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/components/accumulate_array.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils/assertions.hpp"
#include "cuda/test/utils.hpp"


namespace {


template <typename T>
class AccumulateArray : public ::testing::Test {
protected:
    using value_type = T;
    AccumulateArray()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::CudaExecutor::create(0, ref)),
          total_size(5),
          vals(ref, {1, 4, 2, 5, 18}),
          sum(ref, {0}),
          dsum(ref, {0}),
          dvals(exec, {1, 4, 2, 5, 18})
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
    gko::size_type total_size;
    gko::Array<value_type> sum;
    gko::Array<value_type> dsum;
    gko::Array<value_type> vals;
    gko::Array<value_type> dvals;
};

TYPED_TEST_SUITE(AccumulateArray, gko::test::ValueAndIndexTypes);


TYPED_TEST(AccumulateArray, EqualsReference)
{
    using T = typename TestFixture::value_type;
    gko::kernels::reference::components::accumulate_array(
        this->ref, this->sum.get_data(), this->vals.get_const_data(),
        this->total_size, T(23));
    gko::kernels::cuda::components::accumulate_array(
        this->exec, this->dsum.get_data(), this->dvals.get_const_data(),
        this->total_size, T(23));

    this->dsum.set_executor(this->ref);
    GKO_ASSERT_ARRAY_EQ(this->sum, this->dsum);
}


}  // namespace
