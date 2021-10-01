/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
                                                               modification, are
permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
                documentation and/or other materials provided with the
distribution.

           3. Neither the name of the copyright holder nor the names of its
               contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
            HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
                                                                LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
            ******************************<GINKGO
LICENSE>*******************************/

#include "core/constraints/constraints_handler_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {
template <typename ValueIndexType>
class ConstrainedSystem : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using dense = gko::matrix::Dense<value_type>;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    using handler =
        gko::constraints::ConstraintsHandler<value_type, index_type>;

    ConstrainedSystem() : ref(gko::ReferenceExecutor::create()) {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ConstrainedSystem, gko::test::ValueIndexTypes);

TYPED_TEST(ConstrainedSystem, CanCreateWithIdxsMatrix)
{
    using index_type = typename TestFixture::index_type;
    using mtx = typename TestFixture::mtx;
    using handler = typename TestFixture::handler;
    gko::Array<index_type> idxs{this->ref};
    auto csr = gko::share(mtx::create(this->ref));

    handler ch(idxs, csr);

    GKO_ASSERT_ARRAY_EQ(*ch.get_constrained_indices(), idxs);
    ASSERT_EQ(ch.get_orig_operator(), csr.get());
}


}  // namespace
