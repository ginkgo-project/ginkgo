/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/reorder/mc64.hpp>


#include "core/test/utils/assertions.hpp"
#include "test/utils/executor.hpp"


namespace {


class Mc64 : public CommonTestFixture {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::experimental::reorder::Mc64<v_type, i_type>;
    using result_type = gko::Composition<v_type>;
    using perm_type = gko::matrix::ScaledPermutation<v_type, i_type>;

    Mc64()
        : mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, -1.3, 2.1},
                                       {2.0, 5.0, 1.5, 0.0, 0.0},
                                       {0.0, 1.5, 1.5, 1.1, 0.0},
                                       {-1.3, 0.0, 1.1, 2.0, 0.0},
                                       {2.1, 0.0, 0.0, 0.0, 1.0}},
                                      ref)),
          dmtx(mtx->clone(exec)),
          mc64_factory(reorder_type::build().on(ref)),
          dmc64_factory(reorder_type::build().on(exec))
    {}

    std::pair<std::shared_ptr<const perm_type>,
              std::shared_ptr<const perm_type>>
    unpack(const result_type* result)
    {
        GKO_ASSERT_EQ(result->get_operators().size(), 2);
        return std::make_pair(gko::as<perm_type>(result->get_operators()[0]),
                              gko::as<perm_type>(result->get_operators()[1]));
    }

    std::unique_ptr<reorder_type> mc64_factory;
    std::unique_ptr<reorder_type> dmc64_factory;
    std::shared_ptr<CsrMtx> mtx;
    std::shared_ptr<CsrMtx> dmtx;
};


TEST_F(Mc64, IsEquivalentToReference)
{
    auto perm = mc64_factory->generate(mtx);
    auto dperm = dmc64_factory->generate(dmtx);

    auto ops = unpack(perm.get());
    auto dops = unpack(dperm.get());
    GKO_ASSERT_MTX_NEAR(ops.first, dops.first, 0.0);
    GKO_ASSERT_MTX_NEAR(ops.second, dops.second, 0.0);
}


}  // namespace
