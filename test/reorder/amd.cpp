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

#include <algorithm>
#include <initializer_list>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/reorder/amd.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class Amd : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using reorder_type = gko::experimental::reorder::Amd<index_type>;

    Amd()
    {
        std::ifstream stream{gko::matrices::location_ani4_mtx};
        mtx = gko::read<matrix_type>(stream, ref);
        dmtx = gko::clone(exec, mtx);
    }

    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
};

TYPED_TEST_SUITE(Amd, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Amd, IsEquivalentToRef)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory = reorder_type::build().on(this->exec);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}
