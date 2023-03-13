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

#include <fstream>


#include <gtest/gtest.h>


#include <ginkgo/core/reorder/nested_dissection.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename IndexType>
class NestedDissection : public CommonTestFixture {
protected:
    using v_type = double;
    using i_type = IndexType;
    using matrix_type = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::NestedDissection<v_type, i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;


    NestedDissection()
        : nd_factory(reorder_type::build().on(ref)),
          dnd_factory(reorder_type::build().on(exec))
    {
        std::ifstream stream{gko::matrices::location_ani1_mtx};
        mtx = gko::read<matrix_type>(stream, ref);
        dmtx = gko::clone(exec, mtx);
    }

    std::unique_ptr<reorder_type> nd_factory;
    std::unique_ptr<reorder_type> dnd_factory;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
};

TYPED_TEST_SUITE(NestedDissection, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(NestedDissection, ResultIsEquivalentToRef)
{
    auto perm = this->nd_factory->generate(this->mtx);
    auto dperm = this->dnd_factory->generate(this->dmtx);
}
