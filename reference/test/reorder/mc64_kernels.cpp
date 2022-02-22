/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/reorder/mc64_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class Mc64 : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Mc64()
        : ref(gko::ReferenceExecutor::create()),
          tmp{ref},
          mtx(gko::initialize<matrix_type>({{1., 2., 0., 0., 3., 0.},
                                            {5., 1., 0., 0., 0., 0.},
                                            {0., 0., 0., 6., 0., 4.},
                                            {0., 0., 4., 0., 0., 3.},
                                            {0., 0., 0., 4., 2., 0.},
                                            {0., 5., 8., 0., 0., 0.}},
                                           ref)),
          expected_workspace{
              ref,
              I<real_type>({2., 1., 0., 0., 4., 0., 2., 0., 1., 0., 2., 3., 0.,
                            0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.})},
          expected_perm{ref, I<index_type>({1, 0, 3, 5, -1, 2})},
          expected_inv_perm{ref, I<index_type>({1, 0, 5, 2, -1, 3})}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<real_type> tmp;
    std::shared_ptr<matrix_type> mtx;
    gko::Array<real_type> expected_workspace;
    gko::Array<index_type> expected_perm;
    gko::Array<index_type> expected_inv_perm;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, InitializeWeightsExample)
{
    using matrix_type = typename TestFixture::matrix_type;
    using real_type = typename TestFixture::real_type;

    gko::kernels::reference::mc64::initialize_weights(
        this->ref, this->mtx.get(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(this->tmp, this->expected_workspace);
}


TYPED_TEST(Mc64, InitialMatchingExample)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> p{this->ref,
                             I<index_type>({-1, -1, -1, -1, -1, -1})};
    gko::Array<index_type> ip{this->ref,
                              I<index_type>({-1, -1, -1, -1, -1, -1})};

    gko::kernels::reference::mc64::initial_matching(
        this->ref, this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->expected_workspace, p, ip);

    GKO_ASSERT_ARRAY_EQ(p, this->expected_perm);
    GKO_ASSERT_ARRAY_EQ(ip, this->expected_inv_perm);
}


}  // namespace
