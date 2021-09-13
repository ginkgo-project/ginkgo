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

#include <ginkgo/core/matrix/csr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Csr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Sellp = gko::matrix::Sellp<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Hybrid = gko::matrix::Hybrid<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4,
                          std::make_shared<typename Mtx::load_balance>(2)))
    {
        this->create_mtx(mtx.get());
    }

    void create_mtx(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        auto* s = m->get_srow();
        /*
         * 1   3   2
         * 0   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
        s[0] = 0;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(Csr, gko::test::ValueIndexTypes);


TYPED_TEST(Csr, CanGetSubmatrix)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    /* this->mtx
     * 1   3   2
     * 0   5   0
     */
    auto sub_mat =
        this->mtx->create_submatrix(gko::span(0, 2), gko::span(0, 2));
    auto ref =
        gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{0.0, 5.0}}, this->exec);

    GKO_ASSERT_MTX_NEAR(sub_mat.get(), ref.get(), 0.0);
}


TYPED_TEST(Csr, CanGetSubmatrix2)
{
    using Vec = typename TestFixture::Vec;
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mat = gko::initialize<Mtx>(
        {
            // clang-format off
            I<T>{1.0, 3.0, 4.5, 0.0, 2.0}, // 0
            I<T>{1.0, 0.0, 4.5, 7.5, 3.0}, // 1
            I<T>{0.0, 3.0, 4.5, 0.0, 2.0}, // 2
            I<T>{0.0,-1.0, 2.5, 0.0, 2.0}, // 3
            I<T>{1.0, 0.0,-1.0, 3.5, 1.0}, // 4
            I<T>{0.0, 1.0, 0.0, 0.0, 2.0}, // 5
            I<T>{0.0, 3.0, 0.0, 7.5, 1.0}  // 6
                                           // clang-format on
        },
        this->exec);
    ASSERT_EQ(mat->get_num_stored_elements(), 23);
    {
        auto sub_mat1 = mat->create_submatrix(gko::span(0, 2), gko::span(0, 2));
        auto ref1 =
            gko::initialize<Mtx>({I<T>{1.0, 3.0}, I<T>{1.0, 0.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat1.get(), ref1.get(), 0.0);
    }
    {
        auto sub_mat2 = mat->create_submatrix(gko::span(2, 4), gko::span(0, 2));
        auto ref2 =
            gko::initialize<Mtx>({I<T>{0.0, 3.0}, I<T>{0.0, -1.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat2.get(), ref2.get(), 0.0);
    }
    {
        auto sub_mat3 = mat->create_submatrix(gko::span(0, 2), gko::span(3, 5));
        auto ref3 =
            gko::initialize<Mtx>({I<T>{0.0, 2.0}, I<T>{7.5, 3.0}}, this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat3.get(), ref3.get(), 0.0);
    }
    {
        auto sub_mat4 = mat->create_submatrix(gko::span(1, 6), gko::span(2, 4));
        /*
           4.5, 7.5
           4.5, 0.0
           2.5, 0.0
           1.0, 3.5
           0.0, 0.0
        */
        auto ref4 = gko::initialize<Mtx>(
            {I<T>{4.5, 7.5}, I<T>{4.5, 0.0}, I<T>{2.5, 0.0}, I<T>{-1.0, 3.5},
             I<T>{0.0, 0.0}},
            this->exec);

        GKO_EXPECT_MTX_NEAR(sub_mat4.get(), ref4.get(), 0.0);
    }
}


}  // namespace
