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

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SparsityCsr : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::SparsityCsr<v_type, i_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;
    using DenseMtx = gko::matrix::Dense<v_type>;

    SparsityCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {
        i_type *c = mtx->get_col_idxs();
        i_type *r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_CASE(SparsityCsr, gko::test::ValueIndexTypes);


TYPED_TEST(SparsityCsr, CanBeCreatedFromExistingCsrMatrix)
{
    using Csr = typename TestFixture::Csr;
    using DenseMtx = typename TestFixture::DenseMtx;
    using Mtx = typename TestFixture::Mtx;
    auto csr_mtx = gko::initialize<Csr>(
        {{2.0, 3.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, -3.0}}, this->exec);
    auto comp_mtx = gko::initialize<DenseMtx>(
        {{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}, this->exec);

    auto mtx = Mtx::create(this->exec, std::move(csr_mtx));

    GKO_ASSERT_MTX_NEAR(comp_mtx.get(), mtx.get(), 0.0);
}


TYPED_TEST(SparsityCsr, CanBeCreatedFromExistingDenseMatrix)
{
    using DenseMtx = typename TestFixture::DenseMtx;
    using Mtx = typename TestFixture::Mtx;
    auto dense_mtx = gko::initialize<DenseMtx>(
        {{2.0, 3.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, -3.0}}, this->exec);
    auto comp_mtx = gko::initialize<DenseMtx>(
        {{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}, this->exec);

    auto mtx = Mtx::create(this->exec, std::move(dense_mtx));

    GKO_ASSERT_MTX_NEAR(comp_mtx.get(), mtx.get(), 0.0);
}


}  // namespace
