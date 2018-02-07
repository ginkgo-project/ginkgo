/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/matrix/coo.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/csr.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Coo : public ::testing::Test {
protected:
    using Csr = gko::matrix::Csr<>;
    using Mtx = gko::matrix::Coo<>;
    using Vec = gko::matrix::Dense<>;

    Coo()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, 2, 3, 4))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *r = mtx->get_row_idxs();
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
        r[3] = 1;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
    }

    void assert_equal_to_mtx_in_csr_format(const Csr *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        ASSERT_EQ(m->get_num_rows(), 2);
        ASSERT_EQ(m->get_num_cols(), 3);
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};


TEST_F(Coo, ConvertsToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx->get_executor());
    mtx->convert_to(csr_mtx.get());
    assert_equal_to_mtx_in_csr_format(csr_mtx.get());
}


TEST_F(Coo, MovesToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx->get_executor());
    mtx->move_to(csr_mtx.get());
    assert_equal_to_mtx_in_csr_format(csr_mtx.get());
}


}  // namespace
