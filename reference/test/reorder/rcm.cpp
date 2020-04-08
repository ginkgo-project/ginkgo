/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/reorder/rcm.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <iostream>
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"

namespace {


class Rcm : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          rcm_factory(reorder_type::build().on(exec)),
          // clang-format off
          p_mtx(gko::initialize<CsrMtx>(
                                        {{1.0, 2.0, 0.0, -1.3, 2.1},
                                         {2.0, 5.0, 1.5, 0.0, 0.0},
                                         {0.0, 1.5, 1.5, 1.1, 0.0},
                                         {-1.3, 0.0, 1.1, 2.0, 0.0},
                                         {2.1, 0.0, 0.0, 0.0, 1.0}},
                                        exec)),
          // clang-format on
          reorder_op(rcm_factory->generate(p_mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> p_mtx;
    std::unique_ptr<reorder_type::Factory> rcm_factory;
    std::unique_ptr<reorder_type> reorder_op;
};


TEST_F(Rcm, FactoryCreatesCorrectReorderOp)
{
    auto adj_mtx = reorder_op->get_adjacency_matrix();
    auto tmp = gko::matrix::SparsityCsr<v_type, i_type>::create(exec, p_mtx);
    auto comp_mtx = tmp->to_adjacency_matrix();

    ASSERT_NE(reorder_op->get_adjacency_matrix(), nullptr);
    GKO_ASSERT_MTX_NEAR(adj_mtx.get(), comp_mtx.get(), 0);
}


TEST_F(Rcm, CanBeCleared)
{
    reorder_op->clear();

    auto reorder_op_mtx = reorder_op->get_adjacency_matrix();
    ASSERT_EQ(reorder_op_mtx, nullptr);
}


}  // namespace
