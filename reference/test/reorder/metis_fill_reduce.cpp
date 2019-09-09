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

#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


namespace {


class MetisFillReduce : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = long int;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::MetisFillReduce<v_type, i_type>;
    MetisFillReduce()
        : exec(gko::ReferenceExecutor::create()),
          ani4_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_ani4_mtx, std::ios::in),
              exec)),
          metis_fill_reduce_factory(reorder_type::build().on(exec)),
          reorder_op(metis_fill_reduce_factory->generate(ani4_mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> ani4_mtx;
    std::unique_ptr<reorder_type::Factory> metis_fill_reduce_factory;
    std::unique_ptr<reorder_type> reorder_op;
};


TEST_F(MetisFillReduce, FactoryCreatesCorrectReorderOp)
{
    auto sys_mtx = reorder_op->get_system_matrix();

    ASSERT_NE(reorder_op->get_system_matrix(), nullptr);
    GKO_ASSERT_MTX_NEAR(sys_mtx.get(), ani4_mtx.get(), 0);
}


TEST_F(MetisFillReduce, CanBeCleared)
{
    reorder_op->clear();

    auto reorder_op_mtx = reorder_op->get_system_matrix();
    ASSERT_EQ(reorder_op_mtx, nullptr);
}


}  // namespace
