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


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "reference/log/batch_logger.hpp"


namespace {


template <typename T>
class BatchFinalLogger : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BatchLog =
        gko::kernels::host::batch_log::SimpleFinalLogger<real_type>;

    BatchFinalLogger() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const int nrows = 100;
    const size_t nbatch = 3;
};

TYPED_TEST_SUITE(BatchFinalLogger, gko::test::ValueTypes);


TYPED_TEST(BatchFinalLogger, LogsOneRhsConvergedOneIteration)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    gko::array<real_type> res_norms_log(this->exec, this->nbatch);
    gko::array<int> iters_log(this->exec, this->nbatch);
    for (int i = 0; i < this->nbatch; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 10;
    const int iter = 5;
    BatchLog blog_h(res_norms_log.get_data(), iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        real_type resnv = 1.0;
        blog.log_iteration(ib, iter, resnv);
    }

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(res_norms_log.get_const_data()[i], 1.0);
        ASSERT_EQ(iters_log.get_const_data()[i], iter);
    }
}


}  // namespace
