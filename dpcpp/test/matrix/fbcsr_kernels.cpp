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


#include <ginkgo/core/base/executor.hpp>


#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
#if GINKGO_SYCL_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif  // GINKGO_SYCL_SINGLE_MODE
    using Mtx = gko::matrix::Fbcsr<vtype>;

    void SetUp()
    {
        ASSERT_GT(gko::SyclExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        sycl = gko::SyclExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (sycl != nullptr) {
            ASSERT_NO_THROW(sycl->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::SyclExecutor> sycl;

    std::unique_ptr<Mtx> mtx;
};


TEST_F(Fbcsr, CanWriteFromMatrixOnDevice)
{
    using value_type = Mtx::value_type;
    using index_type = Mtx::index_type;
    using MatData = gko::matrix_data<value_type, index_type>;
    gko::testing::FbcsrSample<value_type, index_type> sample(ref);
    auto refmat = sample.generate_fbcsr();
    auto syclmat = gko::clone(sycl, refmat);
    MatData refdata;
    MatData sycldata;

    refmat->write(refdata);
    syclmat->write(sycldata);

    ASSERT_TRUE(refdata.nonzeros == sycldata.nonzeros);
}


}  // namespace
