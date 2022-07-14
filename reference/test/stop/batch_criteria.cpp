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

#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace {


class SimpleRes : public ::testing::Test {
protected:
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;

    SimpleRes() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    const int nrows = 100;
    const real_type tol = 1e-5;

    void check_helper(const bool relative, const bool check_converged)
    {
        gko::array<real_type> h_resnorms(this->exec, 1);
        gko::array<real_type> h_bnorms(this->exec, 1);
        if (check_converged) {
            h_bnorms.get_data()[0] = 1.0e6;
            if (relative) {
                h_resnorms.get_data()[0] = h_bnorms.get_data()[0] * tol / 10;
            } else {
                h_resnorms.get_data()[0] = tol / 10.0;
            }
        } else {
            h_bnorms.get_data()[0] = 1.0e-6;
            if (relative) {
                h_resnorms.get_data()[0] = 5 * tol * h_bnorms.get_data()[0];
            } else {
                h_resnorms.get_data()[0] = 5 * tol;
            }
        }

        bool all_conv = false;

        if (relative) {
            using BatchStop =
                gko::kernels::host::stop::SimpleRelResidual<value_type>;
            BatchStop bstop(tol, h_bnorms.get_const_data());
            all_conv = bstop.check_converged(h_resnorms.get_const_data());
        } else {
            using BatchStop =
                gko::kernels::host::stop::SimpleAbsResidual<value_type>;
            BatchStop bstop(tol, h_bnorms.get_const_data());
            all_conv = bstop.check_converged(h_resnorms.get_const_data());
        }

        if (check_converged) {
            ASSERT_TRUE(all_conv);
        } else {
            ASSERT_FALSE(all_conv);
        }
    }
};


TEST_F(SimpleRes, RelDetectsConvergence) { check_helper(true, true); }

TEST_F(SimpleRes, RelDetectsDivergence) { check_helper(true, false); }

TEST_F(SimpleRes, AbsDetectsConvergence) { check_helper(false, true); }

TEST_F(SimpleRes, AbsDetectsDivergence) { check_helper(false, false); }


}  // namespace
