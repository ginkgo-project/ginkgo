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

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <cuda_runtime.h>
// force-top: off


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/scoped_device_id.hpp>


#include <gtest/gtest.h>


namespace {


class ScopedDeviceId : public ::testing::Test {
protected:
    ScopedDeviceId()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref))
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
};


TEST_F(ScopedDeviceId, SetsId)
{
    auto new_device_id = std::max(cuda->get_num_devices() - 1, 0);

    gko::detail::cuda_scoped_device_id g{new_device_id};

    ASSERT_EQ(cuda->get_device_id(), new_device_id);
}


TEST_F(ScopedDeviceId, ResetsId)
{
    auto old_device_id = cuda->get_device_id();

    {
        auto new_device_id = std::max(cuda->get_num_devices() - 1, 0);
        gko::detail::cuda_scoped_device_id g{new_device_id};
    }

    ASSERT_EQ(cuda->get_device_id(), old_device_id);
}


}  // namespace
