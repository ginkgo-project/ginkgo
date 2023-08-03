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

#ifndef GKO_CUDA_TEST_UTILS_HPP_
#define GKO_CUDA_TEST_UTILS_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "core/test/gtest/environments.hpp"
#include "cuda/base/device.hpp"


namespace {


class CudaTestFixture : public ::testing::Test {
protected:
    CudaTestFixture()
        : ref(gko::ReferenceExecutor::create()),
          stream(ResourceEnvironment::cuda_device_id),
          exec(gko::CudaExecutor::create(
              ResourceEnvironment::cuda_device_id, ref,
              std::make_shared<gko::CudaAllocator>(), stream.get())),
          guard(exec->get_scoped_device_id_guard())
    {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            exec->synchronize();
        }
    }

    gko::cuda_stream stream;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
    gko::scoped_device_id_guard guard;
};


}  // namespace


#endif  // GKO_CUDA_TEST_UTILS_HPP_
