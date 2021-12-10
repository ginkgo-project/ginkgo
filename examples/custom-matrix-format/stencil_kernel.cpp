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

#include <ginkgo/ginkgo.hpp>


#include <ginkgo/kernels/kernel_launch.hpp>


namespace GKO_DEVICE_NAMESPACE {


using namespace gko::kernels::GKO_DEVICE_NAMESPACE;


template <typename ValueType>
void stencil_kernel(std::shared_ptr<const DefaultExecutor> exec,
                    std::size_t size, const ValueType* coefs,
                    const ValueType* b, ValueType* x)
{
    run_kernel(
        exec,
        GKO_KERNEL(auto i, auto coefs, auto b, auto x, auto size) {
            auto result = coefs[1] * b[i];
            if (i > 0) {
                result += coefs[0] * b[i - 1];
            }
            if (i < size - 1) {
                result += coefs[2] * b[i + 1];
            }
            x[i] = result;
        },
        size, coefs, b, x, size);
}

template void stencil_kernel<double>(std::shared_ptr<const DefaultExecutor>,
                                     std::size_t, const double*, const double*,
                                     double*);
template void stencil_kernel<float>(std::shared_ptr<const DefaultExecutor>,
                                    std::size_t, const float*, const float*,
                                    float*);


}  // namespace GKO_DEVICE_NAMESPACE
