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

#include "core/components/absolute_array.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace components {


template <typename ValueType>
void inplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                            ValueType *data, size_type n)
{
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx_id) {
            const auto idx = idx_id[0];
            data[idx] = dpcpp::abs(data[idx]);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_ARRAY_KERNEL);


template <typename ValueType>
void outplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                             const ValueType *in, size_type n,
                             remove_complex<ValueType> *out)
{
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx_id) {
            const auto idx = idx_id[0];
            out[idx] = dpcpp::abs(in[idx]);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_ARRAY_KERNEL);


}  // namespace components
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
