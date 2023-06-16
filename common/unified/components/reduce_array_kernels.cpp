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

#include "core/components/reduce_array_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace components {


template <typename ValueType>
void reduce_add_array(std::shared_ptr<const DefaultExecutor> exec,
                      const array<ValueType>& arr, array<ValueType>& result)
{
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto arr, auto result) {
            return i == 0 ? (arr[i] + result[0]) : arr[i];
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result.get_data(),
        arr.get_num_elems(), arr, result);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
