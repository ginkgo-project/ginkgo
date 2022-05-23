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

#include "core/base/device_matrix_data_kernels.hpp"


#include <ginkgo/core/base/types.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType, typename IndexType>
void soa_to_aos(std::shared_ptr<const DefaultExecutor> exec,
                const device_matrix_data<ValueType, IndexType>& in,
                array<matrix_data_entry<ValueType, IndexType>>& out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto rows, auto cols, auto vals, auto out) {
            out[i] = {rows[i], cols[i], vals[i]};
        },
        in.get_num_elems(), in.get_const_row_idxs(), in.get_const_col_idxs(),
        in.get_const_values(), out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL);


template <typename ValueType, typename IndexType>
void aos_to_soa(std::shared_ptr<const DefaultExecutor> exec,
                const array<matrix_data_entry<ValueType, IndexType>>& in,
                device_matrix_data<ValueType, IndexType>& out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in, auto rows, auto cols, auto vals) {
            rows[i] = in[i].row;
            cols[i] = in[i].column;
            vals[i] = unpack_member(in[i].value);
        },
        in.get_num_elems(), in, out.get_row_idxs(), out.get_col_idxs(),
        out.get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
