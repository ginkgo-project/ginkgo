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

#include "core/solver/batch_direct_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_direct {


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           matrix::BatchDense<ValueType>* const a,
           matrix::BatchDense<ValueType>* const b,
           gko::log::BatchLogData<ValueType>& logdata) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL);


template <typename ValueType>
void transpose_scale_copy(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const scaling_vec,
    const matrix::BatchDense<ValueType>* const orig,
    matrix::BatchDense<ValueType>* const scaled) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_TRANSPOSE_SCALE_COPY);


template <typename ValueType>
void pre_diag_scale_system_transpose(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDense<ValueType>* const a,
    const matrix::BatchDense<ValueType>* const b,
    const matrix::BatchDiagonal<ValueType>* const scalevec,
    const matrix::BatchDiagonal<ValueType>* const scalevec2,
    matrix::BatchDense<ValueType>* const a_scaled_t,
    matrix::BatchDense<ValueType>* const b_scaled_t) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_PRE_DIAG_SCALE_SYSTEM_TRANSPOSE);


}  // namespace batch_direct
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
