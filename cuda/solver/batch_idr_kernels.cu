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

#include "core/solver/batch_idr_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>

#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {


constexpr int default_block_size = 128;
constexpr int sm_multiplier = 4;

/**
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {

#include "common/components/uninitialized_array.hpp.inc"


#include "common/log/batch_logger.hpp.inc"
#include "common/matrix/batch_csr_kernels.hpp.inc"
#include "common/matrix/batch_dense_kernels.hpp.inc"
#include "common/preconditioner/batch_identity.hpp.inc"
#include "common/preconditioner/batch_jacobi.hpp.inc"
#include "common/stop/batch_criteria.hpp.inc"


template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;

template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           log::BatchLogData<ValueType> &logdata) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
