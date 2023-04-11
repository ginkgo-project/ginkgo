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

#include "core/solver/batch_band_solver_kernels.hpp"

#include <chrono>
#include <ginkgo/config.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>
#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/math.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/load_store.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"
#include "cuda/preconditioner/jacobi_common.hpp"

namespace gko {
namespace kernels {
namespace cuda {
namespace batch_band_solver {

namespace {

constexpr int default_block_size = 128;


}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::BatchBand<ValueType>* const band_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x, const int workspace_size,
           ValueType* const workspace_ptr,
           const enum gko::solver::batch_band_solve_approach approach,
           const int blocked_solve_panel_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BAND_SOLVER_APPLY_KERNEL);


}  // namespace batch_band_solver
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
