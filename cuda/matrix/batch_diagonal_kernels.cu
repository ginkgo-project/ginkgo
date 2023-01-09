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

#include "core/matrix/batch_diagonal_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


constexpr auto default_block_size = 256;


#include "common/cuda_hip/matrix/batch_diagonal_kernels.hpp.inc"


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchDiagonal<ValueType>* const diag,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x)
{
    if (!b->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto b_stride = b->get_stride().at();
    const auto x_stride = x->get_stride().at();
    const auto nrows = static_cast<int>(diag->get_size().at()[0]);
    const auto ncols = static_cast<int>(diag->get_size().at()[1]);
    const auto nrhs = static_cast<int>(x->get_size().at()[1]);
    const int num_blocks = b->get_num_batch_entries();
    uniform_batch_diag_apply<<<num_blocks, default_block_size>>>(
        num_blocks, nrows, ncols, as_cuda_type(diag->get_const_values()), nrhs,
        b_stride, as_cuda_type(b->get_const_values()), x_stride,
        as_cuda_type(x->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIAGONAL_APPLY_KERNEL);


template <typename ValueType>
void apply_in_place(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDiagonal<ValueType>* const diag,
                    matrix::BatchDense<ValueType>* const b)
{
    if (!diag->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto stride = b->get_stride().at();
    const auto nrows = b->get_size().at()[0];
    const auto nrhs = b->get_size().at()[1];
    const auto nbatch = b->get_num_batch_entries();
    const int num_blocks = b->get_num_batch_entries();
    uniform_batch_diag_apply_inplace<<<num_blocks, default_block_size>>>(
        nrows, stride, nrhs, nbatch, as_cuda_type(diag->get_const_values()),
        as_cuda_type(b->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIAGONAL_APPLY_IN_PLACE_KERNEL);


}  // namespace batch_diagonal
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
