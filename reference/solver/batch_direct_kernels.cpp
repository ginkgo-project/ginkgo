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

#include "core/solver/batch_direct_kernels.hpp"


#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_direct {


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           matrix::BatchDense<ValueType>* const a,
           matrix::BatchDense<ValueType>* const b,
           gko::log::BatchLogData<ValueType>& logdata) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL);


template <typename ValueType>
void transpose_scale_copy(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const scaling_vec,
    const matrix::BatchDense<ValueType>* const orig,
    matrix::BatchDense<ValueType>* const scaled)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_TRANSPOSE_SCALE_COPY);


template <typename ValueType>
void pre_diag_scale_system_transpose(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::BatchDense<ValueType>* const a,
    const matrix::BatchDense<ValueType>* const b,
    const matrix::BatchDiagonal<ValueType>* const left_scale,
    const matrix::BatchDiagonal<ValueType>* const right_scale,
    matrix::BatchDense<ValueType>* const a_scaled_t,
    matrix::BatchDense<ValueType>* const b_scaled_t)
{
    const size_type nbatch = a->get_num_batch_entries();
    const int nrows = static_cast<int>(a->get_size().at()[0]);
    const int ncols = static_cast<int>(a->get_size().at()[1]);
    const int nrhs = static_cast<int>(b->get_size().at()[1]);
    const size_type a_stride = a->get_stride().at();
    const size_type a_scaled_stride = a_scaled_t->get_stride().at();
    const size_type b_stride = b->get_stride().at();
    const size_type b_scaled_stride = b_scaled_t->get_stride().at();
    constexpr size_type left_scale_stride = 1;
    constexpr size_type right_scale_stride = 1;
    for (size_type ib = 0; ib < nbatch; ib++) {
        auto ai = gko::batch::batch_entry_ptr(a->get_const_values(), a_stride,
                                              nrows, ib);
        auto asti = gko::batch::batch_entry_ptr(a_scaled_t->get_values(),
                                                a_scaled_stride, ncols, ib);
        auto bi = gko::batch::batch_entry_ptr(b->get_const_values(), b_stride,
                                              nrows, ib);
        auto bsti = gko::batch::batch_entry_ptr(b_scaled_t->get_values(),
                                                b_scaled_stride, nrhs, ib);
        auto lscalei = gko::batch::batch_entry_ptr(
            left_scale->get_const_values(), left_scale_stride, nrows, ib);
        auto rscalei = gko::batch::batch_entry_ptr(
            right_scale->get_const_values(), right_scale_stride, ncols, ib);
        for (int i = 0; i < nrows; i++) {
            const ValueType l_scale_factor = lscalei[i * left_scale_stride];
            for (int j = 0; j < ncols; j++) {
                asti[j * a_scaled_stride + i] = ai[i * a_stride + j] *
                                                l_scale_factor *
                                                rscalei[j * right_scale_stride];
            }
            for (int j = 0; j < nrhs; j++) {
                bsti[j * b_scaled_stride + i] =
                    bi[i * b_stride + j] * l_scale_factor;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_PRE_DIAG_SCALE_SYSTEM_TRANSPOSE);


}  // namespace batch_direct
}  // namespace reference
}  // namespace kernels
}  // namespace gko
