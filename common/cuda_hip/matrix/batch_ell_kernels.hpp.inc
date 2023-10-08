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


template <typename ValueType>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::batch_ell::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x)
{
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp;
    }
}

template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void simple_apply_kernel(const gko::batch::matrix::
                                                      batch_ell::uniform_batch<
                                                          const ValueType>
                                                          mat,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              const ValueType>
                                                              b,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              ValueType>
                                                              x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        simple_apply(mat_b, b_b.values, x_b.values);
    }
}


template <typename ValueType>
__device__ __forceinline__ void advanced_apply(
    const ValueType alpha,
    const gko::batch::matrix::batch_ell::batch_item<const ValueType>& mat,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ x)
{
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += alpha * val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp + beta * x[tidx];
    }
}

template <typename ValueType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void advanced_apply_kernel(const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                alpha,
                                                    const gko::batch::matrix::
                                                        batch_ell::
                                                            uniform_batch<
                                                                const ValueType>
                                                                mat,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                b,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                beta,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                ValueType>
                                                                x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        advanced_apply(alpha_b.values[0], mat_b, b_b.values, beta_b.values[0],
                       x_b.values);
    }
}
