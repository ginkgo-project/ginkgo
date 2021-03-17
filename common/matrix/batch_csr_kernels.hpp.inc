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


namespace batch_csr {

/**
 * Device kernel for SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void spmv_ker(
    const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const gko::batch_dense::BatchEntry<ValueType> &c)
{
    for (int row = threadIdx.x; row < a.num_rows; row += blockDim.x) {
        for (int j = 0; j < b.num_rhs; ++j) {
            c.values[row * c.stride + j] = zero<ValueType>();
        }
        for (auto k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            auto val = a.values[k];
            auto col = a.col_idxs[k];
            for (int j = 0; j < b->num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    val * b.values[col * b.stride + j];
            }
        }
    }
}


/**
 * Device kernel for 'advanced' SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void adv_spmv_ker(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType> &c)
{
    for (int row = threadIdx.x; row < a.num_rows; row += blockDim.x) {
        for (int j = 0; j < c.num_rhs; ++j) {
            c.values[row * c.stride + j] *= beta;
        }
        for (int k = a.row_ptrs[row]; k < a.row_ptrs[row + 1]; ++k) {
            const auto val = a.values[k];
            const auto col = a.col_idxs[k];
            for (int j = 0; j < c.num_rhs; ++j) {
                c.values[row * c.stride + j] +=
                    alpha * val * b.values[col * b.stride + j];
            }
        }
    }
}

}  // namespace batch_csr
