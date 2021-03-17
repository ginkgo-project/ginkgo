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

/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
using BatchEntry = gko::batch_dense::BatchEntry<ValueType>;


template <typename ValueType>
__device__ __forceinline__ void add_scaled(
    const BatchEntry<const ValueType> &alpha,
    const BatchEntry<const ValueType> &x, const BatchEntry<ValueType> &y)
{
    if (alpha.num_rhs == 1) {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = threadIdx.x; i < x.num_rows; i += blockDim.x) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}


/**
 * Computes the 2-norms of some column vectors in global or shared memory.
 *
 * @param result  Holds norm value for each vector in x.
 */
template <typename ValueType>
__device__ __forceinline__ void compute_norm2(
    const BatchEntry<const ValueType> &x,
    const BatchEntry<remove_complex<ValueType>> &result)
{
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = zero<remove_complex<ValueType>>();
    }
    // TODO: put a parallel reduce here
    if (threadIdx.x == 0) {
        for (int i = 0; i < x.num_rows; i++) {
            for (int j = 0; j < x.num_rhs; ++j) {
                result.values[j] += squared_norm(x.values[i * x.stride + j]);
            }
        }
    }
    for (int j = threadIdx.x; j < x.num_rhs; j += blockDim.x) {
        result.values[j] = sqrt(result.values[j]);
    }
}


}  // namespace batch_dense
