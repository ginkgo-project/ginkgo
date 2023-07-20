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
inline void scale_kernel(
    const gko::batch_multi_vector::BatchEntry<const ValueType>& alpha,
    const gko::batch_multi_vector::BatchEntry<ValueType>& x)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[0];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[j];
            }
        }
    }
}


template <typename ValueType>
inline void add_scaled_kernel(
    const gko::batch_multi_vector::BatchEntry<const ValueType>& alpha,
    const gko::batch_multi_vector::BatchEntry<const ValueType>& x,
    const gko::batch_multi_vector::BatchEntry<ValueType>& y)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}


template <typename ValueType>
inline void compute_dot_product_kernel(
    const gko::batch_multi_vector::BatchEntry<const ValueType>& x,
    const gko::batch_multi_vector::BatchEntry<const ValueType>& y,
    const gko::batch_multi_vector::BatchEntry<ValueType>& result)
{
    for (int c = 0; c < result.num_rhs; c++) {
        result.values[c] = gko::zero<ValueType>();
    }

    for (int r = 0; r < x.num_rows; r++) {
        for (int c = 0; c < x.num_rhs; c++) {
            result.values[c] +=
                conj(x.values[r * x.stride + c]) * y.values[r * y.stride + c];
        }
    }
}


template <typename ValueType>
inline void compute_norm2_kernel(
    const gko::batch_multi_vector::BatchEntry<const ValueType>& x,
    const gko::batch_multi_vector::BatchEntry<gko::remove_complex<ValueType>>&
        result)
{
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = gko::zero<gko::remove_complex<ValueType>>();
    }
    for (int i = 0; i < x.num_rows; ++i) {
        for (int j = 0; j < x.num_rhs; ++j) {
            result.values[j] += squared_norm(x.values[i * x.stride + j]);
        }
    }
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = sqrt(result.values[j]);
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
inline void copy_kernel(
    const gko::batch_multi_vector::BatchEntry<const ValueType>& in,
    const gko::batch_multi_vector::BatchEntry<ValueType>& out)
{
    for (int iz = 0; iz < in.num_rows * in.num_rhs; iz++) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}
