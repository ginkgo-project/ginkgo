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
inline void simple_apply_kernel(
    const gko::batch::matrix::dense::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    for (int row = 0; row < c.num_rows; ++row) {
        for (int col = 0; col < c.num_rhs; ++col) {
            c.values[row * c.stride + col] = gko::zero<ValueType>();
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_cols; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::dense::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    if (beta != gko::zero<ValueType>()) {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= beta;
            }
        }
    } else {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] = gko::zero<ValueType>();
            }
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_cols; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    alpha * a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}
