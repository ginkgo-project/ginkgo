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

#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
using BatchEntry = gko::batch_dense::BatchEntry<ValueType>;


template <typename ValueType>
void simple_apply(const BatchEntry<const ValueType> &a,
                  const BatchEntry<const ValueType> &b,
                  const BatchEntry<ValueType> &c)
{
    for (int row = 0; row < c.num_rows; ++row) {
        for (int col = 0; col < c.num_rhs; ++col) {
            c.values[row * c.stride + col] = zero<ValueType>();
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_rhs; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c->stride + col] +=
                    a.values[row * a->stride + inner] *
                    b.values[inner * b->stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void apply(const ValueType alpha, const BatchEntry<const ValueType> &a,
                  const BatchEntry<const ValueType> &b, const ValueType beta,
                  const BatchEntry<ValueType> &c)
{
    if (beta != zero<ValueType>()) {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= beta;
            }
        }
    } else {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= zero<ValueType>();
            }
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_rhs; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] += alpha *
                                                  a.at[row * a.stride + inner] *
                                                  b.at[inner * a.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void scale(const BatchEntry<const ValueType> &alpha,
                  const BatchEntry<ValueType> &x)
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
inline void add_scaled(const BatchEntry<const ValueType> &alpha,
                       const BatchEntry<const ValueType> &x,
                       const BatchEntry<ValueType> &y)
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
inline void compute_norm2(const BatchEntry<const ValueType> &x,
                          const BatchEntry<remove_complex<ValueType>> &result)
{
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = zero<remove_complex<ValueType>>();
    }
    for (int i = 0; i < x.num_rows; ++i) {
        for (size_type j = 0; j < x.num_rhs; ++j) {
            result.values[j] += squared_norm(x.values[i * x.stride + j]);
        }
    }
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = sqrt(result.values[j]);
    }
}


}  // namespace batch_dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
