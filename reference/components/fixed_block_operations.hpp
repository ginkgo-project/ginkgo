/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#ifndef GKO_REFERENCE_COMPONENTS_FIXED_BLOCK_OPERATIONS_HPP_
#define GKO_REFERENCE_COMPONENTS_FIXED_BLOCK_OPERATIONS_HPP_


#include <algorithm>
#include <cmath>


#include <ginkgo/core/base/math.hpp>


#include "core/components/fixed_block.hpp"


namespace gko {
namespace kernels {


template <typename ValueType, int bs>
using Blkv_t = blockutils::FixedBlock<ValueType, bs, bs>;


template <typename ValueType, int block_size>
inline void swap_rows(const int row1, const int row2,
                      Blkv_t<ValueType, bs> &block)
{
    for (int i = 0; i < block_size; ++i) {
        std::swap(block(row1, i), block(row2, i));
    }
}


template <typename ValueType, int bs>
inline bool apply_gauss_jordan_transform(const int row, const int col,
                                         Blkv_t<ValueType, bs> &block)
{
    const auto d = block(row, col);
    if (d == zero<ValueType>()) {
        return false;
    }
    for (int i = 0; i < bs; ++i) {
        block(i, col) /= -d;
    }
    block(row, col) = zero<ValueType>();
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
            block(i, j) += block(i, col) * block(row, j);
        }
    }
    for (IndexType j = 0; j < bs; ++j) {
        block(row, j) /= d;
    }
    block(row, col) = one<ValueType>() / d;
    return true;
}


template <typename ValueType, int block_size>
inline bool invert_block(int *const perm, Blkv_t<ValueType, block_size> &block)
{
    constexpr int stride = block_size;
    using std::swap;
    for (int k = 0; k < block_size; ++k) {
        // choose pivot
        int cp = k;
        for (int i = k + 1; i < block_size; i++)
            if (abs(block(k, k)) < abs(block(i, k))) cp = i;

        swap_rows(k, cp, block_size, block, stride);
        swap(perm[k], perm[cp]);

        const bool status =
            apply_gauss_jordan_transform<ValueType, block_size>(k, k, block);
        if (!status) return false;
    }
    return true;
}


}  // namespace kernels
}  // namespace gko

#endif
