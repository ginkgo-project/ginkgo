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

#ifndef GKO_CORE_TEST_UTILS_MATRIX_SHUFFLE_HPP_
#define GKO_CORE_TEST_UTILS_MATRIX_SHUFFLE_HPP_

#include <algorithm>
#include <random>


#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace test {


// Plan for now: shuffle values and column indices to unsort the given matrix
// without changing the represented matrix.
template <typename ValueType, typename IndexType, typename RandomEngine>
void shuffle_nz_storage(matrix::Csr<ValueType, IndexType> *mtx,
                        RandomEngine &&engine)
{
    using value_type = ValueType;
    using index_type = IndexType;
    auto size = mtx->get_size();
    auto vals = mtx->get_values();
    auto row_ptrs = mtx->get_row_ptrs();
    auto cols = mtx->get_col_idxs();
    for (index_type row = 0; row < size[0]; ++row) {
        auto start = row_ptrs[row];
        auto end = row_ptrs[row + 1] - 1;
        std::uniform_int_distribution<index_type> dist{start, end};
        for (index_type i = start; i < start + (start + end) / 2; ++i) {
            auto a = dist(engine);
            auto b = dist(engine);
            using std::swap;
            if (a != b) {
                swap(vals[a], vals[b]);
                swap(cols[a], cols[b]);
            }
        }
    }
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_MATRIX_SHUFFLE_HPP_
