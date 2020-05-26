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

#include <limits>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace reference {


template <typename ValueType, typename IndexType, typename BeginCallback,
          typename EntryCallback, typename EndCallback>
void abstract_spgeam(const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     BeginCallback begin_cb, EntryCallback entry_cb,
                     EndCallback end_cb)
{
    auto num_rows = a->get_size()[0];
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    for (size_type row = 0; row < num_rows; ++row) {
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto b_begin = b_row_ptrs[row];
        auto b_end = b_row_ptrs[row + 1];
        auto total_size = (a_end - a_begin) + (b_end - b_begin);
        bool skip{};
        auto local_data = begin_cb(row);
        for (IndexType i = 0; i < total_size; ++i) {
            if (skip) {
                skip = false;
                continue;
            }
            // load column indices or sentinel
            auto a_col = checked_load(a_col_idxs, a_begin, a_end, sentinel);
            auto b_col = checked_load(b_col_idxs, b_begin, b_end, sentinel);
            auto a_val =
                checked_load(a_vals, a_begin, a_end, zero<ValueType>());
            auto b_val =
                checked_load(b_vals, b_begin, b_end, zero<ValueType>());
            auto col = min(a_col, b_col);
            // callback
            entry_cb(row, col, a_col == col ? a_val : zero<ValueType>(),
                     b_col == col ? b_val : zero<ValueType>(), local_data);
            // advance indices
            a_begin += (a_col <= b_col);
            b_begin += (b_col <= a_col);
            skip = a_col == b_col;
        }
        end_cb(row, local_data);
    }
}


}  // namespace reference
}  // namespace kernels
}  // namespace gko