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

#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace csr {


template <typename IndexType>
inline void convert_ptrs_to_idxs(const IndexType* ptrs, size_type num_rows,
                                 IndexType* idxs)
{
    size_type ind = 0;

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = ptrs[row]; i < static_cast<size_type>(ptrs[row + 1]);
             ++i) {
            idxs[ind++] = row;
        }
    }
}


}  // namespace csr


template <typename ValueType, typename IndexType>
inline size_type mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                                const IndexType* row_idxs,
                                const IndexType* col_idxs,
                                const IndexType num_rows,
                                const IndexType block_size) GKO_NOT_IMPLEMENTED;
/*
{
    size_type p = 0;
    for (size_type b = 0; b < nb; b++) {
        size_type k = b * BLOCK;
        size_type r = row_idxs[k];
        size_type c = 0;
        for (size_type l = 0; l < BLOCK && k < nz; l++, k++) {
            if (row_idxs[k] != r) { // new row
                r = row_idxs[k];
                c = 0;
                p++;
            }
            size_type d = col_idxs[k] - c;
            if (d < 0x7d) {
                p++;
            } else if (d < 0xffff) {
                p += 3;
            } else {
                p += 5;
            }
            c = col_idxs[k];
        }
    }
    return p;
}
*/


}  // namespace reference
}  // namespace kernels
}  // namespace gko
