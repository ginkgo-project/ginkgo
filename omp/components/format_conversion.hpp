/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <omp.h>


namespace gko {
namespace kernels {
namespace omp {


/**
 * @internal
 *
 * Converts an array of indexes 'idxs' in any order to an array of pointers
 * 'ptrs'. This is used for transposing a csr matrix when calculating the row
 * pointers of the transposed matrix out of the column indices of the original
 * matrix.
 */
template <typename IndexType>
inline void convert_unsorted_idxs_to_ptrs(const IndexType *idxs,
                                          size_type num_nonzeros,
                                          IndexType *ptrs, size_type length)
{
#pragma omp parallel
    {
        const size_type tid = omp_get_thread_num();
        const size_type work_size = ceildiv(length, omp_get_num_threads());
        const size_type start = work_size * tid;
        const size_type end = min(length, work_size * (tid + 1));
        std::fill(ptrs + start, ptrs + end, 0);
    }

#pragma omp parallel
    {
        const size_type tid = omp_get_thread_num();
        const size_type work_size =
            ceildiv(num_nonzeros, omp_get_num_threads());
        const size_type start = work_size * tid;
        const size_type end = min(num_nonzeros, work_size * (tid + 1));
        std::for_each(idxs + start, idxs + end, [&](IndexType v) {
            if (v + 1 < length) {
#pragma omp atomic
                ++ptrs[v + 1];
            }
        });
    }

    std::partial_sum(ptrs, ptrs + length, ptrs);
}


/**
 * @internal
 *
 * Converts an array of indexes which are already stored in an increasing order
 * to an array of pointers. This is used to calculate the row pointers when
 * converting a coo matrix to a csr matrix.
 */
template <typename IndexType>
inline void convert_sorted_idxs_to_ptrs(const IndexType *idxs,
                                        size_type num_nonzeros, IndexType *ptrs,
                                        size_type length)
{
    ptrs[0] = 0;
    ptrs[length - 1] = num_nonzeros;

#pragma omp parallel
    {
        const size_type tid = omp_get_thread_num();
        const size_type work_size =
            ceildiv(num_nonzeros, omp_get_num_threads());
        const size_type start = work_size * tid;
        const size_type end = min(num_nonzeros - 1, work_size * (tid + 1));

        for (size_type i = start; i < end; i++) {
            for (size_type j = idxs[i] + 1; j <= idxs[i + 1]; j++) {
                ptrs[j] = i + 1;
            }
        }
    }
}


template <typename IndexType>
inline void convert_ptrs_to_idxs(const IndexType *ptrs, size_type num_rows,
                                 IndexType *idxs)
{
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = ptrs[row]; i < static_cast<size_type>(ptrs[row + 1]);
             ++i) {
            idxs[i] = row;
        }
    }
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
