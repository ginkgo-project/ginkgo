/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


// only works for ILU(0) for now
template <typename ValueType>
__device__ __forceinline__ void generate(
    const int num_rows, const int* const __restrict__ a_row_ptrs,
    const int* const __restrict__ a_col_idxs,
    const ValueType* const __restrict__ a_values,
    const int* const __restrict__ l_row_ptrs,
    const int* const __restrict__ l_col_idxs,
    ValueType* const __restrict__ l_values,
    const int* const __restrict__ u_row_ptrs,
    const int* const __restrict__ u_col_idxs,
    ValueType* const __restrict__ u_values)
{
    // FIXME: sequential
    if (threadIdx.x == 0) {
        int l_nnz = 0;
        int u_nnz = 0;
        for (int irow = 0; irow < num_rows; irow++) {
            for (int jz = a_row_ptrs[irow]; jz < a_row_ptrs[irow + 1]; jz++) {
                const int col = a_col_idxs[jz];
                const auto minrowcol = min(irow, col);
                ValueType sum{};
                for (int kz = l_row_ptrs[irow]; l_col_idxs[kz] < minrowcol;
                     kz++) {
                    const int l_col = l_col_idxs[kz];
                    for (int lz = u_row_ptrs[l_col]; lz < u_row_ptrs[l_col + 1];
                         lz++) {
                        if (u_col_idxs[lz] == col) {
                            sum += l_values[kz] * u_values[lz];
                        }
                    }
                }
                if (irow > col) {
                    // lower
                    l_values[l_nnz] =
                        (a_values[jz] - sum) / u_values[u_row_ptrs[col]];
                    l_nnz++;
                } else {
                    // upper
                    u_values[u_nnz] = a_values[jz] - sum;
                    u_nnz++;
                    if (irow == col) {
                        l_values[l_nnz] = 1.0;
                        l_nnz++;
                    }
                }
            }
        }
    }
}

template <typename ValueType>
__global__ void generate(const size_type batch_size, const int num_rows,
                         const int nnz,
                         const int* const __restrict__ a_row_ptrs,
                         const int* const __restrict__ a_col_idxs,
                         const ValueType* const __restrict__ a_values,
                         const int l_nnz,
                         const int* const __restrict__ l_row_ptrs,
                         const int* const __restrict__ l_col_idxs,
                         ValueType* const __restrict__ l_values,
                         const int u_nnz,
                         const int* const __restrict__ u_row_ptrs,
                         const int* const __restrict__ u_col_idxs,
                         ValueType* const __restrict__ u_values)
{
    const size_type batch_id = blockIdx.x;
    const auto a_values_i = a_values + batch_id * nnz;
    const auto l_values_i = l_values + batch_id * l_nnz;
    const auto u_values_i = u_values + batch_id * u_nnz;
    generate(num_rows, a_row_ptrs, a_col_idxs, a_values_i, l_row_ptrs,
             l_col_idxs, l_values_i, u_row_ptrs, u_col_idxs, u_values_i);
}
