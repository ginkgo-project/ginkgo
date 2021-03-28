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

namespace kernel {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_block_ptrs(
    size_type num_blocks, const size_type *__restrict__ block_sizes,
    IndexType *__restrict__ ptrs)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_blocks) {
        ptrs[tidx] = block_sizes[tidx];
    }
}


// template <typename IndexType>
// __global__ __launch_bounds__(default_block_size) void spmv(
//     size_type num_blocks, const IndexType *__restrict__ block_sizes,
//     IndexType *__restrict__ ptrs)
// {
//     const auto tidx = thread::get_thread_id_flat();
//     if (tidx < num_blocks) {
//         for (size_type i = 0; i < a->get_num_blocks(); ++i) {
//             auto loc_size = a->get_block_dimensions()[i];
//             size_type offset = block_ptrs[i];
//             auto loc_mtx = a->get_block_mtxs()[i];
//             auto row_ptrs = loc_mtx->get_const_row_ptrs();
//             auto col_idxs = loc_mtx->get_const_col_idxs();
//             auto vals = loc_mtx->get_const_values();
//             const auto loc_b =
//                 dense_b->create_submatrix(span{offset, offset + loc_size[0]},
//                                           span{0, dense_b->get_size()[1]});
//             auto loc_x = c->create_submatrix(span{offset, offset +
//             loc_size[0]},
//                                              span{0, c->get_size()[1]});

//             for (size_type row = 0; row < loc_size[0]; ++row) {
//                 for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
//                     loc_x->at(row, j) = zero<ValueType>();
//                 }
//                 for (size_type k = row_ptrs[row];
//                      k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
//                     auto val = vals[k];
//                     auto col = col_idxs[k];
//                     for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
//                         loc_x->at(row, j) += val * loc_b->at(col, j);
//                     }
//                 }
//             }
//         }
//     }
// }


}  // namespace kernel
