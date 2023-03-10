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

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, size_type slice_size,
    const size_type* __restrict__ slice_sets, const ValueType* __restrict__ a,
    const IndexType* __restrict__ cols, const ValueType* __restrict__ b,
    ValueType* __restrict__ c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] = val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void advanced_spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, size_type slice_size,
    const size_type* __restrict__ slice_sets,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ a,
    const IndexType* __restrict__ cols, const ValueType* __restrict__ b,
    const ValueType* __restrict__ beta, ValueType* __restrict__ c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] =
            beta[0] * c[row * c_stride + column_id] + alpha[0] * val;
    }
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Sellp<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const auto block_size = default_block_size;
    const dim3 grid(ceildiv(a->get_size()[0], block_size), b->get_size()[1]);

    if (grid.x > 0 && grid.y > 0) {
        spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a->get_size()[0], b->get_size()[1], b->get_stride(),
            c->get_stride(), a->get_slice_size(), a->get_const_slice_sets(),
            as_device_type(a->get_const_values()), a->get_const_col_idxs(),
            as_device_type(b->get_const_values()),
            as_device_type(c->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Sellp<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    const auto block_size = default_block_size;
    const dim3 grid(ceildiv(a->get_size()[0], block_size), b->get_size()[1]);

    if (grid.x > 0 && grid.y > 0) {
        advanced_spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a->get_size()[0], b->get_size()[1], b->get_stride(),
            c->get_stride(), a->get_slice_size(), a->get_const_slice_sets(),
            as_device_type(alpha->get_const_values()),
            as_device_type(a->get_const_values()), a->get_const_col_idxs(),
            as_device_type(b->get_const_values()),
            as_device_type(beta->get_const_values()),
            as_device_type(c->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);
