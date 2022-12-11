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

namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ic_init(
    const IndexType* __restrict__ l_row_ptrs, ValueType* __restrict__ l_vals,
    size_type num_rows)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }
    auto l_nz = l_row_ptrs[row + 1] - 1;
    auto diag = sqrt(l_vals[l_nz]);
    if (is_finite(diag)) {
        l_vals[l_nz] = diag;
    } else {
        l_vals[l_nz] = one<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ic_sweep(
    const IndexType* __restrict__ a_row_idxs,
    const IndexType* __restrict__ a_col_idxs,
    const ValueType* __restrict__ a_vals,
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_col_idxs, ValueType* __restrict__ l_vals,
    IndexType l_nnz)
{
    const auto l_nz = thread::get_thread_id_flat<IndexType>();
    if (l_nz >= l_nnz) {
        return;
    }
    const auto row = a_row_idxs[l_nz];
    const auto col = l_col_idxs[l_nz];
    const auto a_val = a_vals[l_nz];
    auto l_row_begin = l_row_ptrs[row];
    const auto l_row_end = l_row_ptrs[row + 1];
    auto lh_col_begin = l_row_ptrs[col];
    const auto lh_col_end = l_row_ptrs[col + 1];
    ValueType sum{};
    auto last_entry = col;
    while (l_row_begin < l_row_end && lh_col_begin < lh_col_end) {
        auto l_col = l_col_idxs[l_row_begin];
        auto lh_row = l_col_idxs[lh_col_begin];
        if (l_col == lh_row && l_col < last_entry) {
            sum += l_vals[l_row_begin] * conj(l_vals[lh_col_begin]);
        }
        l_row_begin += l_col <= lh_row;
        lh_col_begin += l_col >= lh_row;
    }
    auto to_write = row == col
                        ? sqrt(a_val - sum)
                        : (a_val - sum) / l_vals[l_row_ptrs[col + 1] - 1];
    if (is_finite(to_write)) {
        l_vals[l_nz] = to_write;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::Csr<ValueType, IndexType>* l)
{
    auto num_rows = l->get_size()[0];
    auto num_blocks = ceildiv(num_rows, default_block_size);
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_vals = l->get_values();
    if (num_blocks > 0) {
        kernel::
            ic_init<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                l_row_ptrs, as_device_type(l_vals), num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    const matrix::Coo<ValueType, IndexType>* a_lower,
                    matrix::Csr<ValueType, IndexType>* l)
{
    auto nnz = l->get_num_stored_elements();
    auto num_blocks = ceildiv(nnz, default_block_size);
    if (num_blocks > 0) {
        for (size_type i = 0; i < iterations; ++i) {
            kernel::ic_sweep<<<num_blocks, default_block_size, 0,
                               exec->get_stream()>>>(
                a_lower->get_const_row_idxs(), a_lower->get_const_col_idxs(),
                as_device_type(a_lower->get_const_values()),
                l->get_const_row_ptrs(), l->get_const_col_idxs(),
                as_device_type(l->get_values()),
                static_cast<IndexType>(l->get_num_stored_elements()));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);
