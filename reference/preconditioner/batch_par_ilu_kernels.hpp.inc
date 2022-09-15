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


template <typename ValueType>
inline void initialize_batch_l_and_batch_u_individual_entry_impl(
    const size_type batch_id, const int nnz, const int* const col_idxs,
    const ValueType* const vals, const int l_nnz,
    const int* const l_col_holders, int* const l_col_idxs,
    ValueType* const l_vals, const int u_nnz, const int* const u_col_holders,
    int* const u_col_idxs, ValueType* const u_vals)
{
    int col;
    ValueType val;

    for (int i = 0; i < l_nnz; i++) {
        const int l_A_index = l_col_holders[i];
        if (l_A_index >= 0) {
            col = col_idxs[l_A_index];
            val = vals[l_A_index + batch_id * nnz];
        } else {
            col = -1 * (l_A_index + 1);
            val = one<ValueType>();
        }

        if (batch_id == 0) {
            l_col_idxs[i] = col;
        }

        l_vals[i + batch_id * l_nnz] = val;
    }

    for (int i = 0; i < u_nnz; i++) {
        const int u_A_index = u_col_holders[i];

        col = col_idxs[u_A_index];
        val = vals[u_A_index + batch_id * nnz];

        if (batch_id == 0) {
            u_col_idxs[i] = col;
        }

        u_vals[i + batch_id * u_nnz] = val;
    }
}

template <typename ValueType>
inline void parilu0_sweep_for_batch_entry(const int nnz,
                                          const ValueType* const values_entry,
                                          ValueType* const L_values_entry,
                                          ValueType* const U_values_entry,
                                          const int* dependencies,
                                          const int* nz_ptrs)
{
    for (int el = 0; el < nnz;
         el++)  // with this fixed order (of computing values), just 1 sweep is
                // sufficient (it is basically equivalent to exact ilu0)
    {
        ValueType diag_val{one<ValueType>()};

        const int st = nz_ptrs[el];
        const int end = nz_ptrs[el + 1] - 1;

        bool has_diag_dependency = (end + 1 - st) % 2 == 0 ? true : false;

        ValueType sum{zero<ValueType>()};

        for (int idx = st + 1; idx <= end - 1; idx += 2) {
            sum += L_values_entry[dependencies[idx]] *
                   U_values_entry[dependencies[idx + 1]];
        }

        if (has_diag_dependency == true) {
            diag_val = U_values_entry[dependencies[end]];
        }

        ValueType to_write = (values_entry[el] - sum) / diag_val;

        if (has_diag_dependency == true) {
            L_values_entry[dependencies[st]] = to_write;
        } else {
            U_values_entry[dependencies[st]] = to_write;
        }
    }
}

template <typename ValueType>
inline void compute_parilu0_batch_entry_impl(
    const size_type batch_id, const int nrows, const int nnz,
    const ValueType* const values, const int L_nnz, ValueType* const L_values,
    const int U_nnz, ValueType* const U_values, const int num_sweeps,
    const int* const dependencies, const int* const nz_ptrs)
{
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        parilu0_sweep_for_batch_entry(
            nnz, values + batch_id * nnz, L_values + batch_id * L_nnz,
            U_values + batch_id * U_nnz, dependencies, nz_ptrs);
    }
}
