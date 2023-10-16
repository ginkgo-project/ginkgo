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

/**
 * Batch ilu0 preconditioner.
 */
template <typename ValueType>
class batch_ilu final {
private:
    __dpct_inline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id, ValueType* const __restrict__ work)
    {
        mat_factorized_entry_ =
            gko::batch::batch_entry(mat_factorized_batch_, batch_id);
        work_ = work;
    }

    __dpct_inline__ void naive_combined_lower_trsv(
        const ValueType* const __restrict__ r, sycl::nd_item<3> item_ct1) const
    {
        const int num_rows = mat_factorized_entry_.num_rows;
        const int* const __restrict__ row_ptrs = mat_factorized_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = mat_factorized_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
            mat_factorized_entry_.values;

        if (item_ct1.get_local_linear_id() == 0) {
            // L * work_ = r
            for (int row_idx = 0; row_idx < num_rows; row_idx++) {
                ValueType sum{0};
                for (int i = row_ptrs[row_idx]; i < csr_diag_locs_[row_idx];
                     i++) {
                    sum += factored_values_entry[i] * work_[col_idxs[i]];
                }
                work_[row_idx] = r[row_idx] - sum;
            }
        }
    }


    __dpct_inline__ void naive_combined_upper_trsv(
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        const int num_rows = mat_factorized_entry_.num_rows;
        const int* const __restrict__ row_ptrs = mat_factorized_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = mat_factorized_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
            mat_factorized_entry_.values;

        if (item_ct1.get_local_linear_id() == 0) {
            // U * z = work_
            for (int row_idx = num_rows - 1; row_idx >= 0; row_idx--) {
                ValueType sum{0};
                for (int i = row_ptrs[row_idx + 1] - 1;
                     i > csr_diag_locs_[row_idx]; i--) {
                    sum += factored_values_entry[i] * z[col_idxs[i]];
                }
                z[row_idx] = (work_[row_idx] - sum) /
                             factored_values_entry[csr_diag_locs_[row_idx]];
            }
        }
    }

    __dpct_inline__ void warp_synchronous_combined_lower_trsv(
        const ValueType* const __restrict__ r, sycl::nd_item<3> item_ct1) const
    {
        const int num_rows = mat_factorized_entry_.num_rows;
        const int* const __restrict__ row_ptrs = mat_factorized_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = mat_factorized_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
            mat_factorized_entry_.values;


        for (int row_index = static_cast<int>(item_ct1.get_local_linear_id());
             row_index < num_rows;
             row_index += static_cast<int>(item_ct1.get_local_range().size())) {
            ValueType sum = 0;

            const int start = row_ptrs[row_index];
            const int end = csr_diag_locs_[row_index];

            int i = start;
            bool completed = false;

            while (!completed) {
                const int col_index = col_idxs[i];

                if (i < end && is_finite(load(work_, col_index))) {
                    sum += factored_values_entry[i] * load(work_, col_index);
                    i++;
                }

                if (i == end) {
                    ValueType val =
                        (r[row_index] - sum) /
                        1;  // 1 in place of factored_values_entry[end]
                    store(work_, row_index, val);
                    completed = true;
                }
            }
        }
    }

    __dpct_inline__ void warp_synchronous_combined_upper_trsv(
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        const int num_rows = mat_factorized_entry_.num_rows;
        const int* const __restrict__ row_ptrs = mat_factorized_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = mat_factorized_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
            mat_factorized_entry_.values;

        for (int row_index = num_rows - 1 -
                             static_cast<int>(item_ct1.get_local_linear_id());
             row_index >= 0;
             row_index -= static_cast<int>(item_ct1.get_local_range().size())) {
            ValueType sum = 0;

            const int start = csr_diag_locs_[row_index];
            const int end = row_ptrs[row_index + 1] - 1;
            int i = end;

            bool completed = false;

            while (!completed) {
                const int col_index = col_idxs[i];

                if (i > start && is_finite(load(z, col_index))) {
                    sum += factored_values_entry[i] * load(z, col_index);
                    i--;
                }

                if (i == start) {
                    ValueType val =
                        (work_[row_index] - sum) / factored_values_entry[start];
                    store(z, row_index, val);
                    completed = true;
                }
            }
        }
    }
    /*
        __dpct_inline__ void
        independent_thread_scheduling_combined_lower_trsv(
            const ValueType* const __restrict__ r) const
        {
            const int num_rows = mat_factorized_entry_.num_rows;
            const int* const __restrict__ row_ptrs =
       mat_factorized_entry_.row_ptrs; const int* const __restrict__ col_idxs =
       mat_factorized_entry_.col_idxs; const ValueType* const __restrict__
       factored_values_entry = mat_factorized_entry_.values;

            for (int row_index =
       static_cast<int>(item_ct1.get_local_linear_id()); row_index < num_rows;
       row_index += static_cast<int>(item_ct1.get_local_range().size())) {
                ValueType sum = 0;
                const int start = row_ptrs[row_index];
                const int end = csr_diag_locs_[row_index];

                for (int i = start; i < end; i++) {
                    const int col_index = col_idxs[i];
                    while (!is_finite(load(work_, col_index))) {
                    }
                    sum += factored_values_entry[i] * load(work_, col_index);
                }
                ValueType val = r[row_index] - sum;
                store(work_, row_index, val);
            }
        }

        __dpct_inline__ void
        independent_thread_scheduling_combined_upper_trsv(
            ValueType* const __restrict__ z) const
        {
            const int num_rows = mat_factorized_entry_.num_rows;
            const int* const __restrict__ row_ptrs =
       mat_factorized_entry_.row_ptrs; const int* const __restrict__ col_idxs =
       mat_factorized_entry_.col_idxs; const ValueType* const __restrict__
       factored_values_entry = mat_factorized_entry_.values;

            for (int row_index = num_rows - 1 -
       static_cast<int>(item_ct1.get_local_linear_id()); row_index >= 0;
       row_index -= static_cast<int>(item_ct1.get_local_range().size())) {
                ValueType sum = 0;
                const int start = csr_diag_locs_[row_index];
                const int end = row_ptrs[row_index + 1] - 1;

                for (int i = end; i > start; i--) {
                    const int col_index = col_idxs[i];
                    while (!is_finite(load(z, col_index))) {
                    }
                    sum += factored_values_entry[i] * load(z, col_index);
                }

                ValueType val =
                    (work_[row_index] - sum) / factored_values_entry[start];
                store(z, row_index, val);
            }
        }
    */

public:
    using value_type = ValueType;

    /**
     *
     * @param mat_factorized   Factorized matrix (that was factored externally).
     * @param csr_diag_locs  pointers to the diagonal entries in factorized
     *
     */
    batch_ilu(
        const gko::batch_csr::UniformBatch<const value_type>& mat_factorized,
        const int* __restrict__ const csr_diag_locs)
        : mat_factorized_batch_{mat_factorized}, csr_diag_locs_{csr_diag_locs}
    {}


    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    __dpct_inline__ static constexpr int dynamic_work_size(int nrows, int nnz)
    {
        return nrows;
    }

    /**
     * Complete the precond generation process.
     *
     */
    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }

    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }

    __dpct_inline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }

    __dpct_inline__ void apply(const int num_rows,
                               const ValueType* const __restrict__ r,
                               ValueType* const __restrict__ z,
                               sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r  ==> find z: L * U * z = r

        // NOTE: alternative sequential trsv as  busy-waiting while loop trsv
        // deadlocks for hip
        // L * work_ = r
        naive_combined_lower_trsv(r, item_ct1);
        // U * z = work_
        naive_combined_upper_trsv(z, item_ct1);
        return;
        /*
           for (int i = item_ct1.get_local_linear_id(); i < num_rows; i +=
           item_ct1.get_local_range().size()) { work_[i] = nan<ValueType>();
           z[i] = nan<ValueType>();
           }
           item_ct1.barrier(sycl::access::fence_space::local_space);
         */
        // L * work_ = r
        // U * z = work_
        /*
        // busy-waiting while loop approach

        warp_synchronous_combined_lower_trsv(r, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        warp_synchronous_combined_upper_trsv(z, item_ct1);
         */
    }

private:
    const gko::batch_csr::UniformBatch<const value_type> mat_factorized_batch_;
    const int* __restrict__ const csr_diag_locs_;
    gko::batch_csr::BatchEntry<const value_type> mat_factorized_entry_;
    value_type* __restrict__ work_;
};


template <typename ValueType>
__dpct_inline__ void batch_ilu_apply(batch_ilu<ValueType> prec, const int nrows,
                                     const ValueType* const b,
                                     ValueType* const x,
                                     ValueType* const sh_mem,
                                     sycl::nd_item<3> item_ct1)
{
    auto batch_id = item_ct1.get_group_linear_id();
    prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                  sh_mem, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    prec.apply(nrows, b, x, item_ct1);
}
