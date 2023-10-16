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

#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_ILU_ISAI_HPP_INC_
#define GKO_DPCPP_PRECONDITIONER_BATCH_ILU_ISAI_HPP_INC_


namespace batch_ilu_isai_temp {

// TODO: find a way to use single_matvec_kernel in batch_csr_kernels.hpp.inc
template <typename ValueType>
__dpct_inline__ void single_csr_spmv_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c,
    sycl::nd_item<3> item_ct1)
{
    const auto num_rows = a.num_rows;
    const auto val = a.values;
    const auto col = a.col_idxs;
    for (int tidx = item_ct1.get_local_linear_id(); tidx < num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (auto nnz = a.row_ptrs[tidx]; nnz < a.row_ptrs[tidx + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        c[tidx] = temp;
    }
}

}  // namespace batch_ilu_isai_temp

/**
 * Batch ilu-isai preconditioner.
 */
template <typename ValueType>
class batch_ilu_isai final {
private:
    __dpct_inline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id, ValueType* const __restrict__ work,
        sycl::nd_item<3> item_ct1)
    {
        work_ = work;
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        l_isai_entry_ = gko::batch::batch_entry(l_isai_batch_, batch_id);
        u_isai_entry_ = gko::batch::batch_entry(u_isai_batch_, batch_id);
        if (apply_type_ ==
            gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm) {
            mult_inv_entry_ =
                gko::batch::batch_entry(mult_inv_batch_, batch_id);
        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      relaxation_steps_isai_with_spgemm) {
            iter_mat_lower_solve_entry_ =
                gko::batch::batch_entry(iter_mat_lower_solve_batch_, batch_id);
            iter_mat_upper_solve_entry_ =
                gko::batch::batch_entry(iter_mat_upper_solve_batch_, batch_id);
        }
    }


    __dpct_inline__ void apply_spmv_isai_simple(
        const int num_rows, const ValueType* const __restrict__ r,
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r  ==> L * U * z = r ==> lai_U * lai_L * L * U * z =
        // lai_U * laiL * r ===> z = lai_U * laiL * r
        batch_ilu_isai_temp::single_csr_spmv_kernel(l_isai_entry_, r, work_,
                                                    item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        batch_ilu_isai_temp::single_csr_spmv_kernel(u_isai_entry_, work_, z,
                                                    item_ct1);
    }

    __dpct_inline__ void apply_spmv_isai_with_spgemm(
        const int num_rows, const ValueType* const __restrict__ r,
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r  ==> L * U * z = r ==> lai_U * lai_L * L * U * z =
        // lai_U * laiL * r ===> z = lai_U * laiL * r

        batch_ilu_isai_temp::single_csr_spmv_kernel(mult_inv_entry_, r, z,
                                                    item_ct1);
    }

    __dpct_inline__ void solve_system_relaxation_steps_simple(
        const int num_rows,
        const gko::batch_csr::BatchEntry<const ValueType>& mat_entry,
        const gko::batch_csr::BatchEntry<const ValueType>& mat_isai_entry,
        const ValueType* const __restrict__ rhs,
        ValueType* const __restrict__ sol_guess,
        ValueType* const __restrict__ temp1,
        ValueType* const __restrict__ temp2, sycl::nd_item<3> item_ct1) const
    {
        // sol_guess_updated = lai_mat * rhs + (I - lai_mat * mat) * sol_guess
        // (iterations)

        batch_ilu_isai_temp::single_csr_spmv_kernel(mat_isai_entry, rhs, temp1,
                                                    item_ct1);
        for (int i = item_ct1.get_local_linear_id(); i < num_rows;
             i += item_ct1.get_local_range().size()) {
            sol_guess[i] = 0;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int step = 0; step < num_relaxation_steps_; step++) {
            // sol_guess_updated =  temp1 + sol_guess -  (lai_mat * temp2) where
            // temp2 = mat * sol_guess
            batch_ilu_isai_temp::single_csr_spmv_kernel(mat_entry, sol_guess,
                                                        temp2, item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);
            // merged kernel
            for (int row = item_ct1.get_local_linear_id(); row < num_rows;
                 row += item_ct1.get_local_range().size()) {
                auto sum = zero<ValueType>();
                for (auto i = mat_isai_entry.row_ptrs[row];
                     i < mat_isai_entry.row_ptrs[row + 1]; i++) {
                    const auto col_idx = mat_isai_entry.col_idxs[i];
                    sum += mat_isai_entry.values[i] * temp2[col_idx];
                }

                sol_guess[row] = temp1[row] + sol_guess[row] - sum;
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

    __dpct_inline__ void apply_relaxation_steps_isai_simple(
        const int num_rows, const ValueType* const __restrict__ r,
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r
        // L * U * z = r
        // L * y = r  and then U * z = y
        // y_updated = lai_L * r + (I - lai_L * L) * y_old    (iterations)
        // Once y is obtained, z_updated = lai_U * y + (I - lai_U * U) * z_old
        // (iterations)

        ValueType* y = work_;
        ValueType* temp1 = work_ + num_rows;
        ValueType* temp2 = temp1 + num_rows;

        solve_system_relaxation_steps_simple(num_rows, l_entry_, l_isai_entry_,
                                             r, y, temp1, temp2, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        solve_system_relaxation_steps_simple(num_rows, u_entry_, u_isai_entry_,
                                             y, z, temp1, temp2, item_ct1);
    }

    __dpct_inline__ void solve_system_relaxation_steps_external_spgemm(
        const int num_rows,
        const gko::batch_csr::BatchEntry<const ValueType>& iteration_mat_entry,
        const gko::batch_csr::BatchEntry<const ValueType>& mat_isai_entry,
        const ValueType* const __restrict__ rhs,
        ValueType* const __restrict__ sol_guess,
        ValueType* const __restrict__ temp, sycl::nd_item<3> item_ct1) const
    {
        // sol_guess_updated = lai_mat * rhs + (I - lai_mat * mat) * sol_guess
        // (iterations)
        batch_ilu_isai_temp::single_csr_spmv_kernel(mat_isai_entry, rhs, temp,
                                                    item_ct1);
        for (int i = item_ct1.get_local_linear_id(); i < num_rows;
             i += item_ct1.get_local_range().size()) {
            sol_guess[i] = 0;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int step = 0; step < num_relaxation_steps_; step++) {
            // sol_guess_updated =  temp + iter_mat * sol_guess

            // merged kernel
            for (int row = item_ct1.get_local_linear_id(); row < num_rows;
                 row += item_ct1.get_local_range().size()) {
                auto sum = zero<ValueType>();
                for (auto i = iteration_mat_entry.row_ptrs[row];
                     i < iteration_mat_entry.row_ptrs[row + 1]; i++) {
                    const auto col_idx = iteration_mat_entry.col_idxs[i];
                    sum += iteration_mat_entry.values[i] * sol_guess[col_idx];
                }

                sol_guess[row] = temp[row] + sum;
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

    __dpct_inline__ void apply_relaxation_steps_isai_with_spgemm(
        const int num_rows, const ValueType* const __restrict__ r,
        ValueType* const __restrict__ z, sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r
        // L * U * z = r
        // L * y = r  and then U * z = y
        // y_updated = lai_L * r + (I - lai_L * L) * y_old    (iterations)
        // Once y is obtained, z_updated = lai_U * y + (I - lai_U * U) * z_old
        // (iterations)

        ValueType* y = work_;
        ValueType* temp = work_ + num_rows;
        solve_system_relaxation_steps_external_spgemm(
            num_rows, iter_mat_lower_solve_entry_, l_isai_entry_, r, y, temp,
            item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        solve_system_relaxation_steps_external_spgemm(
            num_rows, iter_mat_upper_solve_entry_, u_isai_entry_, y, z, temp,
            item_ct1);
    }

public:
    using value_type = ValueType;

    /**
     * @param l_batch  Lower Triangular factor (that was generated externally).
     *
     * @param u_batch  Upper Triangular factor (that was generated externally).
     *
     * @param l_isai_batch  Lower Triangular factor's Incomplete Sparse
     * Approximate Inverse (that was generated externally).
     *
     * @param u_isai_batch  Upper Triangular factor's Incomplete Sparse
     * Approximate Inverse (that was generated externally).
     *
     * @param mult_inv_batch Mutiplication of inverses:  lai_U * lai_U (that was
     * generated externally).
     *
     * @param iter_mat_lower_solve_batch Iteration matrix (= I - lai_L * L ) for
     * solving L * y = r
     *
     * @param iter_mat_upper_solve_batch Iteration matrix (= I - lai_U * U ) for
     * solving U * z = y
     *
     * @param apply_type How the preconditioner is to be applied?
     *
     * @param num_relaxation_steps
     */
    batch_ilu_isai(
        const gko::batch_csr::UniformBatch<const value_type>& l_batch,
        const gko::batch_csr::UniformBatch<const value_type>& u_batch,
        const gko::batch_csr::UniformBatch<const value_type>& l_isai_batch,
        const gko::batch_csr::UniformBatch<const value_type>& u_isai_batch,
        const gko::batch_csr::UniformBatch<const value_type>& mult_inv_batch,
        const gko::batch_csr::UniformBatch<const value_type>&
            iter_mat_lower_solve_batch,
        const gko::batch_csr::UniformBatch<const value_type>&
            iter_mat_upper_solve_batch,
        const enum gko::preconditioner::batch_ilu_isai_apply& apply_type,
        const int& num_relaxation_steps)
        : l_batch_{l_batch},
          u_batch_{u_batch},
          l_isai_batch_{l_isai_batch},
          u_isai_batch_{u_isai_batch},
          mult_inv_batch_{mult_inv_batch},
          iter_mat_lower_solve_batch_{iter_mat_lower_solve_batch},
          iter_mat_upper_solve_batch_{iter_mat_upper_solve_batch},
          apply_type_{apply_type},
          num_relaxation_steps_{num_relaxation_steps}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application (for application, returns the max. of
     * what is required by each of the 4 methods))
     */
    __dpct_inline__ static constexpr int dynamic_work_size(int nrows, int nnz)
    {
        return 3 * nrows;
    }

    /**
     * Complete the precond generation process.
     *
     */
    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work, item_ct1);
    }

    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work, item_ct1);
    }

    __dpct_inline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id, work, item_ct1);
    }


    __dpct_inline__ void apply(const int num_rows,
                               const ValueType* const __restrict__ r,
                               ValueType* const __restrict__ z,
                               sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r
        if (apply_type_ ==
            gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple) {
            apply_spmv_isai_simple(num_rows, r, z, item_ct1);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      spmv_isai_with_spgemm) {
            apply_spmv_isai_with_spgemm(num_rows, r, z, item_ct1);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      relaxation_steps_isai_simple) {
            apply_relaxation_steps_isai_simple(num_rows, r, z, item_ct1);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      relaxation_steps_isai_with_spgemm) {
            apply_relaxation_steps_isai_with_spgemm(num_rows, r, z, item_ct1);

        } else {
            //            printf("\n No such case: %d and %s ", __LINE__,
            //            __FILE__); //TODO
        }
    }

private:
    value_type* work_;
    const enum gko::preconditioner::batch_ilu_isai_apply apply_type_;
    const int num_relaxation_steps_;
    const gko::batch_csr::UniformBatch<const value_type> l_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_entry_;
    const gko::batch_csr::UniformBatch<const value_type> u_batch_;
    gko::batch_csr::BatchEntry<const value_type> u_entry_;
    const gko::batch_csr::UniformBatch<const value_type> l_isai_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_isai_entry_;
    const gko::batch_csr::UniformBatch<const value_type> u_isai_batch_;
    gko::batch_csr::BatchEntry<const value_type> u_isai_entry_;
    const gko::batch_csr::UniformBatch<const value_type> mult_inv_batch_;
    gko::batch_csr::BatchEntry<const value_type> mult_inv_entry_;
    const gko::batch_csr::UniformBatch<const value_type>
        iter_mat_lower_solve_batch_;
    gko::batch_csr::BatchEntry<const value_type> iter_mat_lower_solve_entry_;
    const gko::batch_csr::UniformBatch<const value_type>
        iter_mat_upper_solve_batch_;
    gko::batch_csr::BatchEntry<const value_type> iter_mat_upper_solve_entry_;
};


template <typename ValueType>
__dpct_inline__ void batch_ilu_isai_apply(batch_ilu_isai<ValueType> prec,
                                          const int nrows,
                                          const ValueType* const b,
                                          ValueType* const x,
                                          ValueType* const slm_values_ptr,
                                          sycl::nd_item<3> item_ct1)
{
    auto batch_id = item_ct1.get_group_linear_id();
    prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                  slm_values_ptr, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    prec.apply(nrows, b, x, item_ct1);
}

#endif  // GKO_DPCPP_PRECONDITIONER_BATCH_ILU_ISAI_HPP_INC_
