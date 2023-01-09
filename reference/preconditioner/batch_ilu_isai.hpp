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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {

namespace batch_ilu_isai_temp {

#include "reference/matrix/batch_csr_kernels.hpp.inc"

}

/**
 * Batch ilu-isai preconditioner.
 */
template <typename ValueType>
class batch_ilu_isai final {
private:
    inline void common_generate_for_all_system_matrix_types(
        size_type batch_id, ValueType* const work)
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


    inline void apply_spmv_isai_simple(
        const gko::batch_dense::BatchEntry<const ValueType>& r,
        const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = precond * r  ==> L * U * z = r ==> lai_U * lai_L * L * U * z =
        // lai_U * laiL * r ===> z = lai_U * laiL * r

        const gko::batch_dense::BatchEntry<ValueType> work_entry{work_, 1,
                                                                 r.num_rows, 1};

        batch_ilu_isai_temp::matvec_kernel(l_isai_entry_, r, work_entry);

        batch_ilu_isai_temp::matvec_kernel(u_isai_entry_,
                                           gko::batch::to_const(work_entry), z);
    }

    inline void apply_spmv_isai_with_spgemm(
        const gko::batch_dense::BatchEntry<const ValueType>& r,
        const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = precond * r  ==> L * U * z = r ==> lai_U * lai_L * L * U * z =
        // lai_U * laiL * r ===> z = lai_U * laiL * r
        batch_ilu_isai_temp::matvec_kernel(mult_inv_entry_, r, z);
    }

    inline void solve_system_relaxation_steps_simple(
        const gko::batch_csr::BatchEntry<const ValueType>& mat_entry,
        const gko::batch_csr::BatchEntry<const ValueType>& mat_isai_entry,
        const gko::batch_dense::BatchEntry<const ValueType>& rhs,
        const gko::batch_dense::BatchEntry<ValueType>& sol_guess,
        const gko::batch_dense::BatchEntry<ValueType>& temp1,
        const gko::batch_dense::BatchEntry<ValueType>& temp2,
        const gko::batch_dense::BatchEntry<ValueType>& temp3) const
    {
        // sol_guess_updated = lai_mat * rhs + (I - lai_mat * mat) * sol_guess
        // (iterations)

        batch_ilu_isai_temp::matvec_kernel(mat_isai_entry, rhs, temp1);

        for (int i = 0; i < sol_guess.num_rows; i++) {
            sol_guess.values[i] = 0;
        }

        for (int step = 0; step < num_relaxation_steps_; step++) {
            // sol_guess_updated =  temp1 + sol_guess -  (lai_mat * temp2) where
            // temp2 = mat * sol_guess
            // ===> sol_guess_updated =  temp1 + sol_guess -  temp3  where temp3
            // = lai_mat * temp2

            batch_ilu_isai_temp::matvec_kernel(
                mat_entry, gko::batch::to_const(sol_guess), temp2);
            batch_ilu_isai_temp::matvec_kernel(
                mat_isai_entry, gko::batch::to_const(temp2), temp3);

            for (int i = 0; i < sol_guess.num_rows; i++) {
                sol_guess.values[i] =
                    temp1.values[i] + sol_guess.values[i] - temp3.values[i];
            }
        }
    }


    inline void solve_system_relaxation_steps_external_spgemm(
        const gko::batch_csr::BatchEntry<const ValueType>& iteration_mat_entry,
        const gko::batch_csr::BatchEntry<const ValueType>& mat_isai_entry,
        const gko::batch_dense::BatchEntry<const ValueType>& rhs,
        const gko::batch_dense::BatchEntry<ValueType>& sol_guess,
        const gko::batch_dense::BatchEntry<ValueType>& temp1,
        const gko::batch_dense::BatchEntry<ValueType>& temp2) const
    {
        // sol_guess_updated = lai_mat * rhs + (I - lai_mat * mat) * sol_guess
        // (iterations)

        batch_ilu_isai_temp::matvec_kernel(mat_isai_entry, rhs, temp1);

        for (int i = 0; i < sol_guess.num_rows; i++) {
            sol_guess.values[i] = 0;
        }

        for (int step = 0; step < num_relaxation_steps_; step++) {
            // sol_guess_updated =  temp1 + iteration_mat * sol_guess

            batch_ilu_isai_temp::matvec_kernel(
                iteration_mat_entry, gko::batch::to_const(sol_guess), temp2);

            for (int i = 0; i < sol_guess.num_rows; i++) {
                sol_guess.values[i] = temp1.values[i] + temp2.values[i];
            }
        }
    }


    inline void apply_relaxation_steps_isai_simple(
        const gko::batch_dense::BatchEntry<const ValueType>& r,
        const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = precond * r
        // L * U * z = r
        // L * y = r  and then U * z = y
        // y_updated = lai_L * r + (I - lai_L * L) * y_old    (iterations)
        // Once y is obtained, z_updated = lai_U * y + (I - lai_U * U) * z_old
        // (iterations)
        const int num_rows = r.num_rows;
        ValueType* y_arr = work_;
        ValueType* temp1_arr = work_ + num_rows;
        ValueType* temp2_arr = temp1_arr + num_rows;
        ValueType* temp3_arr = temp2_arr + num_rows;
        const gko::batch_dense::BatchEntry<ValueType> y{y_arr, 1, num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> temp1{temp1_arr, 1,
                                                            num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> temp2{temp2_arr, 1,
                                                            num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> temp3{temp3_arr, 1,
                                                            num_rows, 1};

        solve_system_relaxation_steps_simple(l_entry_, l_isai_entry_, r, y,
                                             temp1, temp2, temp3);
        solve_system_relaxation_steps_simple(u_entry_, u_isai_entry_,
                                             gko::batch::to_const(y), z, temp1,
                                             temp2, temp3);
    }

    inline void apply_relaxation_steps_isai_with_spgemm(
        const gko::batch_dense::BatchEntry<const ValueType>& r,
        const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = precond * r
        // L * U * z = r
        // L * y = r  and then U * z = y
        // y_updated = lai_L * r + (I - lai_L * L) * y_old    (iterations)
        // Once y is obtained, z_updated = lai_U * y + (I - lai_U * U) * z_old
        // (iterations)

        const int num_rows = r.num_rows;
        ValueType* y_arr = work_;
        ValueType* temp1_arr = work_ + num_rows;
        ValueType* temp2_arr = temp1_arr + num_rows;
        const gko::batch_dense::BatchEntry<ValueType> y{y_arr, 1, num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> temp1{temp1_arr, 1,
                                                            num_rows, 1};
        const gko::batch_dense::BatchEntry<ValueType> temp2{temp2_arr, 1,
                                                            num_rows, 1};

        solve_system_relaxation_steps_external_spgemm(
            iter_mat_lower_solve_entry_, l_isai_entry_, r, y, temp1, temp2);
        solve_system_relaxation_steps_external_spgemm(
            iter_mat_upper_solve_entry_, u_isai_entry_, gko::batch::to_const(y),
            z, temp1, temp2);
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
    static constexpr int dynamic_work_size(int nrows, int) { return 4 * nrows; }

    /**
     * Complete the precond generation process.
     *
     */
    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const work)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const work)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const work)
    {
        common_generate_for_all_system_matrix_types(batch_id, work);
    }


    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = precond * r
        if (apply_type_ ==
            gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple) {
            apply_spmv_isai_simple(r, z);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      spmv_isai_with_spgemm) {
            apply_spmv_isai_with_spgemm(r, z);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      relaxation_steps_isai_simple) {
            apply_relaxation_steps_isai_simple(r, z);

        } else if (apply_type_ == gko::preconditioner::batch_ilu_isai_apply::
                                      relaxation_steps_isai_with_spgemm) {
            apply_relaxation_steps_isai_with_spgemm(r, z);

        } else {
            GKO_NOT_IMPLEMENTED;
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


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
