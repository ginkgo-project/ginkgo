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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 * Batch ilu0 preconditioner.
 */
template <typename ValueType>
class batch_ilu final {
private:
    inline void common_generate_for_all_system_matrix_types(
        size_type batch_id, ValueType* const work)
    {
        mat_factorized_entry_ =
            gko::batch::batch_entry(mat_factorized_batch_, batch_id);
        work_ = work;
    }

public:
    using value_type = ValueType;

    /**
     *
     * @param mat_factorized   Factorized matrix (that was factored externally).
     * @param csr_diag_locs  pointers to the diagonal entries in factorized
     * matrix
     */
    batch_ilu(
        const gko::batch_csr::UniformBatch<const value_type>& mat_factorized,
        const int* const csr_diag_locs)
        : mat_factorized_batch_{mat_factorized}, csr_diag_locs_{csr_diag_locs}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    static constexpr int dynamic_work_size(int nrows, int nnz) { return nrows; }


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
        // special trsv for the combined form
        /*
            z = prec * r
            L * U * z = r
            L * y = r (find y by lower trsv), then U * z = y (find z by upper
           trsv)
        */
        const auto nrows = mat_factorized_entry_.num_rows;
        const auto row_ptrs = mat_factorized_entry_.row_ptrs;
        const auto col_idxs = mat_factorized_entry_.col_idxs;
        const auto values = mat_factorized_entry_.values;

        // Lower Trsv  L * work = r
        for (int row_idx = 0; row_idx < nrows; row_idx++) {
            ValueType sum = zero<ValueType>();
            for (int i = row_ptrs[row_idx]; i < csr_diag_locs_[row_idx]; i++) {
                const int col_idx = col_idxs[i];
                sum += values[i] * work_[col_idx];
            }

            work_[row_idx] = (r.values[row_idx] - sum) / one<ValueType>();
        }

        // Upper Trsv U * z = work
        for (int row_idx = nrows - 1; row_idx >= 0; row_idx--) {
            ValueType sum = zero<ValueType>();

            for (int i = row_ptrs[row_idx + 1] - 1; i > csr_diag_locs_[row_idx];
                 i--) {
                const int col_idx = col_idxs[i];
                sum += values[i] * z.values[col_idx];
            }

            ValueType diag_ele = values[csr_diag_locs_[row_idx]];
            z.values[row_idx] = (work_[row_idx] - sum) / diag_ele;
        }
    }

private:
    const gko::batch_csr::UniformBatch<const value_type> mat_factorized_batch_;
    const int* const csr_diag_locs_;
    gko::batch_csr::BatchEntry<const value_type> mat_factorized_entry_;
    value_type* work_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_
