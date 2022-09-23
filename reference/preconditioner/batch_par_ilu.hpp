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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_PAR_ILU_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_PAR_ILU_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


template <typename ValueType>
class batch_parilu0 final {
public:
    using value_type = ValueType;


    /**
     * @param l_batch  Lower triangular factor that was externally generated.
     * @param u_batch  Upper triangular factor that was externally generated.
     */
    batch_parilu0(const gko::batch_csr::UniformBatch<const ValueType>& l_batch,
                  const gko::batch_csr::UniformBatch<const ValueType>& u_batch,
                  const bool dummy = true)
        : l_batch_{l_batch}, u_batch_{u_batch}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    static constexpr int dynamic_work_size(int nrows, int nnz) { return nrows; }

    /**
     * Complete the precond generation process.
     *
     * @param mat  Matrix for which to build an ILU-type preconditioner.
     */
    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        const int num_rows = r.num_rows;
        // z = precond * r
        // L * U * z = r
        // L * work = r, U * z = work
        for (int row_index = 0; row_index < num_rows; row_index++) {
            ValueType sum = zero<ValueType>();
            for (int i = l_entry_.row_ptrs[row_index];
                 i < l_entry_.row_ptrs[row_index + 1] - 1; i++) {
                const int col_index = l_entry_.col_idxs[i];
                sum += l_entry_.values[i] * work_[col_index];
            }

            const ValueType diag_val =
                l_entry_.values[l_entry_.row_ptrs[row_index + 1] - 1];
            work_[row_index] = (r.values[row_index] - sum) / diag_val;
        }

        for (int row_index = num_rows - 1; row_index >= 0; row_index--) {
            ValueType sum = zero<ValueType>();
            for (int i = u_entry_.row_ptrs[row_index + 1] - 1;
                 i > u_entry_.row_ptrs[row_index]; i--) {
                const int col_index = u_entry_.col_idxs[i];
                sum += u_entry_.values[i] * z.values[col_index];
            }
            const ValueType diag_val =
                u_entry_.values[u_entry_.row_ptrs[row_index]];
            z.values[row_index] = (work_[row_index] - sum) / diag_val;
        }
    }

private:
    const gko::batch_csr::UniformBatch<const value_type> l_batch_;
    const gko::batch_csr::UniformBatch<const value_type> u_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_entry_;
    gko::batch_csr::BatchEntry<const value_type> u_entry_;
    ValueType* work_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_PAR_ILU_HPP_
