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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ISAI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ISAI_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {

namespace {

#include "reference/matrix/batch_csr_kernels.hpp.inc"

}

/**
 * Batch isai preconditioner.
 */
template <typename ValueType>
class batch_isai final {
private:
    inline void common_generate_for_all_system_matrix_types(size_type batch_id)
    {
        approx_inv_entry_ =
            gko::batch::batch_entry(approx_inv_batch_, batch_id);
    }

public:
    using value_type = ValueType;

    /**
     * @param approx_inv  Approximate inverse (that was factored externally)
     */
    batch_isai(const gko::batch_csr::UniformBatch<const value_type>& approx_inv)
        : approx_inv_batch_{approx_inv}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    static constexpr int dynamic_work_size(int nrows, int) { return 0; }

    /**
     * Complete the precond generation process.
     *
     */
    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }


    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // z = approx_inv * r (Spmv)
        matvec_kernel(approx_inv_entry_, r, z);
    }

private:
    const gko::batch_csr::UniformBatch<const value_type> approx_inv_batch_;
    gko::batch_csr::BatchEntry<const value_type> approx_inv_entry_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_ISAI_HPP_
