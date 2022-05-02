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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_TRSV_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_TRSV_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 * Exact batch triangular solves with separately stored L and U matrices.
 */
template <typename ValueType>
class batch_exact_trsv_split {
public:
    using value_type = ValueType;

    int dynamic_work_size(const int num_rows, const int nnz)
    {
        return 0;
        // for a more parallel variant that needs an intermediate vector:
        // return num_rows;
    }

    /*
     * Set local triangular factors.
     */
    void generate(const gko::batch_csr::BatchEntry<const ValueType>& l_factor,
                  const gko::batch_csr::BatchEntry<const ValueType>& u_factor)
    {
        l_factor_ = l_factor;
        u_factor_ = u_factor;
    }

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        for (int i = 0; i < l_factor_.num_rows; i++) {
            for (int j = 0; j < r.num_rhs; j++) {
                ValueType sum{};
                for (int iz = l_factor_.row_ptrs[i];
                     iz < l_factor_.row_ptrs[i + 1] - 1; iz++) {
                    ValueType val =
                        l_factor_.values[iz] *
                        z.values[l_factor_.col_idxs[iz] * r.stride + j];
                    sum += val;
                }
                z.values[i * z.stride + j] =
                    (r.values[i * r.stride + j] - sum) /
                    l_factor_.values[l_factor_.row_ptrs[i + 1] - 1];
            }
        }
        for (int i = u_factor_.num_rows - 1; i >= 0; i--) {
            for (int j = 0; j < r.num_rhs; j++) {
                ValueType sum{};
                for (int iz = u_factor_.row_ptrs[i] + 1;
                     iz < u_factor_.row_ptrs[i + 1]; iz++) {
                    ValueType val =
                        u_factor_.values[iz] *
                        z.values[u_factor_.col_idxs[iz] * z.stride + j];
                    sum += val;
                }
                z.values[i * z.stride + j] =
                    (z.values[i * z.stride + j] - sum) /
                    u_factor_.values[u_factor_.row_ptrs[i]];
            }
        }
    }

private:
    batch_csr::BatchEntry<const ValueType> l_factor_;
    batch_csr::BatchEntry<const ValueType> u_factor_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_TRSV_HPP_
