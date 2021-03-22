/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_JACOBI_HPP
#define GKO_REFERENCE_PRECONDITIONER_BATCH_JACOBI_HPP


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * (Scalar) Jacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchJacobi final {
public:
    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = batch_config<ValueType>::max_num_rows;

    /**
     * The size of the work vector required in case of dynamic allocation.
     *
     * For the Jacobi preconditioner, this is unnecessary, but this function is
     * part of a 'batch preconditioner interface' because other preconditioners
     * may need it.
     */
    static int dynamic_work_size(int, int)
    {
        return batch_config<ValueType>::max_num_rows;
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by work_size or dynamic_work_size.
     */
    BatchJacobi(const gko::batch_csr::BatchEntry<const ValueType> &mat,
                ValueType *const work)
        : matrix_{mat}, work_{work}
    {
        for (int i = 0; i < matrix_.num_rows; i++) {
            for (int j = matrix_.row_ptrs[i]; j < matrix_.row_ptrs[i + 1];
                 j++) {
                if (matrix_.col_idxs[j] == i) {
                    work_[i] = one<ValueType>() / matrix_.values[j];
                    break;
                }
            }
        }
    }

    void apply(const gko::batch_dense::BatchEntry<const ValueType> &r,
               const gko::batch_dense::BatchEntry<ValueType> &z) const
    {
        for (int i = 0; i < matrix_.num_rows; i++) {
            for (int j = 0; j < r.num_rhs; j++) {
                z.values[i * z.stride + j] =
                    work_[i] * r.values[i * r.stride + j];
            }
        }
    }

private:
    ValueType *const work_;
    const gko::batch_csr::BatchEntry<const ValueType> &matrix_;
};


}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif
