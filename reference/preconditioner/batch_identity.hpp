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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 *  Identity preconditioner for batch solvers. ( To be able to have
 * unpreconditioned solves )
 */
template <typename ValueType>
class BatchIdentity final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 0;

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static int dynamic_work_size(int, int) { return 0; }


    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     *
     * @param mat  Matrix for which to build an Ideniity preconditioner.
     * @param work  A 'work-vector', which is unneecessary here as no
     * preconditioner values are to be stored.
     */
    void generate(size_type,
                  const gko::batch_csr::BatchEntry<const ValueType>& mat,
                  ValueType* const work)
    {}

    void generate(size_type,
                  const gko::batch_ell::BatchEntry<const ValueType>& mat,
                  ValueType* const work)
    {}

    void generate(size_type,
                  const gko::batch_dense::BatchEntry<const ValueType>& mat,
                  ValueType* const work)
    {}

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        for (int i = 0; i < r.num_rows; i++) {
            for (int j = 0; j < r.num_rhs; j++)
                z.values[i * z.stride + j] = r.values[i * r.stride + j];
        }
    }
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_
