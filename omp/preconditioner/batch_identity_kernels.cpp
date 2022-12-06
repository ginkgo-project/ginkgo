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

#include "core/preconditioner/batch_identity_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "omp/matrix/batch_struct.hpp"
#include "omp/preconditioner/batch_identity.hpp"


namespace gko {
namespace kernels {
namespace omp {


template <typename ValueType>
void batch_identity_apply(std::shared_ptr<const gko::OmpExecutor> exec,
                          const matrix::BatchCsr<ValueType> *const a,
                          const matrix::BatchDense<ValueType> *const b,
                          matrix::BatchDense<ValueType> *const x)
{
    const auto a_ub = get_batch_struct(a);
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const int local_size_bytes = BatchIdentity<ValueType>::dynamic_work_size(
                                     a_ub.num_rows, a_ub.num_nnz) *
                                 sizeof(ValueType);
    using byte = unsigned char;

#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        Array<byte> local_space(exec, local_size_bytes);
        BatchIdentity<ValueType> prec;

        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);

        const auto prec_work =
            reinterpret_cast<ValueType *>(local_space.get_data());
        prec.generate(a_b, prec_work);
        prec.apply(b_b, x_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDENTITY_KERNEL);


}  // namespace omp
}  // namespace kernels
}  // namespace gko
