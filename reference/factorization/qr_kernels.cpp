/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/factorization/par_ilu_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The qr factorization namespace.
 *
 * @ingroup factor
 */
namespace qr_factorization {


template <typename ValueType, typename IndexType>
void householder_generator(std::shared_ptr<const OmpExecutor> exec,
                           const matrix::Dense<ValueType> *vector,
                           const size_type index,
                           matrix::Dense<ValueType> *factor)
{
    // u = x(k:)
    // alpha = u_k/|u_k| * ||u||
    // u = u + alpha e_k
    // u = u / ||u|| (factor)
    // scalar = -2
    for (size_type i = 0; i < index - 1; i++) {
        factor.at(0, i) = 0;
    }
    for (size_type i = index; i < vector.get_size()[0]; i++) {
        factor.at(0, i) = vector.at(0, i)
    }
    auto norm = matrix::Dense<ValueType>::create(exec, dim<2>(1));
    factor->compute_norm2(lend(norm));
    auto alpha = factor.at(0, index) / std::abs(factor.at(0, index));
    factor.at(0, index) += alpha;
    factor->compute_norm2(lend(norm));
    norm.at(0, 0) = one<ValueType>() / norm.at(0, 0);
    factor->scale(lend(norm));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_QR_HOUSEHOLDER_GENERATOR_KERNEL);


}  // namespace qr_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
