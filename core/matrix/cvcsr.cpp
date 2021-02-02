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

#include <ginkgo/core/matrix/cvcsr.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/matrix/cvcsr_kernels.hpp"


namespace gko {
namespace matrix {


namespace cvcsr {


GKO_REGISTER_OPERATION(spmv, cvcsr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, cvcsr::advanced_spmv);


}  // namespace cvcsr


template <typename ValueType, typename StorageType, typename IndexType>
void Cvcsr<ValueType, StorageType, IndexType>::apply_impl(const LinOp *b,
                                                          LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;

    if (dynamic_cast<const Dense<ValueType> *>(b)) {
        this->get_executor()->run(cvcsr::make_spmv(
            this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
    } else {
        auto dense_b = as<ComplexDense>(b);
        auto dense_x = as<ComplexDense>(x);
        this->apply(dense_b->create_real_view().get(),
                    dense_x->create_real_view().get());
    }
}


template <typename ValueType, typename StorageType, typename IndexType>
void Cvcsr<ValueType, StorageType, IndexType>::apply_impl(const LinOp *alpha,
                                                          const LinOp *b,
                                                          const LinOp *beta,
                                                          LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using RealDense = Dense<remove_complex<ValueType>>;

    if (dynamic_cast<const Dense<ValueType> *>(b)) {
        this->get_executor()->run(cvcsr::make_advanced_spmv(
            as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
            as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
    } else {
        auto dense_b = as<ComplexDense>(b);
        auto dense_x = as<ComplexDense>(x);
        auto dense_alpha = as<RealDense>(alpha);
        auto dense_beta = as<RealDense>(beta);
        this->apply(dense_alpha, dense_b->create_real_view().get(), dense_beta,
                    dense_x->create_real_view().get());
    }
}


#define GKO_DECLARE_CVCSR_MATRIX(ValueType, StorageType, IndexType) \
    class Cvcsr<ValueType, StorageType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_STORAGE_AND_INDEX_TYPE(GKO_DECLARE_CVCSR_MATRIX);


}  // namespace matrix
}  // namespace gko
