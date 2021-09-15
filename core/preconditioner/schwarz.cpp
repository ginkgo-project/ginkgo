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

#include <ginkgo/core/preconditioner/schwarz.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/schwarz_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace schwarz {
namespace {


GKO_REGISTER_OPERATION(apply, schwarz::apply);
// GKO_REGISTER_OPERATION(advanced_apply, schwarz::advanced_apply);


}  // anonymous namespace
}  // namespace schwarz


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
    GKO_NOT_IMPLEMENTED;
//{
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_b, auto dense_x) {
//            if (parameters_.max_block_size == 1) {
//                this->get_executor()->run(schwarz::make_simple_scalar_apply(
//                    this->blocks_, dense_b, dense_x));
//            } else {
//                this->get_executor()->run(schwarz::make_simple_apply(
//                    num_blocks_, parameters_.max_block_size, storage_scheme_,
//                    parameters_.storage_optimization.block_wise,
//                    parameters_.block_pointers, blocks_, dense_b, dense_x));
//            }
//        },
//        b, x);
//}


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta,
    LinOp* x) const GKO_NOT_IMPLEMENTED;
//{
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x)
//        {
//            if (parameters_.max_block_size == 1) {
//                this->get_executor()->run(schwarz::make_scalar_apply(
//                    this->blocks_, dense_alpha, dense_b, dense_beta,
//                    dense_x));
//            } else {
//                this->get_executor()->run(schwarz::make_apply(
//                    num_blocks_, parameters_.max_block_size, storage_scheme_,
//                    parameters_.storage_optimization.block_wise,
//                    parameters_.block_pointers, blocks_, dense_alpha, dense_b,
//                    dense_beta, dense_x));
//            }
//        },
//        alpha, b, beta, x);
//}


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::write(mat_data& data) const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Schwarz<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Schwarz<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::generate(
    const LinOp* system_matrix, bool skip_sorting) GKO_NOT_IMPLEMENTED;


#define GKO_DECLARE_SCHWARZ(ValueType, IndexType) \
    class Schwarz<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace gko
