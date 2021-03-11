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

#include <ginkgo/core/preconditioner/ras.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/preconditioner/ras_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace ras {


GKO_REGISTER_OPERATION(simple_apply, ras::simple_apply);
GKO_REGISTER_OPERATION(apply, ras::apply);
GKO_REGISTER_OPERATION(find_blocks, ras::find_blocks);
GKO_REGISTER_OPERATION(generate, ras::generate);
GKO_REGISTER_OPERATION(transpose_ras, ras::transpose_ras);
GKO_REGISTER_OPERATION(conj_transpose_ras, ras::conj_transpose_ras);
GKO_REGISTER_OPERATION(convert_to_dense, ras::convert_to_dense);
GKO_REGISTER_OPERATION(initialize_precisions, ras::initialize_precisions);


}  // namespace ras


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *b,
                                           LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
//    using dense = matrix::Dense<ValueType>;
//    this->get_executor()->run(ras::make_simple_apply(
//        num_blocks_, parameters_.max_block_size, storage_scheme_,
//        parameters_.storage_optimization.block_wise,
//        parameters_.block_pointers, blocks_, as<dense>(b), as<dense>(x)));
//}


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta,
                                           LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
//    using dense = matrix::Dense<ValueType>;
//    this->get_executor()->run(ras::make_apply(
//        num_blocks_, parameters_.max_block_size, storage_scheme_,
//        parameters_.storage_optimization.block_wise,
//        parameters_.block_pointers, blocks_, as<dense>(alpha), as<dense>(b),
//        as<dense>(beta), as<dense>(x)));
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;
//{
//    auto res = std::unique_ptr<Ras<ValueType, IndexType>>(
//        new Ras<ValueType, IndexType>(this->get_executor()));
//    // Ras enforces square matrices, so no dim transposition necessary
//    res->set_size(this->get_size());
//    res->storage_scheme_ = storage_scheme_;
//    res->num_blocks_ = num_blocks_;
//    res->blocks_.resize_and_reset(blocks_.get_num_elems());
//    res->conditioning_ = conditioning_;
//    res->parameters_ = parameters_;
//    this->get_executor()->run(ras::make_transpose_ras(
//        num_blocks_, parameters_.max_block_size,
//        parameters_.storage_optimization.block_wise,
//        parameters_.block_pointers, blocks_, storage_scheme_, res->blocks_));
//
//    return std::move(res);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;
//{
//    auto res = std::unique_ptr<Ras<ValueType, IndexType>>(
//        new Ras<ValueType, IndexType>(this->get_executor()));
//    // Ras enforces square matrices, so no dim transposition necessary
//    res->set_size(this->get_size());
//    res->storage_scheme_ = storage_scheme_;
//    res->num_blocks_ = num_blocks_;
//    res->blocks_.resize_and_reset(blocks_.get_num_elems());
//    res->conditioning_ = conditioning_;
//    res->parameters_ = parameters_;
//    this->get_executor()->run(ras::make_conj_transpose_ras(
//        num_blocks_, parameters_.max_block_size,
//        parameters_.storage_optimization.block_wise,
//        parameters_.block_pointers, blocks_, storage_scheme_, res->blocks_));
//
//    return std::move(res);
//}


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto num_subdomains = this->get_num_subdomains();
}


#define GKO_DECLARE_RAS(ValueType, IndexType) class Ras<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RAS);


}  // namespace preconditioner
}  // namespace gko
