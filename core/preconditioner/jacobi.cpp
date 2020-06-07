/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"


namespace gko {
namespace preconditioner {
namespace jacobi {


GKO_REGISTER_OPERATION(simple_apply, jacobi::simple_apply);
GKO_REGISTER_OPERATION(apply, jacobi::apply);
GKO_REGISTER_OPERATION(find_blocks, jacobi::find_blocks);
GKO_REGISTER_OPERATION(generate, jacobi::generate);
GKO_REGISTER_OPERATION(transpose_jacobi, jacobi::transpose_jacobi);
GKO_REGISTER_OPERATION(conj_transpose_jacobi, jacobi::conj_transpose_jacobi);
GKO_REGISTER_OPERATION(convert_to_dense, jacobi::convert_to_dense);
GKO_REGISTER_OPERATION(initialize_precisions, jacobi::initialize_precisions);


}  // namespace jacobi


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(jacobi::make_simple_apply(
        num_blocks_, parameters_.max_block_size, storage_scheme_,
        parameters_.storage_optimization.block_wise, parameters_.block_pointers,
        blocks_, as<dense>(b), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp *alpha,
                                              const LinOp *b, const LinOp *beta,
                                              LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(jacobi::make_apply(
        num_blocks_, parameters_.max_block_size, storage_scheme_,
        parameters_.storage_optimization.block_wise, parameters_.block_pointers,
        blocks_, as<dense>(alpha), as<dense>(b), as<dense>(beta),
        as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::convert_to(
    matrix::Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = matrix::Dense<ValueType>::create(exec, this->get_size());
    exec->run(jacobi::make_convert_to_dense(
        num_blocks_, parameters_.storage_optimization.block_wise,
        parameters_.block_pointers, blocks_, storage_scheme_, tmp->get_values(),
        tmp->get_stride()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::move_to(matrix::Dense<ValueType> *result)
{
    this->convert_to(result);  // no special optimization possible here
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::write(mat_data &data) const
{
    auto local_clone =
        make_temporary_clone(this->get_executor()->get_master(), this);
    data = {local_clone->get_size(), {}};

    const auto ptrs = local_clone->parameters_.block_pointers.get_const_data();
    for (size_type block = 0; block < local_clone->get_num_blocks(); ++block) {
        const auto scheme = local_clone->get_storage_scheme();
        const auto group_data = local_clone->blocks_.get_const_data() +
                                scheme.get_group_offset(block);
        const auto block_size = ptrs[block + 1] - ptrs[block];
        const auto precisions = local_clone->parameters_.storage_optimization
                                    .block_wise.get_const_data();
        const auto prec =
            precisions ? precisions[block] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(ValueType, prec, {
            const auto block_data =
                reinterpret_cast<const resolved_precision *>(group_data) +
                scheme.get_block_offset(block);
            for (IndexType row = 0; row < block_size; ++row) {
                for (IndexType col = 0; col < block_size; ++col) {
                    data.nonzeros.emplace_back(
                        ptrs[block] + row, ptrs[block] + col,
                        static_cast<ValueType>(
                            block_data[row + col * scheme.get_stride()]));
                }
            }
        });
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Jacobi<ValueType, IndexType>::transpose() const
{
    auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
        new Jacobi<ValueType, IndexType>(this->get_executor()));
    res->set_size(this->get_size());
    res->storage_scheme_ = storage_scheme_;
    res->num_blocks_ = num_blocks_;
    res->blocks_.resize_and_reset(blocks_.get_num_elems());
    res->conditioning_ = conditioning_;
    res->parameters_ = parameters_;
    this->get_executor()->run(jacobi::make_transpose_jacobi(
        num_blocks_, parameters_.max_block_size,
        parameters_.storage_optimization.block_wise, parameters_.block_pointers,
        blocks_, storage_scheme_, res->blocks_));

    return res;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Jacobi<ValueType, IndexType>::conj_transpose() const
{
    auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
        new Jacobi<ValueType, IndexType>(this->get_executor()));
    res->set_size(this->get_size());
    res->storage_scheme_ = storage_scheme_;
    res->num_blocks_ = num_blocks_;
    res->blocks_.resize_and_reset(blocks_.get_num_elems());
    res->conditioning_ = conditioning_;
    res->parameters_ = parameters_;
    this->get_executor()->run(jacobi::make_conj_transpose_jacobi(
        num_blocks_, parameters_.max_block_size,
        parameters_.storage_optimization.block_wise, parameters_.block_pointers,
        blocks_, storage_scheme_, res->blocks_));

    return res;
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::detect_blocks(
    const matrix::Csr<ValueType, IndexType> *system_matrix)
{
    parameters_.block_pointers.resize_and_reset(system_matrix->get_size()[0] +
                                                1);
    this->get_executor()->run(
        jacobi::make_find_blocks(system_matrix, parameters_.max_block_size,
                                 num_blocks_, parameters_.block_pointers));
    blocks_.resize_and_reset(
        storage_scheme_.compute_storage_space(num_blocks_));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto csr_mtx = copy_and_convert_to<matrix::Csr<ValueType, IndexType>>(
        exec, system_matrix);

    if (parameters_.block_pointers.get_data() == nullptr) {
        this->detect_blocks(csr_mtx.get());
    }

    const auto all_block_opt = parameters_.storage_optimization.of_all_blocks;
    auto &precisions = parameters_.storage_optimization.block_wise;
    // if adaptive version is used, make sure that the precision array is of the
    // correct size by replicating it multiple times if needed
    if (parameters_.storage_optimization.is_block_wise ||
        all_block_opt != precision_reduction(0, 0)) {
        if (!parameters_.storage_optimization.is_block_wise) {
            precisions = gko::Array<precision_reduction>(exec, {all_block_opt});
        }
        Array<precision_reduction> tmp(
            exec, parameters_.block_pointers.get_num_elems() - 1);
        exec->run(jacobi::make_initialize_precisions(precisions, tmp));
        precisions = std::move(tmp);
        conditioning_.resize_and_reset(num_blocks_);
    }

    exec->run(jacobi::make_generate(
        csr_mtx.get(), num_blocks_, parameters_.max_block_size,
        parameters_.accuracy, storage_scheme_, conditioning_, precisions,
        parameters_.block_pointers, blocks_));
}


#define GKO_DECLARE_JACOBI(ValueType, IndexType) \
    class Jacobi<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI);


}  // namespace preconditioner
}  // namespace gko
