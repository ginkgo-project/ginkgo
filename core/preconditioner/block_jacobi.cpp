/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/preconditioner/block_jacobi.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/extended_float.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"
#include "core/preconditioner/block_jacobi_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace {


template <typename... TArgs>
struct JacobiOperation {
    GKO_REGISTER_OPERATION(simple_apply, block_jacobi::simple_apply<TArgs...>);
    GKO_REGISTER_OPERATION(apply, block_jacobi::apply<TArgs...>);
    GKO_REGISTER_OPERATION(find_blocks, block_jacobi::find_blocks<TArgs...>);
    GKO_REGISTER_OPERATION(generate, block_jacobi::generate<TArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense,
                           block_jacobi::convert_to_dense<TArgs...>);
    GKO_REGISTER_OPERATION(initialize_precisions,
                           block_jacobi::initialize_precisions);
};


}  // namespace


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(
        JacobiOperation<ValueType, IndexType>::make_simple_apply_operation(
            num_blocks_, parameters_.max_block_size, storage_scheme_,
            parameters_.block_precisions, parameters_.block_pointers, blocks_,
            as<dense>(b), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp *alpha,
                                              const LinOp *b, const LinOp *beta,
                                              LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(
        JacobiOperation<ValueType, IndexType>::make_apply_operation(
            num_blocks_, parameters_.max_block_size, storage_scheme_,
            parameters_.block_precisions, parameters_.block_pointers, blocks_,
            as<dense>(alpha), as<dense>(b), as<dense>(beta), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::convert_to(
    matrix::Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = matrix::Dense<ValueType>::create(exec, this->get_size());
    exec->run(
        JacobiOperation<ValueType, IndexType>::make_convert_to_dense_operation(
            num_blocks_, parameters_.block_precisions,
            parameters_.block_pointers, blocks_, storage_scheme_,
            tmp->get_values(), tmp->get_stride()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::move_to(matrix::Dense<ValueType> *result)
{
    this->convert_to(result);  // no special optimization possible here
}


#define RESOLVE_PRECISION(prec, ...)                            \
    if (prec == double_precision) {                             \
        using resolved_precision = ValueType;                   \
        __VA_ARGS__;                                            \
    } else if (prec == single_precision) {                      \
        using resolved_precision = reduce_precision<ValueType>; \
        __VA_ARGS__;                                            \
    } else if (prec == half_precision) {                        \
        using resolved_precision =                              \
            reduce_precision<reduce_precision<ValueType>>;      \
        __VA_ARGS__;                                            \
    } else {                                                    \
        throw NOT_SUPPORTED(best_precision);                    \
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
        const auto precisions =
            local_clone->parameters_.block_precisions.get_const_data();
        const auto prec =
            precisions != nullptr ? precisions[block] : double_precision;
        RESOLVE_PRECISION(prec, {
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
void Jacobi<ValueType, IndexType>::detect_blocks(
    const matrix::Csr<ValueType, IndexType> *system_matrix)
{
    parameters_.block_pointers.resize_and_reset(system_matrix->get_size()[0] +
                                                1);
    this->get_executor()->run(
        JacobiOperation<ValueType, IndexType>::make_find_blocks_operation(
            system_matrix, parameters_.max_block_size, num_blocks_,
            parameters_.block_pointers));
    blocks_.resize_and_reset(
        storage_scheme_.compute_storage_space(num_blocks_));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    ASSERT_EQUAL_DIMENSIONS(system_matrix,
                            transpose(system_matrix->get_size()));
    const auto exec = this->get_executor();
    const auto csr_mtx = copy_and_convert_to<matrix::Csr<ValueType, IndexType>>(
        exec, system_matrix);

    if (parameters_.block_pointers.get_data() == nullptr) {
        this->detect_blocks(csr_mtx.get());
    }

    // if adaptive version is used, make sure that the precision array is of the
    // correct size by replicating it multiple times if needed
    if (parameters_.block_precisions.get_data() != nullptr &&
        parameters_.block_precisions.get_num_elems() <
            parameters_.block_pointers.get_num_elems() - 1) {
        Array<precision> tmp(exec,
                             parameters_.block_pointers.get_num_elems() - 1);
        exec->run(JacobiOperation<ValueType, IndexType>::
                      make_initialize_precisions_operation(
                          parameters_.block_precisions, tmp));
        parameters_.block_precisions = std::move(tmp);
    }

    exec->run(JacobiOperation<ValueType, IndexType>::make_generate_operation(
        csr_mtx.get(), num_blocks_, parameters_.max_block_size, storage_scheme_,
        parameters_.block_precisions, parameters_.block_pointers, blocks_));
}


#define GKO_DECLARE_JACOBI(ValueType, IndexType) \
    class Jacobi<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI);
#undef GKO_DECLARE_JACOBI


}  // namespace preconditioner
}  // namespace gko
