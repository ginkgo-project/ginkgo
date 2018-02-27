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
#include "core/base/utils.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"
#include "core/preconditioner/block_jacobi_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace {


template <typename... TArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(convert_to_dense,
                           block_jacobi::convert_to_dense<TArgs...>);
    GKO_REGISTER_OPERATION(simple_apply, block_jacobi::simple_apply<TArgs...>);
    GKO_REGISTER_OPERATION(apply, block_jacobi::apply<TArgs...>);
    GKO_REGISTER_OPERATION(find_blocks, block_jacobi::find_blocks<TArgs...>);
    GKO_REGISTER_OPERATION(generate, block_jacobi::generate<TArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_simple_apply_operation(
            this->num_blocks_, this->max_block_size_, this->max_block_size_,
            this->block_pointers_, this->blocks_, as<dense>(b), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::apply(const LinOp *alpha,
                                              const LinOp *b, const LinOp *beta,
                                              LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_apply_operation(
            this->num_blocks_, this->max_block_size_, this->max_block_size_,
            this->block_pointers_, this->blocks_, as<dense>(alpha),
            as<dense>(b), as<dense>(beta), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::convert_to(
    matrix::Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = matrix::Dense<ValueType>::create(exec, this->get_num_rows(),
                                                this->get_num_cols());
    exec->run(
        TemplatedOperation<ValueType, IndexType>::
            make_convert_to_dense_operation(
                this->num_blocks_, this->block_pointers_, this->blocks_,
                this->max_block_size_, tmp->get_values(), tmp->get_stride()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::move_to(
    matrix::Dense<ValueType> *result)
{
    this->convert_to(result);  // no special optimization possible here
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    ASSERT_EQUAL_DIMENSIONS(system_matrix, size(system_matrix->get_num_cols(),
                                                system_matrix->get_num_rows()));
    using csr = matrix::Csr<ValueType, IndexType>;
    std::unique_ptr<csr> csr_mtx_handle{};
    const csr *csr_mtx;
    auto exec = this->get_executor();
    auto ptr = dynamic_cast<const csr *>(system_matrix);
    if (ptr != nullptr && ptr->get_executor() == exec) {
        // use the matrix as is
        csr_mtx = ptr;
    } else {
        // convert it and bring it to the right executor
        csr_mtx_handle = csr::create(exec);
        csr_mtx_handle->copy_from(system_matrix);
        csr_mtx = csr_mtx_handle.get();
    }
    if (this->block_pointers_.get_data() == nullptr) {
        this->block_pointers_.resize_and_reset(csr_mtx->get_num_rows());
        exec->run(TemplatedOperation<ValueType, IndexType>::
                      make_find_blocks_operation(csr_mtx, this->max_block_size_,
                                                 this->num_blocks_,
                                                 this->block_pointers_));
    }
    exec->run(TemplatedOperation<ValueType, IndexType>::make_generate_operation(
        csr_mtx, this->num_blocks_, this->max_block_size_, this->get_stride(),
        this->block_pointers_, this->blocks_));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BlockJacobiFactory<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    return BlockJacobi<ValueType, IndexType>::create(
        this->get_executor(), base.get(), this->max_block_size_,
        this->block_pointers_);
}


template <typename ValueType, typename IndexType>
void AdaptiveBlockJacobi<ValueType, IndexType>::apply(const LinOp *b,
                                                      LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    using dense = matrix::Dense<ValueType>;
    // TODO: launch adaptive version of apply
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_simple_apply_operation(
            this->num_blocks_, this->max_block_size_, this->max_block_size_,
            this->block_pointers_, this->blocks_, as<dense>(b), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void AdaptiveBlockJacobi<ValueType, IndexType>::apply(const LinOp *alpha,
                                                      const LinOp *b,
                                                      const LinOp *beta,
                                                      LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    using dense = matrix::Dense<ValueType>;
    // TODO: launch adaptive version of apply
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_apply_operation(
            this->num_blocks_, this->max_block_size_, this->max_block_size_,
            this->block_pointers_, this->blocks_, as<dense>(alpha),
            as<dense>(b), as<dense>(beta), as<dense>(x)));
}


template <typename ValueType, typename IndexType>
void AdaptiveBlockJacobi<ValueType, IndexType>::convert_to(
    matrix::Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = matrix::Dense<ValueType>::create(exec, this->get_num_rows(),
                                                this->get_num_cols());
    // TODO: launch an adaptive version of the kernel
    exec->run(
        TemplatedOperation<ValueType, IndexType>::
            make_convert_to_dense_operation(
                this->num_blocks_, this->block_pointers_, this->blocks_,
                this->max_block_size_, tmp->get_values(), tmp->get_stride()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void AdaptiveBlockJacobi<ValueType, IndexType>::move_to(
    matrix::Dense<ValueType> *result)
{
    this->convert_to(result);  // no special optimization possible here
}


template <typename ValueType, typename IndexType>
void AdaptiveBlockJacobi<ValueType, IndexType>::generate(
    const LinOp *system_matrix)
{
    ASSERT_EQUAL_DIMENSIONS(system_matrix, size(system_matrix->get_num_cols(),
                                                system_matrix->get_num_rows()));
    using csr = matrix::Csr<ValueType, IndexType>;
    std::unique_ptr<csr> csr_mtx_handle{};
    const csr *csr_mtx;
    auto exec = this->get_executor();
    if (auto ptr = dynamic_cast<const csr *>(system_matrix)) {
        // use the matrix as is if it's already in CSR
        csr_mtx = ptr;
    } else {
        // otherwise, try to convert it
        csr_mtx_handle = csr::create(exec);
        as<ConvertibleTo<csr>>(system_matrix)->convert_to(csr_mtx_handle.get());
        csr_mtx = csr_mtx_handle.get();
    }
    if (this->block_pointers_.get_data() == nullptr) {
        this->block_pointers_.resize_and_reset(csr_mtx->get_num_rows());
        exec->run(TemplatedOperation<ValueType, IndexType>::
                      make_find_blocks_operation(csr_mtx, this->max_block_size_,
                                                 this->num_blocks_,
                                                 this->block_pointers_));
    }
    if (this->block_precisions_.get_data() == nullptr) {
        this->block_precisions_.resize_and_reset(this->num_blocks_);
        // TODO: launch a kernel to initialize block precisions
    }
    // TODO: replace with adaptive version of the kernel
    exec->run(TemplatedOperation<ValueType, IndexType>::make_generate_operation(
        csr_mtx, this->num_blocks_, this->max_block_size_, this->get_stride(),
        this->block_pointers_, this->blocks_));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp>
AdaptiveBlockJacobiFactory<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    return AdaptiveBlockJacobi<ValueType, IndexType>::create(
        this->get_executor(), base.get(), this->max_block_size_,
        this->block_pointers_, this->block_precisions_);
}


#define GKO_DECLARE_BLOCK_JACOBI_FACTORY(ValueType, IndexType) \
    class BlockJacobiFactory<ValueType, IndexType>
#define GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_FACTORY(ValueType, IndexType) \
    class AdaptiveBlockJacobiFactory<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BLOCK_JACOBI_FACTORY);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_FACTORY);
#undef GKO_DECLARE_BLOCK_JACOBI_FACTORY
#undef GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_FACTORY


}  // namespace preconditioner
}  // namespace gko
