// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/sketch/sparse_stack.hpp>

#include "core/sketch/sparse_stack_kernels.hpp"


namespace gko {
namespace sketch {
namespace sparse_stack {
namespace {


GKO_REGISTER_OPERATION(generate, sparse_stack::generate);
GKO_REGISTER_OPERATION(apply, sparse_stack::apply);
GKO_REGISTER_OPERATION(rapply, sparse_stack::rapply);


}  // anonymous namespace
}  // namespace sparse_stack


template <typename ValueType, typename IndexType>
SparseStack<ValueType, IndexType>::SparseStack(
    std::shared_ptr<const Executor> exec, size_type sketch_size,
    size_type input_size, size_type zeta, uint64 seed)
    : EnableLinOp<SparseStack, SketchOperator<ValueType>>(
          exec, dim<2>{sketch_size, input_size}),
      zeta_{zeta},
      hash_map_{exec, input_size * zeta},
      signs_{exec, input_size * zeta},
      seed_{seed}
{
    exec->run(sparse_stack::make_generate(sketch_size, input_size, zeta_,
                                          hash_map_, signs_, seed_));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<SparseStack<ValueType, IndexType>>
SparseStack<ValueType, IndexType>::create(std::shared_ptr<const Executor> exec,
                                          size_type sketch_size,
                                          size_type input_size, size_type zeta,
                                          uint64 seed)
{
    return std::unique_ptr<SparseStack>{
        new SparseStack(exec, sketch_size, input_size, zeta, seed)};
}

template <typename ValueType, typename IndexType>
std::unique_ptr<SparseStack<ValueType, IndexType>>
SparseStack<ValueType, IndexType>::create(std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<SparseStack>{new SparseStack(exec)};
}


template <typename ValueType, typename IndexType>
void SparseStack<ValueType, IndexType>::apply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    this->get_executor()->run(sparse_stack::make_apply(
        zeta_, hash_map_, signs_, b->get_const_device_view(),
        x->get_device_view()));
}


template <typename ValueType, typename IndexType>
void SparseStack<ValueType, IndexType>::rapply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    this->get_executor()->run(sparse_stack::make_rapply(
        zeta_, hash_map_, signs_, b->get_const_device_view(),
        x->get_device_view()));
}


#define GKO_DECLARE_SPARSE_STACK(ValueType, IndexType) \
    class SparseStack<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK);


}  // namespace sketch
}  // namespace gko
