// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/count_sketch.hpp>

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/sketch/count_sketch_kernels.hpp"


namespace gko {
namespace sketch {
namespace count_sketch {
namespace {


GKO_REGISTER_OPERATION(generate, count_sketch::generate);
GKO_REGISTER_OPERATION(apply, count_sketch::apply);
GKO_REGISTER_OPERATION(rapply, count_sketch::rapply);


}  // anonymous namespace
}  // namespace count_sketch


template <typename ValueType, typename IndexType>
CountSketch<ValueType, IndexType>::CountSketch(
    std::shared_ptr<const Executor> exec, size_type sketch_size,
    size_type input_size, uint64 seed)
    : EnableLinOp<CountSketch, SketchOperator<ValueType>>(exec, dim<2>{sketch_size, input_size}),
      hash_map_{exec, input_size},
      signs_{exec, input_size},
      seed_{seed}
{
    exec->run(count_sketch::make_generate(sketch_size, hash_map_, signs_,
                                          seed_));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<CountSketch<ValueType, IndexType>>
CountSketch<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, size_type sketch_size,
    size_type input_size, uint64 seed)
{
    return std::unique_ptr<CountSketch>{
        new CountSketch(exec, sketch_size, input_size, seed)};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<CountSketch<ValueType, IndexType>>
CountSketch<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<CountSketch>{new CountSketch(exec)};
}


template <typename ValueType, typename IndexType>
void CountSketch<ValueType, IndexType>::apply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    this->get_executor()->run(count_sketch::make_apply(
        hash_map_, signs_, b->get_const_device_view(),
        x->get_device_view()));
}


template <typename ValueType, typename IndexType>
void CountSketch<ValueType, IndexType>::rapply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    this->get_executor()->run(count_sketch::make_rapply(
        hash_map_, signs_, b->get_const_device_view(),
        x->get_device_view()));
}


#define GKO_DECLARE_COUNT_SKETCH(ValueType, IndexType) \
    class CountSketch<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COUNT_SKETCH);


}  // namespace sketch
}  // namespace gko
