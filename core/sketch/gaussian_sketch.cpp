// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/sketch/gaussian_sketch.hpp>

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/sketch/gaussian_sketch_kernels.hpp"


namespace gko {
namespace sketch {
namespace gaussian_sketch {
namespace {


GKO_REGISTER_OPERATION(generate, gaussian_sketch::generate);


}  // anonymous namespace
}  // namespace gaussian_sketch


template <typename ValueType>
GaussianSketch<ValueType>::GaussianSketch(
    std::shared_ptr<const Executor> exec, size_type sketch_size,
    size_type input_size, uint64 seed)
    : EnableLinOp<GaussianSketch, SketchOperator<ValueType>>(exec, dim<2>{sketch_size, input_size}),
      seed_{seed}
{
    sketch_matrix_ = matrix::Dense<ValueType>::create(
        exec, dim<2>{sketch_size, input_size});
    exec->run(gaussian_sketch::make_generate(
        sketch_matrix_->get_device_view(), seed_));

    // Compute and store transpose for rapply
    sketch_matrix_t_ = matrix::Dense<ValueType>::create(
        exec, dim<2>{input_size, sketch_size});
    sketch_matrix_->transpose(sketch_matrix_t_);
}


template <typename ValueType>
std::unique_ptr<GaussianSketch<ValueType>> GaussianSketch<ValueType>::create(
    std::shared_ptr<const Executor> exec, size_type sketch_size,
    size_type input_size, uint64 seed)
{
    return std::unique_ptr<GaussianSketch>{
        new GaussianSketch(exec, sketch_size, input_size, seed)};
}


template <typename ValueType>
void GaussianSketch<ValueType>::apply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    sketch_matrix_->apply(b, x);
}


template <typename ValueType>
void GaussianSketch<ValueType>::rapply_sketch_impl(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    b->apply(sketch_matrix_t_, x);
}


#define GKO_DECLARE_GAUSSIAN_SKETCH(ValueType) class GaussianSketch<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GAUSSIAN_SKETCH);


}  // namespace sketch
}  // namespace gko
