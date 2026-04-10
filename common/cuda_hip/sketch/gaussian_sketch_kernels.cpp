// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/gaussian_sketch_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/randlib_bindings.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace gaussian_sketch {


template <typename ValueType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              matrix::view::dense<ValueType> sketch_matrix, uint64 seed)
{
    auto k = sketch_matrix.size[0];
    auto m = sketch_matrix.size[1];
    auto scale = 1.0 / sqrt(static_cast<double>(k));
    auto gen = randlib::rand_generator(seed, RANDLIB_RNG_PSEUDO_DEFAULT,
                                       exec->get_stream());
    randlib::rand_vector(
        gen, k * m, zero<remove_complex<ValueType>>(),
        static_cast<remove_complex<ValueType>>(scale), sketch_matrix.values);
    randlib::destroy(gen);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GAUSSIAN_SKETCH_GENERATE);


}  // namespace gaussian_sketch
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
