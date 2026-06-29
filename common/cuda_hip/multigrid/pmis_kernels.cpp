// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <random>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/randlib_bindings.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace pmis {


template <typename ValueType>
void initialize_random_weight(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num, ValueType* weight)
{
    auto gen = randlib::rand_generator(
        std::random_device{}(), RANDLIB_RNG_PSEUDO_DEFAULT, exec->get_stream());
    randlib::uniform_rand_vector(gen, num, weight);
    randlib::destroy(gen);
}
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
