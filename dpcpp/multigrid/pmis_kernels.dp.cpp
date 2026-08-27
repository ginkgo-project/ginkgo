// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/random>

#include "core/multigrid/pmis_kernels.hpp"

#include <random>

#include <sycl/sycl.hpp>

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
    auto seed = std::random_device{}();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num), [=](sycl::item<1> idx) {
            std::uint64_t offset = idx.get_linear_id();
            oneapi::dpl::minstd_rand engine(seed, offset);
            oneapi::dpl::uniform_real_distribution<device_type<ValueType>>
                distr(0, 1);
            work[idx] = distr(engine);
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
