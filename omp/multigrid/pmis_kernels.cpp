// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <random>

#include <ginkgo/core/base/exception_helpers.hpp>

namespace gko {
namespace kernels {
namespace omp {
namespace pmis {


template <typename ValueType>
void initialize_random_weight(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num, ValueType* weight)
{
    std::default_random_engine gen(42);
    std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
    for (size_type row = 0; row < num; row++) {
        weight[row] = dist(gen);
    }
}
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL);


}  // namespace pmis
}  // namespace omp
}  // namespace kernels
}  // namespace gko
