// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/gather_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType, typename IndexType>
void gather(std::shared_ptr<const DefaultExecutor> exec, size_type num_res,
            const ValueType* orig, const IndexType* gather_map,
            ValueType* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto orig, auto gather_map, auto result) {
            result[i] = orig[gather_map[i]];
        },
        num_res, orig, gather_map, result);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_AND_INDEX_TYPE(GKO_DECLARE_GATHER_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
