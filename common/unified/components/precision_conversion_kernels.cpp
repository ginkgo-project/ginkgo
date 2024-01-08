// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/precision_conversion_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename SourceType, typename TargetType>
void convert_precision(std::shared_ptr<const DefaultExecutor> exec,
                       size_type size, const SourceType* in, TargetType* out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto in, auto out) { out[idx] = in[idx]; },
        size, in, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_CONVERT_PRECISION_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
