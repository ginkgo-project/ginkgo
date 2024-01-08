// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/precision_conversion_kernels.hpp"


#include <algorithm>


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename SourceType, typename TargetType>
void convert_precision(std::shared_ptr<const DefaultExecutor> exec,
                       size_type size, const SourceType* in, TargetType* out)
{
    std::copy_n(in, size, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_CONVERT_PRECISION_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
