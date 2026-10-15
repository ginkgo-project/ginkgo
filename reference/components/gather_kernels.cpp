// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/gather_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType, typename IndexType>
void gather(std::shared_ptr<const DefaultExecutor> exec, size_type num_res,
            const ValueType* orig, const IndexType* gather_map,
            ValueType* result)
{
    for (size_type i = 0; i < num_res; ++i) {
        result[i] = orig[gather_map[i]];
    }
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_AND_INDEX_TYPE(GKO_DECLARE_GATHER_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
