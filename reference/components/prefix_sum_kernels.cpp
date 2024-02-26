// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename IndexType>
void prefix_sum_nonnegative(std::shared_ptr<const ReferenceExecutor> exec,
                            IndexType* counts, size_type num_entries)
{
    constexpr auto max = std::numeric_limits<IndexType>::max();
    IndexType partial_sum{};
    for (size_type i = 0; i < num_entries; ++i) {
        auto nnz = i < num_entries - 1 ? counts[i] : IndexType{};
        counts[i] = partial_sum;
        if (max - partial_sum < nnz) {
            throw OverflowError(
                __FILE__, __LINE__,
                name_demangling::get_type_name(typeid(IndexType)));
        }
        partial_sum += nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL(size_type);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
