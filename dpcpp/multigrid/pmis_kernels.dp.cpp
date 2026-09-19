// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <sycl/sycl.hpp>

#include <ginkgo/core/base/exception_helpers.hpp>

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace pmis {


template <typename IndexType>
void add_at_indices(std::shared_ptr<const DefaultExecutor> exec, size_type num,
                    const IndexType* idxs, const IndexType* values,
                    IndexType* out)
{
    // idxs may repeat across neighbours, hence the atomic
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num), [=](sycl::item<1> item) {
            const auto i = item.get_linear_id();
            sycl::atomic_ref<IndexType, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                ref(out[idxs[i]]);
            ref.fetch_add(values ? values[i] : IndexType{1});
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PMIS_ADD_AT_INDICES);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
