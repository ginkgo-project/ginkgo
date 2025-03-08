// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_BITVECTOR_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_BITVECTOR_KERNELS_HPP_


#include "core/components/bitvector.hpp"

#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL(IndexType)         \
    void compute_bits_and_ranks(                                               \
        std::shared_ptr<const DefaultExecutor> exec, const IndexType* indices, \
        IndexType num_indices, IndexType size,                                 \
        typename device_bitvector<IndexType>::storage_type* bits,              \
        IndexType* ranks)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename IndexType>    \
    GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(bitvector,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_BITVECTOR_KERNELS_HPP_
