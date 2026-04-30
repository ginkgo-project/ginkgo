// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/sparse_stack_kernels.hpp"
#include <ginkgo/core/base/exception_helpers.hpp>

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace sparse_stack {

template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              size_type sketch_size, size_type input_size, size_type zeta,
              array<IndexType>& hash_map, array<ValueType>& signs, uint64 seed)
    GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK_GENERATE);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec, size_type zeta,
           const array<IndexType>& hash_map, const array<ValueType>& signs,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<ValueType> x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK_APPLY);

template <typename ValueType, typename IndexType>
void rapply(std::shared_ptr<const DefaultExecutor> exec, size_type zeta,
            const array<IndexType>& hash_map, const array<ValueType>& signs,
            matrix::view::dense<const ValueType> b,
            matrix::view::dense<ValueType> x) GKO_NOT_IMPLEMENTED;
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK_RAPPLY);

}  // namespace sparse_stack
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
