// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SKETCH_COUNT_SKETCH_KERNELS_HPP_
#define GKO_CORE_SKETCH_COUNT_SKETCH_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_COUNT_SKETCH_GENERATE(ValueType, IndexType)    \
    void generate(std::shared_ptr<const DefaultExecutor> exec,     \
                  size_type sketch_size,                            \
                  array<IndexType>& hash_map,                      \
                  array<ValueType>& signs, gko::uint64 seed)

#define GKO_DECLARE_COUNT_SKETCH_APPLY(ValueType, IndexType)       \
    void apply(std::shared_ptr<const DefaultExecutor> exec,        \
               const array<IndexType>& hash_map,                   \
               const array<ValueType>& signs,                      \
               matrix::view::dense<const ValueType> b,             \
               matrix::view::dense<ValueType> x)

#define GKO_DECLARE_COUNT_SKETCH_RAPPLY(ValueType, IndexType)      \
    void rapply(std::shared_ptr<const DefaultExecutor> exec,       \
                const array<IndexType>& hash_map,                  \
                const array<ValueType>& signs,                     \
                matrix::view::dense<const ValueType> b,            \
                matrix::view::dense<ValueType> x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_COUNT_SKETCH_GENERATE(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_COUNT_SKETCH_APPLY(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_COUNT_SKETCH_RAPPLY(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(count_sketch,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SKETCH_COUNT_SKETCH_KERNELS_HPP_
