// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"
#include "core/components/bit_packed_storage.hpp"


namespace gko {
namespace kernels {

#define GKO_DECLARE_ROW_SCATTER_SIMPLE_APPLY(_vtype, _otype, _itype) \
    void row_scatter(std::shared_ptr<const DefaultExecutor> exec,    \
                     const array<_itype>* gather_indices,            \
                     const matrix::Dense<_vtype>* orig,              \
                     matrix::Dense<_otype>* target, bool& invalid_access)

#define GKO_DECLARE_ROW_SCATTER_ADVANCED_APPLY(_vtype, _otype, _itype)        \
    void advanced_row_scatter(                                                \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const array<_itype>* row_idxs, const matrix::Dense<_vtype>* alpha,    \
        const matrix::Dense<_vtype>* orig, const matrix::Dense<_otype>* beta, \
        matrix::Dense<_otype>* target,                                        \
        bit_packed_span<bool, _itype, uint32> mask, bool& invalid_access)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename ValueType, typename OutputType, typename IndexType>  \
    GKO_DECLARE_ROW_SCATTER_SIMPLE_APPLY(ValueType, OutputType, IndexType); \
    template <typename ValueType, typename OutputType, typename IndexType>  \
    GKO_DECLARE_ROW_SCATTER_ADVANCED_APPLY(ValueType, OutputType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(row_scatter,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko
