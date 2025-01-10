// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "benchmark/utils/stencil_matrix.hpp"

#include "core/base/kernel_declaration.hpp"

namespace gko {
namespace kernels {


#define GKO_DECLARE_GENERATE_2D_STENCIL_BOX(ValueType, IndexType)             \
    gko::device_matrix_data<ValueType, IndexType> generate_2d_stencil_box(    \
        std::shared_ptr<const gko::Executor> exec, std::array<int, 2> dims,   \
        std::array<int, 2> positions, const gko::size_type target_local_size, \
        bool restricted)


#define GKO_DECLARE_GENERATE_3D_STENCIL_BOX(ValueType, IndexType)             \
    gko::device_matrix_data<ValueType, IndexType> generate_3d_stencil_box(    \
        std::shared_ptr<const gko::Executor> exec, std::array<int, 3> dims,   \
        std::array<int, 3> positions, const gko::size_type target_local_size, \
        bool restricted)


#define GKO_DECLARE_ALL_AS_TEMPLATES                           \
    template <typename ValueType, typename IndexType>          \
    GKO_DECLARE_GENERATE_2D_STENCIL_BOX(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>          \
    GKO_DECLARE_GENERATE_3D_STENCIL_BOX(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(stencil, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko
