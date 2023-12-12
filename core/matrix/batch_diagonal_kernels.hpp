// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_BATCH_DIAGONAL_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_DIAGONAL_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_diagonal.hpp>


// Copyright (c) 2017-2023, the Ginkgo authors
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_DIAGONAL_SIMPLE_APPLY_KERNEL(_vtype)     \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const batch::matrix::Diagonal<_vtype>* a,    \
                      const batch::MultiVector<_vtype>* b,         \
                      batch::MultiVector<_vtype>* c)

#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_DIAGONAL_SIMPLE_APPLY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_diagonal,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_DIAGONAL_KERNELS_HPP_
