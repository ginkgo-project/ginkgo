// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_PRECONDITIONER_SOR_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_SOR_KERNELS_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L(ValueType, IndexType)        \
    void initialize_weighted_l(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        matrix::view::csr<const ValueType, const IndexType> system_matrix, \
        remove_complex<ValueType> weight,                                  \
        matrix::view::csr<ValueType, IndexType> l_factor)


#define GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U(ValueType, IndexType)      \
    void initialize_weighted_l_u(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        matrix::view::csr<const ValueType, const IndexType> system_matrix, \
        remove_complex<ValueType> weight,                                  \
        matrix::view::csr<ValueType, IndexType> l_factor,                  \
        matrix::view::csr<ValueType, IndexType> u_factor)


#define GKO_DECLARE_ALL_AS_TEMPLATES                             \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(sor, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_PRECONDITIONER_SOR_KERNELS_HPP_
