// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_EIGENSOLVER_LOBPCG_KERNELS_HPP_
#define GKO_CORE_EIGENSOLVER_LOBPCG_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace lobpcg {


#define GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL(_type)                 \
    void symm_generalized_eig(                                                \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type>* a, \
        matrix::Dense<_type>* b, array<remove_complex<_type>>* e_vals,        \
        array<char>* workspace)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL(ValueType)


}  // namespace lobpcg


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(lobpcg, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_EIGENSOLVER_LOBPCG_KERNELS_HPP_
