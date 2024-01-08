// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_PERMUTATION_KERNELS_HPP_
#define GKO_CORE_MATRIX_PERMUTATION_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PERMUTATION_INVERT_KERNEL(IndexType)              \
    void invert(std::shared_ptr<const DefaultExecutor> exec,          \
                const IndexType* permutation_indices, size_type size, \
                IndexType* inv_permutation)

#define GKO_DECLARE_PERMUTATION_COMPOSE_KERNEL(IndexType)             \
    void compose(std::shared_ptr<const DefaultExecutor> exec,         \
                 const IndexType* first_permutation,                  \
                 const IndexType* second_permutation, size_type size, \
                 IndexType* combined_permutation)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename IndexType>                     \
    GKO_DECLARE_PERMUTATION_INVERT_KERNEL(IndexType); \
    template <typename IndexType>                     \
    GKO_DECLARE_PERMUTATION_COMPOSE_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(permutation,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_PERMUTATION_KERNELS_HPP_
