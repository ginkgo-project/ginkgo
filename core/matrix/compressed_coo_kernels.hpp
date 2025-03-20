// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_COMPRESSED_COO_KERNELS_HPP_
#define GKO_CORE_MATRIX_COMPRESSED_COO_KERNELS_HPP_


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/compressed_coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CRCOO_SPMV_KERNEL(ValueType, IndexType)         \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,          \
              const matrix::CompactRowCoo<ValueType, IndexType>* a, \
              const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)


#define GKO_DECLARE_CRCOCOCOO_SPMV_KERNEL(ValueType, IndexType)               \
    void spmv(                                                                \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::CompactRowCompressedColumnCoo<ValueType, IndexType>* a, \
        const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)


#define GKO_DECLARE_CRCOO_IDXS_TO_BITS_KERNEL(IndexType)                  \
    void idxs_to_bits(std::shared_ptr<const DefaultExecutor> exec,        \
                      const IndexType* idxs, size_type nnz, uint32* bits, \
                      IndexType* ranks)


#define GKO_DECLARE_CRCOO_BITS_TO_IDXS_KERNEL(IndexType)           \
    void bits_to_idxs(std::shared_ptr<const DefaultExecutor> exec, \
                      const uint32* bits, const IndexType* ranks,  \
                      size_type nnz, IndexType* idxs)


#define GKO_DECLARE_ALL_AS_TEMPLATES                         \
    template <typename ValueType, typename IndexType>        \
    GKO_DECLARE_CRCOO_SPMV_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>        \
    GKO_DECLARE_CRCOCOCOO_SPMV_KERNEL(ValueType, IndexType); \
    template <typename IndexType>                            \
    GKO_DECLARE_CRCOO_IDXS_TO_BITS_KERNEL(IndexType);        \
    template <typename IndexType>                            \
    GKO_DECLARE_CRCOO_BITS_TO_IDXS_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(compressed_coo,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COMPRESSED_COO_KERNELS_HPP_
