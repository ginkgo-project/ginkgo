/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_
#define GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_


#include <ginkgo/core/base/index_set.hpp>


#include <memory>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL(IndexType)       \
    void compute_validity(std::shared_ptr<const DefaultExecutor> exec, \
                          const array<IndexType>* local_indices,       \
                          array<bool>* validity_array)

#define GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL(IndexType)       \
    void to_global_indices(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType num_subsets, const IndexType* subset_begin,     \
        const IndexType* subset_end, const IndexType* superset_indices, \
        IndexType* decomp_indices)

#define GKO_DECLARE_INDEX_SET_POPULATE_KERNEL(IndexType)                   \
    void populate_subsets(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const IndexType index_space_size, const array<IndexType>* indices, \
        array<IndexType>* subset_begin, array<IndexType>* subset_end,      \
        array<IndexType>* superset_indices, const bool is_sorted)

#define GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL(IndexType)         \
    void global_to_local(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType index_space_size, const IndexType num_subsets,  \
        const IndexType* subset_begin, const IndexType* subset_end,     \
        const IndexType* superset_indices, const IndexType num_indices, \
        const IndexType* global_indices, IndexType* local_indices,      \
        const bool is_sorted)

#define GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL(IndexType)         \
    void local_to_global(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType num_subsets, const IndexType* subset_begin,     \
        const IndexType* superset_indices, const IndexType num_indices, \
        const IndexType* local_indices, IndexType* global_indices,      \
        const bool is_sorted)


#define GKO_DECLARE_ALL_AS_TEMPLATES                           \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL(IndexType);  \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL(IndexType); \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_POPULATE_KERNEL(IndexType);          \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL(IndexType);   \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(idx_set, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_
