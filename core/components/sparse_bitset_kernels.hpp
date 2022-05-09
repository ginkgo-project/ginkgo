/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_COMPONENTS_SPARSE_BITSET_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_SPARSE_BITSET_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_SPARSE_BITSET_SORT_KERNEL(GlobalIndexType) \
    void sort(std::shared_ptr<const DefaultExecutor> exec,     \
              GlobalIndexType* indices, size_type size)


#define GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_KERNEL(GlobalIndexType) \
    void build_bitmap(std::shared_ptr<const DefaultExecutor> exec,     \
                      const GlobalIndexType* indices, size_type size,  \
                      uint32* bitmap, size_type num_blocks)


#define GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_RANKS_KERNEL(LocalIndexType) \
    void build_bitmap_ranks(std::shared_ptr<const DefaultExecutor> exec,    \
                            const uint32* bitmap, size_type num_blocks,     \
                            LocalIndexType* ranks)


#define GKO_DECLARE_SPARSE_BITSET_BUILD_MULTILEVEL_KERNEL(LocalIndexType,      \
                                                          GlobalIndexType)     \
    void build_multilevel(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const GlobalIndexType* values, size_type size, array<uint32>& bitmaps, \
        array<LocalIndexType>& ranks, int depth, GlobalIndexType* offsets)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename GlobalIndexType>                                  \
    GKO_DECLARE_SPARSE_BITSET_SORT_KERNEL(GlobalIndexType);              \
    template <typename GlobalIndexType>                                  \
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_KERNEL(GlobalIndexType);      \
    template <typename LocalIndexType>                                   \
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_RANKS_KERNEL(LocalIndexType); \
    template <typename LocalIndexType, typename GlobalIndexType>         \
    GKO_DECLARE_SPARSE_BITSET_BUILD_MULTILEVEL_KERNEL(LocalIndexType,    \
                                                      GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(sparse_bitset,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_SPARSE_BITSET_KERNELS_HPP_
