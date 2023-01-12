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

#ifndef GINKGO_PARTITION_HELPERS_KERNELS_HPP
#define GINKGO_PARTITION_HELPERS_KERNELS_HPP


#include <ginkgo/core/base/array.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START(_type) \
    void sort_by_range_start(                                    \
        std::shared_ptr<const DefaultExecutor> exec,             \
        array<_type>& range_start_ends,                          \
        array<experimental::distributed::comm_index_type>& part_ids)


#define GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES(_type)          \
    void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec, \
                                  array<_type>& range_start_ends,              \
                                  bool* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    template <typename GlobalIndexType>                                 \
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START(GlobalIndexType); \
    template <typename GlobalIndexType>                                 \
    GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES(GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(partition_helpers,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GINKGO_PARTITION_HELPERS_KERNELS_HPP
