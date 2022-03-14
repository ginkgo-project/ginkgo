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

#ifndef GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BUILD_DIAG_OFFDIAG(ValueType, LocalIndexType,      \
                                       GlobalIndexType)                \
    void build_diag_offdiag(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        const device_matrix_data<ValueType, GlobalIndexType>& input,   \
        const distributed::Partition<LocalIndexType, GlobalIndexType>* \
            row_partition,                                             \
        const distributed::Partition<LocalIndexType, GlobalIndexType>* \
            col_partition,                                             \
        comm_index_type local_part,                                    \
        device_matrix_data<ValueType, LocalIndexType>& diag_data,      \
        device_matrix_data<ValueType, LocalIndexType>& offdiag_data,   \
        Array<LocalIndexType>& local_gather_idxs,                      \
        comm_index_type* recv_offsets,                                 \
        Array<GlobalIndexType>& local_to_global_ghost)


#define GKO_DECLARE_ALL_AS_TEMPLATES                       \
    using comm_index_type = distributed::comm_index_type;  \
    template <typename ValueType, typename LocalIndexType, \
              typename GlobalIndexType>                    \
    GKO_DECLARE_BUILD_DIAG_OFFDIAG(ValueType, LocalIndexType, GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(distributed_matrix,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_
