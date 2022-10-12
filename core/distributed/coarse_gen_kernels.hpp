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

#ifndef GKO_CORE_DISTRIBUTED_COARSE_GEN_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_COARSE_GEN_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/device_matrix_data.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace coarse_gen {

#define GKO_DECLARE_COARSE_GEN_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType) \
    void find_strongest_neighbor(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Csr<ValueType, IndexType>* weight_mtx_diag,            \
        const matrix::Csr<ValueType, IndexType>* weight_mtx_offdiag,         \
        const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg,      \
        array<IndexType>& strongest_neighbor)

#define GKO_DECLARE_COARSE_GEN_ASSIGN_TO_EXIST_AGG(ValueType, IndexType) \
    void assign_to_exist_agg(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* weight_mtx_diag,        \
        const matrix::Csr<ValueType, IndexType>* weight_mtx_offdiag,     \
        const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg,  \
        array<IndexType>& intermediate_agg)

#define GKO_DECLARE_COARSE_GEN_FILL_COARSE(ValueType, IndexType)          \
    void fill_coarse(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const device_matrix_data<ValueType, IndexType>& fine_matrix_data, \
        device_matrix_data<ValueType, IndexType>& coarse_data,            \
        array<IndexType>& coarse_indices)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_COARSE_GEN_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_COARSE_GEN_ASSIGN_TO_EXIST_AGG(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_COARSE_GEN_FILL_COARSE(ValueType, IndexType)


}  // namespace coarse_gen


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(coarse_gen,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_COARSE_GEN_KERNELS_HPP_
