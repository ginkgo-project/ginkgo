/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_BUILD_DIAG_OFFDIAG(ValueType, LocalIndexType)            \
    void build_diag_offdiag(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const Array<matrix_data_entry<ValueType, global_index_type>>& input, \
        const distributed::Partition<LocalIndexType>* partition,             \
        comm_index_type local_part,                                          \
        Array<matrix_data_entry<ValueType, LocalIndexType>>& diag_data,      \
        Array<matrix_data_entry<ValueType, LocalIndexType>>& offdiag_data,   \
        Array<LocalIndexType>& local_gather_idxs,                            \
        comm_index_type* recv_offsets,                                       \
        Array<global_index_type>& local_row_to_global,                       \
        Array<global_index_type>& local_offdiag_col_to_global,               \
        ValueType deduction_help)


#define GKO_DECLARE_MAP_TO_GLOBAL_IDXS(SourceType, TargetType)           \
    void map_to_global_idxs(std::shared_ptr<const DefaultExecutor> exec, \
                            const SourceType* input, size_t n,           \
                            TargetType* output, const TargetType* map)

#define GKO_DECLARE_MERGE_DIAG_OFFDIAG(ValueType, LocalIndexType) \
    void merge_diag_offdiag(                                      \
        std::shared_ptr<const DefaultExecutor> exec,              \
        const matrix::Csr<ValueType, LocalIndexType>* diag,       \
        const matrix::Csr<ValueType, LocalIndexType>* offdiag,    \
        matrix::Csr<ValueType, LocalIndexType>* result)

#define GKO_DECLARE_COMBINE_LOCAL_MTXS(ValueType, LocalIndexType) \
    void combine_local_mtxs(                                      \
        std::shared_ptr<const DefaultExecutor> exec,              \
        const matrix::Csr<ValueType, LocalIndexType>* local,      \
        matrix::Csr<ValueType, LocalIndexType>* result)

#define GKO_DECLARE_CHECK_COLUMN_INDEX_EXISTS(IndexType)   \
    void check_column_index_exists(                        \
        std::shared_ptr<const DefaultExecutor> exec,       \
        const IndexType* col_idxs, size_type num_elements, \
        const global_index_type* map, Array<bool>& col_exists)

#define GKO_DECLARE_BUILD_RECV_SIZES(IndexType)                           \
    void build_recv_sizes(                                                \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const IndexType* col_idxs, size_type num_cols,                    \
        const distributed::Partition<IndexType>* partition,               \
        const global_index_type* map, Array<comm_index_type>& recv_sizes, \
        Array<comm_index_type>& recv_offsets, Array<IndexType>& recv_indices)

#define GKO_DECLARE_COMPRESS_OFFDIAG_DATA(ValueType, LocalIndexType) \
    void compress_offdiag_data(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                 \
        device_matrix_data<ValueType, LocalIndexType>& offdiag_data, \
        Array<global_index_type>& col_map)

#define GKO_DECLARE_CHECK_INDICES_WITHIN_SPAN(LocalIndexType)            \
    void check_indices_within_span(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const Array<LocalIndexType>& indices,                            \
        const Array<global_index_type>& to_global, gko::span valid_span, \
        Array<bool>& index_is_valid)

#define GKO_DECLARE_ZERO_OUT_INVALID_COLUMNS(ValueType, IndexType) \
    void zero_out_invalid_columns(                                 \
        std::shared_ptr<const DefaultExecutor> exec,               \
        const Array<bool>& column_index_is_valid,                  \
        device_matrix_data<ValueType, IndexType>& data)

// TODO: would be unnecessary if Dense<int> is possible
#define GKO_DECLARE_ADD_TO_ARRAY(ValueType)                        \
    void add_to_array(std::shared_ptr<const DefaultExecutor> exec, \
                      Array<ValueType>& array, const ValueType value)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                 \
    using global_index_type = distributed::global_index_type;        \
    using comm_index_type = distributed::comm_index_type;            \
    template <typename ValueType, typename LocalIndexType>           \
    GKO_DECLARE_BUILD_DIAG_OFFDIAG(ValueType, LocalIndexType);       \
    template <typename SourceType, typename TargetType>              \
    GKO_DECLARE_MAP_TO_GLOBAL_IDXS(SourceType, TargetType);          \
    template <typename ValueType, typename LocalIndexType>           \
    GKO_DECLARE_MERGE_DIAG_OFFDIAG(ValueType, LocalIndexType);       \
    template <typename ValueType, typename LocalIndexType>           \
    GKO_DECLARE_COMBINE_LOCAL_MTXS(ValueType, LocalIndexType);       \
    template <typename LocalIndexType>                               \
    GKO_DECLARE_CHECK_COLUMN_INDEX_EXISTS(LocalIndexType);           \
    template <typename LocalIndexType>                               \
    GKO_DECLARE_BUILD_RECV_SIZES(LocalIndexType);                    \
    template <typename ValueType, typename LocalIndexType>           \
    GKO_DECLARE_COMPRESS_OFFDIAG_DATA(ValueType, LocalIndexType);    \
    template <typename LocalIndexType>                               \
    GKO_DECLARE_CHECK_INDICES_WITHIN_SPAN(LocalIndexType);           \
    template <typename ValueType, typename LocalIndexType>           \
    GKO_DECLARE_ZERO_OUT_INVALID_COLUMNS(ValueType, LocalIndexType); \
    template <typename ValueType>                                    \
    GKO_DECLARE_ADD_TO_ARRAY(ValueType)


namespace omp {
namespace distributed_matrix {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_matrix
}  // namespace omp


namespace cuda {
namespace distributed_matrix {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_matrix
}  // namespace cuda


namespace reference {
namespace distributed_matrix {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_matrix
}  // namespace reference


namespace hip {
namespace distributed_matrix {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_matrix
}  // namespace hip


namespace dpcpp {
namespace distributed_matrix {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace distributed_matrix
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_MATRIX_KERNELS_HPP_
