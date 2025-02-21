// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_lookup.hpp"

#include "core/base/array_utils.hpp"
#include "core/matrix/csr_kernels.hpp"

namespace gko {
namespace matrix {
namespace csr {
namespace {


GKO_REGISTER_OPERATION(build_lookup_offsets, csr::build_lookup_offsets);
GKO_REGISTER_OPERATION(build_lookup, csr::build_lookup);


}  // namespace


template <typename ValueType, typename IndexType>
lookup_data<IndexType> build_lookup(const Csr<ValueType, IndexType>* mtx,
                                    sparsity_type allowed_sparsity)
{
    const auto exec = mtx->get_executor();
    const auto size = mtx->get_size()[0];
    lookup_data<IndexType> result{exec, size};
    exec->run(make_build_lookup_offsets(
        mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(), size,
        allowed_sparsity, result.storage_offsets.get_data()));
    const auto storage_size = get_element(result.storage_offsets, size);
    result.storage.resize_and_reset(static_cast<size_type>(storage_size));
    exec->run(make_build_lookup(
        mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(), size,
        allowed_sparsity, result.storage_offsets.get_const_data(),
        result.row_descs.get_data(), result.storage.get_data()));
    return result;
}


#define GKO_INSTANTIATE_BUILD_LOOKUP(ValueType, IndexType)                    \
    lookup_data<IndexType> build_lookup(const Csr<ValueType, IndexType>* mtx, \
                                        sparsity_type allowed_sparsity)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_INSTANTIATE_BUILD_LOOKUP);


}  // namespace csr
}  // namespace matrix
}  // namespace gko
