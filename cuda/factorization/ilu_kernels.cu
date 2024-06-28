// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ilu_kernels.hpp"

#include <ginkgo/core/base/array.hpp>

#include "common/cuda_hip/base/sparselib_bindings.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace ilu_factorization {


template <typename ValueType, typename IndexType>
void compute_lu(std::shared_ptr<const DefaultExecutor> exec,
                matrix::Csr<ValueType, IndexType>* m)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_sparselib_handle();
    auto desc = sparselib::create_mat_descr();
    auto info = sparselib::create_ilu0_info();

    // get buffer size for ILU
    IndexType num_rows = m->get_size()[0];
    IndexType nnz = m->get_num_stored_elements();
    size_type buffer_size{};
    sparselib::ilu0_buffer_size(handle, num_rows, nnz, desc,
                                m->get_const_values(), m->get_const_row_ptrs(),
                                m->get_const_col_idxs(), info, buffer_size);

    array<char> buffer{exec, buffer_size};

    // set up ILU(0)
    sparselib::ilu0_analysis(handle, num_rows, nnz, desc, m->get_const_values(),
                             m->get_const_row_ptrs(), m->get_const_col_idxs(),
                             info, SPARSELIB_SOLVE_POLICY_USE_LEVEL,
                             buffer.get_data());

    sparselib::ilu0(handle, num_rows, nnz, desc, m->get_values(),
                    m->get_const_row_ptrs(), m->get_const_col_idxs(), info,
                    SPARSELIB_SOLVE_POLICY_USE_LEVEL, buffer.get_data());

    // CUDA 11.4 has a use-after-free bug on Turing
#if (CUDA_VERSION >= 11040)
    exec->synchronize();
#endif

    sparselib::destroy_ilu0_info(info);
    sparselib::destroy(desc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ILU_COMPUTE_LU_KERNEL);


}  // namespace ilu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
