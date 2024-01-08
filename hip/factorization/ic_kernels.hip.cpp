// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ic_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>


#include "hip/base/hipsparse_bindings.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The ic factorization namespace.
 *
 * @ingroup factor
 */
namespace ic_factorization {


template <typename ValueType, typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             matrix::Csr<ValueType, IndexType>* m)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_hipsparse_handle();
    auto desc = hipsparse::create_mat_descr();
    auto info = hipsparse::create_ic0_info();

    // get buffer size for IC
    IndexType num_rows = m->get_size()[0];
    IndexType nnz = m->get_num_stored_elements();
    size_type buffer_size{};
    hipsparse::ic0_buffer_size(handle, num_rows, nnz, desc,
                               m->get_const_values(), m->get_const_row_ptrs(),
                               m->get_const_col_idxs(), info, buffer_size);

    array<char> buffer{exec, buffer_size};

    // set up IC(0)
    hipsparse::ic0_analysis(handle, num_rows, nnz, desc, m->get_const_values(),
                            m->get_const_row_ptrs(), m->get_const_col_idxs(),
                            info, HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                            buffer.get_data());

    hipsparse::ic0(handle, num_rows, nnz, desc, m->get_values(),
                   m->get_const_row_ptrs(), m->get_const_col_idxs(), info,
                   HIPSPARSE_SOLVE_POLICY_USE_LEVEL, buffer.get_data());

    hipsparse::destroy_ic0_info(info);
    hipsparse::destroy(desc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC_COMPUTE_KERNEL);


}  // namespace ic_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
