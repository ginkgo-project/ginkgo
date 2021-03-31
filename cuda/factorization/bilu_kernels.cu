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

#include "core/factorization/bilu_kernels.hpp"


#include <ginkgo/core/base/array.hpp>


#include "cuda/base/cusparse_block_bindings.hpp"
#include "cuda/base/device_guard.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Bilu factorization namespace.
 *
 * @ingroup factor
 */
namespace bilu_factorization {


// FIXME: For now, this only computes the (scalar) ILU0 factorization for
//  a matrix in BCSR format.
template <typename ValueType, typename IndexType>
void compute_bilu(const std::shared_ptr<const DefaultExecutor> exec,
                  matrix::Fbcsr<ValueType, IndexType> *const m)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_cusparse_handle();
    gko::cuda::device_guard g{id};
    auto desc = cusparse::create_bsr_mat_descr();
    auto info = cusparse::create_bilu0_info();

    // get buffer size for ILU
    const IndexType num_rows = m->get_size()[0];
    const IndexType nnz = m->get_num_stored_elements();
    const int bs = m->get_block_size();
    const size_type buffer_size = cusparse::bilu0_buffer_size(
        handle, num_rows, nnz, desc, m->get_const_values(),
        m->get_const_row_ptrs(), m->get_const_col_idxs(), bs, info);

    Array<char> buffer{exec, buffer_size};

    // set up ILU(0)
    cusparse::bilu0_analysis(handle, num_rows, nnz, desc, m->get_values(),
                             m->get_const_row_ptrs(), m->get_const_col_idxs(),
                             bs, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                             buffer.get_data());

    cusparse::bilu0(handle, num_rows, nnz, desc, m->get_values(),
                    m->get_const_row_ptrs(), m->get_const_col_idxs(), bs, info,
                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer.get_data());

    cusparse::destroy(info);
    cusparse::destroy(desc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BILU_COMPUTE_BLU_KERNEL);


}  // namespace bilu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
