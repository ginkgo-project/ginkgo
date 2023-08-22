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

#include "core/matrix/diagonal_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace sycl {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


constexpr int default_block_size = 256;


namespace kernel {


template <typename ValueType, typename IndexType>
void apply_to_csr(size_type num_rows, const ValueType* __restrict__ diag,
                  const IndexType* __restrict__ row_ptrs,
                  ValueType* __restrict__ result_values, bool inverse,
                  ::sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));
    const auto row = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto tid_in_warp = warp_tile.thread_rank();

    if (row >= num_rows) {
        return;
    }

    const auto diag_val = inverse ? one<ValueType>() / diag[row] : diag[row];

    for (size_type idx = row_ptrs[row] + tid_in_warp; idx < row_ptrs[row + 1];
         idx += warp_size) {
        result_values[idx] *= diag_val;
    }
}

GKO_ENABLE_DEFAULT_HOST(apply_to_csr, apply_to_csr);


}  // namespace kernel


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const SyclExecutor> exec,
                  const matrix::Diagonal<ValueType>* a,
                  const matrix::Csr<ValueType, IndexType>* b,
                  matrix::Csr<ValueType, IndexType>* c, bool inverse)
{
    const auto num_rows = b->get_size()[0];
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();

    const auto grid_dim =
        ceildiv(num_rows * config::warp_size, default_block_size);
    kernel::apply_to_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                         num_rows, diag_values, csr_row_ptrs, csr_values,
                         inverse);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


}  // namespace diagonal
}  // namespace sycl
}  // namespace kernels
}  // namespace gko
