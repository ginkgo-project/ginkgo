// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
namespace dpcpp {
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
                  sycl::nd_item<3> item_ct1)
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
void apply_to_csr(std::shared_ptr<const DpcppExecutor> exec,
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
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
