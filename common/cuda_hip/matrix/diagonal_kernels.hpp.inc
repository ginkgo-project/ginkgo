// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void apply_to_csr(
    size_type num_rows, const ValueType* __restrict__ diag,
    const IndexType* __restrict__ row_ptrs,
    ValueType* __restrict__ result_values, bool inverse)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto row = thread::get_subwarp_id_flat<warp_size>();
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


}  // namespace kernel


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const DefaultExecutor> exec,
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
    if (grid_dim > 0) {
        kernel::apply_to_csr<<<grid_dim, default_block_size, 0,
                               exec->get_stream()>>>(
            num_rows, as_device_type(diag_values), csr_row_ptrs,
            as_device_type(csr_values), inverse);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);
