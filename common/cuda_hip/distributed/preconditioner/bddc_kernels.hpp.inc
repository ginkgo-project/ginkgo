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

namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void restrict_residual(
    size_type num_global_rows, const ValueType* global_residual,
    const IndexType* global_to_local, const ValueType* weights,
    ValueType* local_residual)
{
    const auto global_idx = thread::get_thread_id_flat();

    if (global_idx < num_global_rows) {
        const auto local_idx = global_to_local[global_idx];
        local_residual[local_idx] =
            weights[local_idx] * global_residual[global_idx];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_reordered(
    size_type buf_size, const IndexType* src_numbering,
    const IndexType* dst_numbering, const ValueType* src, ValueType* dst,
    bool add)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < buf_size) {
        const auto dst_idx =
            dst_numbering == nullptr ? tidx : dst_numbering[tidx];
        const auto src_idx =
            src_numbering == nullptr ? tidx : src_numbering[tidx];

        if (src_idx == -1 || dst_idx == -1) {
            return;
        }
        if (add) {
            atomic_add(&dst[dst_idx], src[src_idx]);
        } else {
            dst[dst_idx] = src[src_idx];
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void coarsen(
    size_type coarse_size, const IndexType* global_to_local,
    const ValueType* local_coarse_residual, ValueType* global_coarse_residual,
    ValueType* global_coarse_solution)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < coarse_size) {
        global_coarse_residual[tidx] =
            local_coarse_residual[global_to_local[tidx]];
        global_coarse_solution[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void finalize(
    size_type num_rows, const ValueType* weights,
    const ValueType* coarse_solution, ValueType* local_solution)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_rows) {
        local_solution[tidx] =
            weights[tidx] * (local_solution[tidx] + coarse_solution[tidx]);
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void restrict_residual1(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Dense<ValueType>* global_residual,
                        const array<IndexType>& local_to_local,
                        const array<IndexType>& local_to_send_buffer,
                        const matrix::Diagonal<ValueType>* weights,
                        array<ValueType>& send_buffer,
                        matrix::Dense<ValueType>* local_residual)
{
    auto local_to_local_data = local_to_local.get_const_data();
    auto local_to_send_data = local_to_send_buffer.get_const_data();
    auto send_data = as_device_type(send_buffer.get_data());
    auto global_vals = as_device_type(global_residual->get_const_values());
    auto w = as_device_type(weights->get_const_values());
    auto local_vals = as_device_type(local_residual->get_values());
    auto global_rows = global_residual->get_size()[0];
    auto buf_size = local_to_send_buffer.get_num_elems();

    const auto restrict_grid_dim = ceildiv(global_rows, config::warp_size);
    kernel::restrict_residual<<<restrict_grid_dim, config::warp_size>>>(
        global_rows, global_vals, local_to_local_data, w, local_vals);

    const auto fill_grid_dim = ceildiv(buf_size, config::warp_size);
    kernel::fill_reordered<<<fill_grid_dim, config::warp_size>>>(
        buf_size, local_to_send_data, static_cast<const IndexType*>(nullptr),
        local_vals, send_data, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL1);


template <typename ValueType, typename IndexType>
void restrict_residual2(std::shared_ptr<const DefaultExecutor> exec,
                        const array<IndexType>& non_local_to_local,
                        const array<IndexType>& global_to_recv_buffer,
                        const array<IndexType>& non_local_idxs,
                        const array<ValueType>& recv_buffer,
                        matrix::Dense<ValueType>* local_residual)
{
    auto non_local_to_local_data = non_local_to_local.get_const_data();
    auto global_to_recv_data = global_to_recv_buffer.get_const_data();
    auto non_local_idxs_data = non_local_idxs.get_const_data();
    auto recv_data = as_device_type(recv_buffer.get_const_data());
    auto local_vals = as_device_type(local_residual->get_values());
    auto buf_size = non_local_to_local.get_num_elems();

    auto grid_dim = ceildiv(buf_size, config::warp_size);
    kernel::fill_reordered<<<grid_dim, config::warp_size>>>(
        buf_size, global_to_recv_data, non_local_to_local_data, recv_data,
        local_vals, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RESTRICT_RESIDUAL2);


template <typename ValueType, typename IndexType>
void coarsen_residual1(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_local_to_local,
                       const array<IndexType>& coarse_local_to_send,
                       const matrix::Dense<ValueType>* local_coarse_residual,
                       array<ValueType>& coarse_send_buffer,
                       ValueType* coarse_residual, ValueType* coarse_solution)
{
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto local_to_send_data = coarse_local_to_send.get_const_data();
    auto send_data = as_device_type(coarse_send_buffer.get_data());
    auto global_res_vals = as_device_type(coarse_residual);
    auto global_sol_vals = as_device_type(coarse_solution);
    auto local_vals = as_device_type(local_coarse_residual->get_const_values());
    auto coarse_size = coarse_local_to_local.get_num_elems();
    auto buf_size = coarse_local_to_send.get_num_elems();

    auto coarsen_grid_dim = ceildiv(coarse_size, config::warp_size);

    kernel::coarsen<<<coarsen_grid_dim, config::warp_size>>>(
        coarse_size, local_to_local_data, local_vals, global_res_vals,
        global_sol_vals);

    auto fill_grid_dim = ceildiv(buf_size, config::warp_size);
    kernel::fill_reordered<<<fill_grid_dim, config::warp_size>>>(
        buf_size, local_to_send_data, static_cast<const IndexType*>(nullptr),
        local_vals, send_data, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL1);


template <typename ValueType, typename IndexType>
void coarsen_residual2(std::shared_ptr<const DefaultExecutor> exec,
                       const array<IndexType>& coarse_recv_to_local,
                       const array<ValueType>& coarse_recv_buffer,
                       ValueType* coarse_residual)
{
    auto recv_to_local_data = coarse_recv_to_local.get_const_data();
    auto recv_data = as_device_type(coarse_recv_buffer.get_const_data());
    auto coarse_vals = as_device_type(coarse_residual);
    auto buf_size = coarse_recv_to_local.get_num_elems();

    auto grid_dim = ceildiv(buf_size, config::warp_size);
    kernel::fill_reordered<<<grid_dim, config::warp_size>>>(
        buf_size, static_cast<const IndexType*>(nullptr), recv_to_local_data,
        recv_data, coarse_vals, true);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COARSEN_RESIDUAL2);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& coarse_local_to_local,
    const matrix::Dense<ValueType>* coarse_solution_local,
    const array<IndexType>& coarse_non_local_to_local,
    const array<IndexType>& coarse_local_to_non_local,
    const matrix::Dense<ValueType>* non_local,
    matrix::Dense<ValueType>* local_intermediate)
{
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto non_local_to_local_data = coarse_non_local_to_local.get_const_data();
    auto local_to_non_local_data = coarse_local_to_non_local.get_const_data();
    auto coarse_vals =
        as_device_type(coarse_solution_local->get_const_values());
    auto non_local_vals = as_device_type(non_local->get_const_values());
    auto intermediate_vals = as_device_type(local_intermediate->get_values());
    auto local_to_local_size = coarse_local_to_local.get_num_elems();
    auto local_to_non_local_size = coarse_local_to_non_local.get_num_elems();

    auto local_grid_dim = ceildiv(local_to_local_size, config::warp_size);
    kernel::fill_reordered<<<local_grid_dim, config::warp_size>>>(
        local_to_local_size, static_cast<const IndexType*>(nullptr),
        local_to_local_data, coarse_vals, intermediate_vals, false);

    auto non_local_grid_dim =
        ceildiv(local_to_non_local_size, config::warp_size);
    if (non_local->get_size()[0] > 0) {
        kernel::fill_reordered<<<non_local_grid_dim, config::warp_size>>>(
            local_to_non_local_size, local_to_non_local_data,
            non_local_to_local_data, non_local_vals, intermediate_vals, false);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution1(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_recv_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              array<ValueType>& coarse_recv_buffer)
{
    auto recv_size = coarse_recv_buffer.get_num_elems();
    auto recv_to_local_data = coarse_recv_to_local.get_const_data();
    auto coarse_vals = as_device_type(coarse_solution->get_const_values());
    auto recv_data = as_device_type(coarse_recv_buffer.get_data());

    auto grid_dim = ceildiv(recv_size, config::warp_size);
    kernel::fill_reordered<<<grid_dim, config::warp_size>>>(
        recv_size, recv_to_local_data, static_cast<const IndexType*>(nullptr),
        coarse_vals, recv_data, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION1);


template <typename ValueType, typename IndexType>
void prolong_coarse_solution2(std::shared_ptr<const DefaultExecutor> exec,
                              const array<IndexType>& coarse_local_to_local,
                              const matrix::Dense<ValueType>* coarse_solution,
                              const array<IndexType>& coarse_local_to_send,
                              const array<ValueType>& coarse_send_buffer,
                              matrix::Dense<ValueType>* local_intermediate)
{
    auto local_to_local_size = coarse_local_to_local.get_num_elems();
    auto send_size = coarse_send_buffer.get_num_elems();
    auto local_to_local_data = coarse_local_to_local.get_const_data();
    auto local_to_send_data = coarse_local_to_send.get_const_data();
    auto coarse_vals = as_device_type(coarse_solution->get_const_values());
    auto send_data = as_device_type(coarse_send_buffer.get_const_data());
    auto intermediate_vals = as_device_type(local_intermediate->get_values());

    auto local_to_local_grid_dim =
        ceildiv(local_to_local_size, config::warp_size);
    kernel::fill_reordered<<<local_to_local_grid_dim, config::warp_size>>>(
        local_to_local_size, static_cast<const IndexType*>(nullptr),
        local_to_local_data, coarse_vals, intermediate_vals, false);

    auto send_grid_dim = ceildiv(send_size, config::warp_size);
    kernel::fill_reordered<<<send_grid_dim, config::warp_size>>>(
        send_size, static_cast<const IndexType*>(nullptr), local_to_send_data,
        send_data, intermediate_vals, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PROLONG_COARSE_SOLUTION2);


template <typename ValueType, typename IndexType>
void finalize1(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* coarse_solution,
               const matrix::Diagonal<ValueType>* weights,
               const array<IndexType>& recv_to_local,
               const array<IndexType>& non_local_to_local,
               array<ValueType>& recv_buffer,
               matrix::Dense<ValueType>* local_solution)
{
    auto num_rows = coarse_solution->get_size()[0];
    auto w = as_device_type(weights->get_const_values());
    auto coarse_vals = as_device_type(coarse_solution->get_const_values());
    auto local_vals = as_device_type(local_solution->get_values());
    auto non_local_to_local_data = non_local_to_local.get_const_data();
    auto recv_to_local_data = recv_to_local.get_const_data();
    auto recv_data = as_device_type(recv_buffer.get_data());
    auto non_local_size = non_local_to_local.get_num_elems();

    auto finalize_grid_dim = ceildiv(num_rows, config::warp_size);
    kernel::finalize<<<finalize_grid_dim, config::warp_size>>>(
        num_rows, w, coarse_vals, local_vals);

    auto fill_grid_dim = ceildiv(non_local_size, config::warp_size);
    kernel::fill_reordered<<<fill_grid_dim, config::warp_size>>>(
        non_local_size, non_local_to_local_data, recv_to_local_data, local_vals,
        recv_data, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE1);


template <typename ValueType, typename IndexType>
void finalize2(std::shared_ptr<const DefaultExecutor> exec,
               const array<ValueType>& send_buffer,
               const array<IndexType>& local_to_send_buffer,
               const array<IndexType>& local_to_local,
               matrix::Dense<ValueType>* local_solution,
               ValueType* global_solution)
{
    auto num_rows = local_to_local.get_num_elems();
    auto send_data = as_device_type(send_buffer.get_const_data());
    auto local_vals = as_device_type(local_solution->get_values());
    auto global_vals = as_device_type(global_solution);
    auto local_to_send_data = local_to_send_buffer.get_const_data();
    auto local_to_local_data = local_to_local.get_const_data();
    auto buf_size = send_buffer.get_num_elems();

    auto add_grid_dim = ceildiv(buf_size, config::warp_size);
    kernel::fill_reordered<<<add_grid_dim, config::warp_size>>>(
        buf_size, static_cast<const IndexType*>(nullptr), local_to_send_data,
        send_data, local_vals, true);

    auto fill_grid_dim = ceildiv(num_rows, config::warp_size);
    kernel::fill_reordered<<<fill_grid_dim, config::warp_size>>>(
        num_rows, local_to_local_data, static_cast<const IndexType*>(nullptr),
        local_vals, global_vals, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FINALIZE2);


template <typename ValueType, typename IndexType>
void static_condensation1(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* residual,
                          const array<IndexType>& inner_to_local,
                          matrix::Dense<ValueType>* inner_residual)
{
    auto inner_size = inner_to_local.get_num_elems();
    auto inner_to_local_data = inner_to_local.get_const_data();
    auto res_vals = as_device_type(residual->get_const_values());
    auto inner_vals = as_device_type(inner_residual->get_values());

    auto grid_dim = ceildiv(inner_size, config::warp_size);
    kernel::fill_reordered<<<grid_dim, config::warp_size>>>(
        inner_size, static_cast<const IndexType*>(nullptr), inner_to_local_data,
        res_vals, inner_vals, false);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION1);


template <typename ValueType, typename IndexType>
void static_condensation2(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* inner_solution,
                          const array<IndexType>& inner_to_local,
                          ValueType* solution)
{
    auto inner_size = inner_to_local.get_num_elems();
    auto inner_vals = as_device_type(inner_solution->get_const_values());
    auto inner_to_local_data = inner_to_local.get_const_data();
    auto sol_vals = as_device_type(solution);

    auto grid_dim = ceildiv(inner_size, config::warp_size);
    kernel::fill_reordered<<<grid_dim, config::warp_size>>>(
        inner_size, inner_to_local_data, static_cast<const IndexType*>(nullptr),
        inner_vals, sol_vals, true);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_STATIC_CONDENSATION2);
