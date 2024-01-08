// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_2d_solver(
    int64 rows, int64 cols, int64 default_stride, KernelFunction fn,
    KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat<int64>();
    auto col = tidx % cols;
    auto row = tidx / cols;
    if (row >= rows) {
        return;
    }
    fn(row, col,
       device_unpack_solver_impl<KernelArgs>::unpack(args, default_stride)...);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const DefaultExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs&&... args)
{
    if (size[0] > 0 && size[1] > 0) {
        constexpr auto block_size = default_block_size;
        auto num_blocks = ceildiv(size[0] * size[1], block_size);
        generic_kernel_2d_solver<<<num_blocks, block_size, 0,
                                   exec->get_stream()>>>(
            static_cast<int64>(size[0]), static_cast<int64>(size[1]),
            static_cast<int64>(default_stride), fn, map_to_device(args)...);
    }
}
