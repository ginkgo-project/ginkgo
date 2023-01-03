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

template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_1d(
    int64 size, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat<int64>();
    if (tidx >= size) {
        return;
    }
    fn(tidx, args...);
}


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_2d(
    int64 rows, int64 cols, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat<int64>();
    auto col = tidx % cols;
    auto row = tidx / cols;
    if (row >= rows) {
        return;
    }
    fn(row, col, args...);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DefaultExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs&&... args)
{
    if (size > 0) {
        constexpr auto block_size = default_block_size;
        auto num_blocks = ceildiv(size, block_size);
        generic_kernel_1d<<<num_blocks, block_size>>>(
            static_cast<int64>(size), fn, map_to_device(args)...);
    }
}

template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DefaultExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs&&... args)
{
    if (size[0] > 0 && size[1] > 0) {
        constexpr auto block_size = default_block_size;
        auto num_blocks = ceildiv(size[0] * size[1], block_size);
        generic_kernel_2d<<<num_blocks, block_size>>>(
            static_cast<int64>(size[0]), static_cast<int64>(size[1]), fn,
            map_to_device(args)...);
    }
}
