/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void subwarp_apply(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const auto alpha_val = alpha[0];
    auto num_subwarp = blockDim.x / subwarp_size * gridDim.x;
    int i = 0;
    while (i < iteration) {
        const auto rhs = b[row * b_stride + column_id];
        const auto ind_end = row_ptrs[row + 1];
        ValueType temp_val = zero<ValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += val[ind] * c[col_idxs[ind] * c_stride + column_id];
        }
        auto subwarp_result = reduce(
            subwarp_tile, temp_val,
            [](const ValueType& a, const ValueType& b) { return a + b; });
        if (subid == 0) {
            c[row * c_stride + column_id] += alpha_val * (rhs - subwarp_result);
        }
#if USE_THREADFENCE
        __threadfence();
#endif
        row += num_subwarp;
        if (row >= num_rows) {
            row -= num_rows;
            i++;
        }
    }
}

namespace {
template <typename ResultType, typename ValueType>
__forceinline__ __device__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType&>(val);
}
}  // namespace

template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void subwarp_apply_flow(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const int alpha_val = real(alpha[0]);
    // num_subwarp < num_rows
    auto num_subwarp = blockDim.x / subwarp_size * gridDim.x;
    int i = 0;
    while (i < iteration) {
        const auto rhs = b[row * b_stride + column_id];
        const auto ind_end = row_ptrs[row + 1];
        unsigned long long temp_val = 0;
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val +=
                (((reinterpret<unsigned long long>(static_cast<double>(
                       real(c[col_idxs[ind] * c_stride + column_id]))) >>
                   (2 * 12)) &
                  0xFFF)
                 << (static_cast<unsigned long long>(real(val[ind])) * 12));
        }

        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const unsigned long long& a,
                      const unsigned long long& b) { return a + b; }) +
            (1 << (2 * 12));
        if (subid == 0) {
            // c[row * c_stride + column_id] +=
            //     alpha_val * (rhs - subwarp_result);
            c[row * c_stride + column_id] =
                static_cast<ValueType>(reinterpret<double>(
                    alpha_val *
                    (subwarp_result - (subwarp_result & (0xFFFull << 24)) +
                     (static_cast<unsigned long long>(i + 1) << 24)))) +
                (1 - alpha_val) * (rhs + c[row * c_stride + column_id]);
        }
#if USE_THREADFENCE
        __threadfence();
#endif
        row += num_subwarp;
        if (row >= num_rows) {
            row -= num_rows;
            i++;
        }
    }
}

template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void subwarp_apply_halfflow(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const int alpha_val = real(alpha[0]);
    auto num_subwarp = blockDim.x / subwarp_size * gridDim.x;
    int i = 0;
    while (i < iteration) {
        const auto rhs = b[row * b_stride + column_id];
        const auto ind_end = row_ptrs[row + 1];
        unsigned long long temp_val = 0;
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val +=
                (((reinterpret<unsigned long long>(static_cast<double>(
                       real(c[col_idxs[ind] * c_stride + column_id]))) >>
                   (2 * 12)) &
                  0xFFF)
                 << (static_cast<unsigned long long>(real(val[ind])) * 12));
        }

        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const unsigned long long& a,
                      const unsigned long long& b) { return a + b; }) +
            (1 << (2 * 12));
        if (subid == 0) {
            if (i < iteration / 2) {
                c[row * c_stride + column_id] =
                    static_cast<ValueType>(reinterpret<double>(
                        alpha_val *
                        (subwarp_result - (subwarp_result & (0xFFFull << 24)) +
                         (static_cast<unsigned long long>(i + 1) << 24)))) +
                    (1 - alpha_val) * (rhs + c[row * c_stride + column_id]);
            } else {
                const auto t = reinterpret<unsigned long long>(
                    static_cast<double>(real(c[row * c_stride + column_id])));
                c[row * c_stride + column_id] =
                    static_cast<ValueType>(reinterpret<double>(
                        (1 - alpha_val) * subwarp_result + t -
                        (t & (0xFFFull << 24)) +
                        (static_cast<unsigned long long>(i + 1) << 24))) +
                    (1 - alpha_val) * rhs;
            }
        }
#if USE_THREADFENCE
        __threadfence();
#endif
        row += num_subwarp;
        if (row >= num_rows) {
            row -= num_rows;
            i++;
        }
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void subwarp_apply_time(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const int alpha_val = real(alpha[0]);
    unsigned int time = 0;
    unsigned int start = 0;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(start));
    auto num_subwarp = blockDim.x / subwarp_size * gridDim.x;
    int i = 0;
    while (i < iteration) {
        const auto rhs = b[row * b_stride + column_id];
        const auto ind_end = row_ptrs[row + 1];
        unsigned long long temp_val = 0;
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val +=
                (((reinterpret<unsigned long long>(static_cast<double>(
                       real(c[col_idxs[ind] * c_stride + column_id]))) >>
                   (2 * 12)) &
                  0xFFF)
                 << (static_cast<unsigned long long>(real(val[ind])) * 12));
        }

        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const unsigned long long& a,
                      const unsigned long long& b) { return a + b; });

        if (subid == 0) {
            asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(time));
            c[row * c_stride + column_id] =
                static_cast<ValueType>(reinterpret<double>(
                    (1 - alpha_val) * subwarp_result +
                    ((static_cast<unsigned long long>(start) << 32) + time))) +
                (1 - alpha_val) * (rhs + c[row * c_stride + column_id]);
        }
#if USE_THREADFENCE
        __threadfence();
#endif
        row += num_subwarp;
        if (row >= num_rows) {
            row -= num_rows;
            i++;
        }
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void second_subwarp_apply(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ beta, const ValueType* __restrict__ b,
    const size_type b_stride, ValueType* __restrict__ c,
    const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    if (row < num_rows) {
        const auto rhs = b[row * b_stride + column_id];
        ValueType old_x = c[row * c_stride + column_id];  // x_0
        {
            // first iteration: 1st order
            const auto ind_end = row_ptrs[row + 1];
            ValueType temp_val = zero<ValueType>();
            for (auto ind = row_ptrs[row] + subid; ind < ind_end;
                 ind += subwarp_size) {
                temp_val += val[ind] * c[col_idxs[ind] * c_stride + column_id];
            }
            auto subwarp_result = reduce(
                subwarp_tile, temp_val,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            if (subid == 0) {
                c[row * c_stride + column_id] =
                    old_x + alpha_val * (rhs - subwarp_result);
            }
        }
        for (int i = 1; i < iteration; i++) {
            const auto curr_x = c[row * c_stride + column_id];  // curr_x
            const auto ind_end = row_ptrs[row + 1];
            ValueType temp_val = zero<ValueType>();
            for (auto ind = row_ptrs[row] + subid; ind < ind_end;
                 ind += subwarp_size) {
                temp_val += val[ind] * c[col_idxs[ind] * c_stride + column_id];
            }
            auto subwarp_result = reduce(
                subwarp_tile, temp_val,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            if (subid == 0) {
                // new_x
                c[row * c_stride + column_id] =
                    (1 + beta_val) *
                        (curr_x + alpha_val * (rhs - subwarp_result)) -
                    beta_val * old_x;
            }
            old_x = curr_x;
        }
    }
}