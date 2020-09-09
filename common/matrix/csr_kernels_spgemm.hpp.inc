/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

constexpr auto spgemm_block_size = 64;
constexpr auto spgemm_child_count = 4;


__device__ int bitmask_select(config::lane_mask_type mask, int rank)
{
    return synchronous_fixed_binary_search<config::warp_size>([&](int pos) {
        return popcnt(mask & (~config::lane_mask_type{} >>
                              (config::warp_size - pos - 1)));
    });
}


template <typename ValueType, typename IndexType>
struct memory_row_accessor {
    __device__ memory_row_accessor(const ValueType *val, const IndexType *idx,
                                   IndexType base_idx, IndexType size)
        : val_{val + base_idx}, idx_{idx + base_idx}, size_{size}
    {}

    __device__ __forceinline__ IndexType col(IndexType i) const
    {
        return idx_[i];
    }

    __device__ __forceinline__ ValueType val(IndexType i) const
    {
        return val_[i];
    }

    const ValueType *val_;
    const IndexType *idx_;
    IndexType size_;
};


template <typename ValueType, typename IndexType>
struct merge_row_accessor {
    __device__ merge_row_accessor(IndexType base_idx, IndexType size)
        : base_idx_{base_idx}, size_{size}
    {}

    __device__ __forceinline__ IndexType col(IndexType i) const
    {
        return base_idx_ + i;
    }

    __device__ __forceinline__ ValueType val(IndexType i) const
    {
        return one<ValueType>();
    }

    IndexType base_idx_;
    IndexType size_;
};


template <typename IndexType>
struct count_accumulator {
    constexpr static auto use_val = false;

    __device__ count_accumulator() : count_{} {}

    template <typename ValueType>
    __device__ __forceinline__ void accumulate(ValueType val, IndexType col)
    {}

    __device__ __forceinline__ void output_column(IndexType col) { count_++; }

    IndexType count_;
};


template <typename ValueType, typename IndexType>
struct output_accumulator {
    constexpr static auto use_val = true;

    __device__ output_accumulator(IndexType *out_cols, ValueType *out_vals,
                                  IndexType out_begin, bool write)
        : out_cols_{out_cols},
          out_vals_{out_vals},
          out_idx_{out_begin},
          out_acc_{zero<ValueType>()},
          write_{write}
    {}

    __device__ __forceinline__ void accumulate(ValueType val, IndexType col)
    {
        out_acc_ += val;
    }

    __device__ __forceinline__ void output_column(IndexType col)
    {
        if (write_) {
            out_cols_[out_idx_] = col;
            out_vals_[out_idx_] = out_acc_;
        }
        ++out_idx_;
        out_acc_ = zero<ValueType>();
    }

    IndexType *out_cols_;
    ValueType *out_vals_;
    IndexType out_idx_;
    ValueType out_acc_;
    bool write_;
};


template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}


template <typename T>
__forceinline__ __device__ void replace_if(bool use_new, T &val, T &new_val,
                                           T &storage)
{
    storage = use_new ? val : new_val;
    val = use_new ? new_val : val;
}


template <typename Callback>
__device__ void forall_bits(config::lane_mask_type mask, Callback cb)
{
    while (mask) {
        cb(ffs(mask) - 1);
        mask &= mask - 1;
    }
}


template <int max_num_rows, typename ValueType, typename IndexType,
          typename AAccessor, typename Accumulator>
__device__ void spgemm_multiway_merge_short(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    Accumulator &acc)
{
    constexpr auto use_val = Accumulator::use_val;
    // short rows of a: we can keep the state for each row of b in registers
    // partitioning of data: thread groups of size subwarp_size store the
    // content of a single row of b as a shift register. We only need to look
    // at the first element in each group, but can avoid many smaller loads.
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = warp_size / max_num_rows;
    auto warp = group::tiled_partition<warp_size>(group::this_thread_block());

    const auto warp_rank = warp.thread_rank();
    const auto subwarp_id =
        min(static_cast<IndexType>(warp_rank / subwarp_size), a.size_ - 1);
    const auto last_subwarp_base = (a.size_ - 1) * subwarp_size;
    const auto last_subwarp = subwarp_id == a.size_ - 1;
    const auto last_subwarp_size = warp_size - last_subwarp_base;
    const auto subwarp_base = subwarp_id * subwarp_size;
    const auto subwarp_rank = static_cast<IndexType>(warp_rank - subwarp_base);
    // load the first subwarp_size elements for each row of b
    const auto a_idx = subwarp_id;
    const auto a_col = a.col(a_idx);
    const auto a_val = use_val ? a.val(a_idx) : zero<ValueType>();
    const auto b_begin = b_row_ptrs[a_col];
    const auto b_end = b_row_ptrs[a_col + 1];
    auto b_idx = b_begin + subwarp_rank;
    auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
    auto b_val = (use_val && b_idx < b_end) ? b_vals[b_idx] : zero<ValueType>();
    auto b_remaining = last_subwarp ? last_subwarp_size : subwarp_size;
    while (warp.any(b_col < sentinel)) {
        // compute the minimum column
        auto min_col = b_col;
#pragma unroll
        for (auto i = subwarp_size; i < warp_size; i *= 2) {
            min_col = min(min_col, warp.shfl_xor(min_col, i));
        }
        // min_col is only correct for the first thread in the subwarp!
        // all other values are larger
        auto min_mask = warp.ballot(b_col == min_col);
        auto sub_is_min = bool((min_mask >> subwarp_base) & 1);
        // accumulate all products
        auto cur_val =
            sub_is_min && subwarp_rank == 0 ? a_val * b_val : zero<ValueType>();
        if (use_val) {
#pragma unroll
            for (auto i = subwarp_size; i < warp_size; i *= 2) {
                cur_val += warp.shfl_xor(cur_val, i);
            }
        }
        // divergent values of cur_val! fix for this:
        // cur_val = warp.shfl(cur_val, 0);
        // unnecessary, since only thread 0 writes to output

        // output
        acc.accumulate(cur_val, min_col);
        acc.output_column(min_col);
        // advance elements
        auto b_next_col = warp.shfl_down(b_col, 1);
        auto b_next_val = use_val ? warp.shfl_down(b_val, 1) : b_val;
        if (sub_is_min) {
            b_idx++;
            if (subwarp_size == 1) {
                b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                b_val = (use_val && b_idx < b_end) ? b_vals[b_idx]
                                                   : zero<ValueType>();
            } else {
                b_remaining--;
                // simulate behavior of subwarp-local shfl_down
                if (last_subwarp || subwarp_rank < subwarp_size - 1) {
                    b_col = b_next_col;
                    b_val = b_next_val;
                }
                // if there are none left: load new elements
                if (b_remaining == 0) {
                    b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                    b_val = (use_val && b_idx < b_end) ? b_vals[b_idx]
                                                       : zero<ValueType>();
                    b_remaining =
                        last_subwarp ? last_subwarp_size : subwarp_size;
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType, typename AAccessor,
          typename Accumulator>
__device__ void spgemm_multiway_merge_medium(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *sh_cols, IndexType *sh_idxs, IndexType *sh_ends,
    ValueType *sh_vals, Accumulator &acc)
{
    constexpr auto use_val = Accumulator::use_val;
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = spgemm_child_count;
    auto thread_block = group::this_thread_block();
    auto warp = group::tiled_partition<warp_size>(thread_block);
    auto subwarp = group::tiled_partition<subwarp_size>(thread_block);
    const auto warp_rank = warp.thread_rank();
    // the first threadIdx in the warp
    const auto warp_base = threadIdx.x / warp_size * warp_size;
    const auto subwarp_rank = subwarp.thread_rank();
    const auto subwarp_id = warp_rank / subwarp_size;
    // first thread from the subwarp (warp-local)
    const auto subwarp_base = subwarp_id * subwarp_size;
    // lane mask containing only the current subwarp (warp-local)
    const auto subwarp_mask = ((config::lane_mask_type{1} << subwarp_size) - 1)
                              << subwarp_base;
    // index for shared-memory heap
    auto heap_idx = [&](int thread, int child) {
        return (thread + warp_base) * subwarp_size + child;
    };
    auto min_op = [](IndexType a, IndexType b) { return min(a, b); };
    auto sum_op = [](ValueType a, ValueType b) { return a + b; };

    // build heap
    auto a_col = a.col(warp_rank);
    auto a_val = use_val ? a.val(warp_rank) : zero<ValueType>();
    auto b_idx = b_row_ptrs[a_col];
    auto b_end = b_row_ptrs[a_col + 1];
    auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
    for (int i = 0; i < subwarp_size; ++i) {
        // memory barrier
        warp.sync();
        auto a_idx = static_cast<IndexType>((i + 1) * warp_size + warp_rank);
        auto a_new_col = a_idx < a.size_ ? a.col(a_idx) : sentinel;
        auto a_new_val =
            (a_idx < a.size_ && use_val) ? a.val(a_idx) : zero<ValueType>();
        // ensure that skipped a_cols result in sentinel b_cols
        auto skip = a_new_col == sentinel;
        auto b_new_idx = skip ? IndexType{} : b_row_ptrs[a_new_col];
        auto b_new_end = skip ? IndexType{} : b_row_ptrs[a_new_col + 1];
        auto b_new_col = checked_load(b_cols, b_new_idx, b_new_end, sentinel);
        auto better = b_new_col < b_col;
        // store the entry with the larger column in shared memory
        auto sh_idx = heap_idx(warp_rank, i);
        replace_if(better, b_col, b_new_col, sh_cols[sh_idx]);
        replace_if(better, b_idx, b_new_idx, sh_idxs[sh_idx]);
        replace_if(better, b_end, b_new_end, sh_ends[sh_idx]);
        if (use_val) {
            replace_if(better, a_val, a_new_val, sh_vals[sh_idx]);
        }
    }

    auto min_col = reduce(warp, b_col, min_op);
    while (min_col < sentinel) {
        // find all minimal heap elements
        auto min_mask = warp.ballot(b_col == min_col);
        auto cur_val = (b_col == min_col && use_val) ? a_val * b_vals[b_idx]
                                                     : zero<ValueType>();
        cur_val = use_val ? reduce(warp, cur_val, sum_op) : cur_val;
        acc.accumulate(cur_val, min_col);
        // update the heap
        if (b_col == min_col) {
            b_idx++;
            b_col = checked_load(b_cols, b_idx, b_end, sentinel);
        }
        // for all threads in subwarp: swap with child entries if necessary
        forall_bits(min_mask & subwarp_mask, [&](int thread) {
            // memory barrier
            subwarp.sync();
            auto sh_idx = heap_idx(thread, subwarp_rank);
            // find its minimum child
            auto b_new_col = sh_cols[sh_idx];
            auto local_min_col = reduce(subwarp, b_new_col, min_op);
            auto local_is_min = subwarp.ballot(local_min_col == b_new_col);
            auto local_min_thread = ffs(local_is_min) - 1;
            // and update its local data if necessary
            if (warp_rank == thread && local_min_col < b_col) {
                sh_idx = heap_idx(warp_rank, local_min_thread);
                swap(b_col, sh_cols[sh_idx]);
                swap(b_idx, sh_idxs[sh_idx]);
                swap(b_end, sh_ends[sh_idx]);
                if (use_val) {
                    swap(a_val, sh_vals[sh_idx]);
                }
            }
        });
        auto new_min_col = reduce(warp, b_col, min_op);
        if (new_min_col > min_col) {
            acc.output_column(min_col);
        }
        min_col = new_min_col;
    }
}


template <typename ValueType, typename IndexType, typename AAccessor,
          typename Accumulator>
__device__ void spgemm_multiway_merge_large(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *__restrict__ sh_cols, IndexType *__restrict__ sh_idxs,
    IndexType *__restrict__ sh_ends, ValueType *__restrict__ sh_vals,
    IndexType *__restrict__ gl_cols, IndexType *__restrict__ gl_idxs,
    IndexType *__restrict__ gl_ends, ValueType *__restrict__ gl_vals,
    Accumulator &acc)
{
    constexpr auto use_val = Accumulator::use_val;
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = spgemm_child_count;
    constexpr auto subwarp_count = warp_size / subwarp_size;
    // reduction operations
    auto min_op = [](IndexType a, IndexType b) { return min(a, b); };
    auto sum_op = [](ValueType a, ValueType b) { return a + b; };

    auto thread_block = group::this_thread_block();

    // warp cooperative group for global minimum and sum reduction
    auto warp = group::tiled_partition<warp_size>(thread_block);
    // rank of the thread within the warp (= sub-heap idx)
    const auto warp_rank = warp.thread_rank();
    // ID of the warp within the block
    const auto warp_id = threadIdx.x / warp_size;
    // first thread in the warp (block-local, for shared memory access)
    const auto warp_base = warp_id * warp_size;

    // subwarp cooperative group for heap updates
    auto subwarp = group::tiled_partition<subwarp_size>(thread_block);
    // rank of the thread within the subwarp
    const auto subwarp_rank = subwarp.thread_rank();
    // ID of the subwarp within the warp
    const auto subwarp_id = warp_rank / subwarp_size;
    // first thread from the subwarp (warp-local)
    const auto subwarp_base = subwarp_id * subwarp_size;
    // lane mask containing only the current subwarp (warp-local)
    const auto subwarp_mask = ((config::lane_mask_type{1} << subwarp_size) - 1)
                              << subwarp_base;

    // child index for shared-memory heap
    auto sh_child_idx = [&](int thread, int child) {
        return (thread + warp_base) * spgemm_child_count + child;
    };
    // translates a shared-memory heap index into a global-memory heap index
    auto sh_to_gl = [&](int idx) {
        return static_cast<IndexType>(idx - warp_base * spgemm_child_count +
                                      warp_size);
    };
    // child index for global-memory heap
    auto child_idx = [](IndexType pos, IndexType child) {
        return static_cast<IndexType>(pos * spgemm_child_count + warp_size +
                                      child);
    };
    // restore the global-memory heap property from node i downwards
    auto sift_down = [&](IndexType i) {
        auto parent_col = gl_cols[i];
        // as long as i has any children:
        while (child_idx(i, 0) < a.size_) {
            // memory barrier
            subwarp.sync();
            // find minimum child
            auto child = child_idx(i, subwarp_rank);
            auto child_col = checked_load(gl_cols, child, a.size_, sentinel);
            auto min_col = reduce(subwarp, child_col, min_op);
            if (min_col >= parent_col) {
                break;
            }
            // swap minimum child up
            auto min_mask = subwarp.ballot(min_col == child_col);
            auto min_child = ffs(min_mask) - 1;
            if (subwarp_rank == min_child) {
                swap(gl_cols[i], gl_cols[child]);
                swap(gl_idxs[i], gl_idxs[child]);
                swap(gl_ends[i], gl_ends[child]);
                if (use_val) {
                    swap(gl_vals[i], gl_vals[child]);
                }
            }
            // proceed from location of minimum child
            i = child_idx(i, min_child);
        }
    };

    // fill heap storage
    for (int i = warp_rank; i < a.size_; i += warp_size) {
        auto a_col = a.col(i);
        auto b_begin = b_row_ptrs[a_col];
        auto b_end = b_row_ptrs[a_col + 1];
        gl_idxs[i] = b_begin;
        gl_ends[i] = b_end;
        gl_cols[i] = checked_load(b_cols, b_begin, b_end, sentinel);
        if (use_val) {
            gl_vals[i] = a.val(i);
        }
    }
    // build heap: each subwarp handles one node
    // this should not create overlaps/races between children/parents
    auto last_parent = (a.size_ - 1 - warp_size) / spgemm_child_count;
    for (IndexType i = last_parent - subwarp_id; i >= 0; i -= subwarp_count) {
        // memory barrier
        subwarp.sync();
        sift_down(i);
    }

    // memory barrier
    warp.sync();
    // load top for each sub-heap into registers
    auto b_col = gl_cols[warp_rank];
    auto b_idx = gl_idxs[warp_rank];
    auto b_end = gl_ends[warp_rank];
    auto a_val = use_val ? gl_vals[warp_rank] : zero<ValueType>();
    // load next level of heap into shared memory
    for (int i = warp_rank; i < spgemm_child_count * warp_size;
         i += warp_size) {
        auto sh_idx = i + warp_base * spgemm_child_count;
        sh_cols[sh_idx] = gl_cols[i + warp_size];
        sh_idxs[sh_idx] = gl_idxs[i + warp_size];
        sh_ends[sh_idx] = gl_ends[i + warp_size];
        if (use_val) {
            sh_vals[sh_idx] = gl_vals[i + warp_size];
        }
    }

    auto min_col = reduce(warp, b_col, min_op);
    while (min_col < sentinel) {
        // find all heap tops that are minimal
        auto min_mask = warp.ballot(b_col == min_col);
        // compute current product
        auto cur_val = (b_col == min_col && use_val) ? a_val * b_vals[b_idx]
                                                     : zero<ValueType>();
        // compute current partial sum and call callback
        cur_val = use_val ? reduce(warp, cur_val, sum_op) : cur_val;
        acc.accumulate(cur_val, min_col);

        // update the heap
        if (b_col == min_col) {
            b_idx++;
            b_col = checked_load(b_cols, b_idx, b_end, sentinel);
        }
        // for all threads in subwarp: restore heap property
        forall_bits(min_mask & subwarp_mask, [&](int heap) {
            // memory barrier
            subwarp.sync();
            // i'th thread loads i'th child
            auto sh_idx = sh_child_idx(heap, subwarp_rank);
            // find its minimum child
            auto child_col = sh_cols[sh_idx];
            auto heap_top_col = subwarp.shfl(b_col, heap - subwarp_base);
            auto min_child_col = reduce(subwarp, child_col, min_op);
            auto min_child_mask = subwarp.ballot(min_child_col == child_col);
            auto min_child = ffs(min_child_mask) - 1;

            // and update its local data if necessary
            if (min_child_col < heap_top_col) {
                sh_idx = sh_child_idx(heap, min_child);
                // if this is the warp holding heap's top in registers: swap
                if (warp_rank == heap) {
                    swap(b_col, sh_cols[sh_idx]);
                    swap(b_idx, sh_idxs[sh_idx]);
                    swap(b_end, sh_ends[sh_idx]);
                    if (use_val) {
                        swap(a_val, sh_vals[sh_idx]);
                    }
                }
                // update global memory heap with shared memory heap
                auto heap_idx = sh_to_gl(sh_idx);
                if (child_idx(heap_idx, 0) < a.size_) {
                    // find minimum child in global heap
                    auto cur_col =
                        checked_load(gl_cols, child_idx(heap_idx, subwarp_rank),
                                     a.size_, sentinel);
                    auto min_col = reduce(subwarp, cur_col, min_op);
                    // if heap property is violated: swap min child up
                    if (min_col < sh_cols[sh_idx]) {
                        min_child_mask = subwarp.ballot(min_col == cur_col);
                        min_child = ffs(min_child_mask) - 1;
                        auto new_heap_idx = child_idx(heap_idx, min_child);
                        if (subwarp_rank == 0) {
                            swap(sh_cols[sh_idx], gl_cols[new_heap_idx]);
                            swap(sh_idxs[sh_idx], gl_idxs[new_heap_idx]);
                            swap(sh_ends[sh_idx], gl_ends[new_heap_idx]);
                            if (use_val) {
                                swap(sh_vals[sh_idx], gl_vals[new_heap_idx]);
                            }
                        }
                        // memory barrier
                        subwarp.sync();
                        sift_down(new_heap_idx);
                    }
                }
            }
        });

        // call finish row callback if we are done
        auto new_min_col = reduce(warp, b_col, min_op);
        if (new_min_col > min_col) {
            acc.output_column(min_col);
        }
        min_col = new_min_col;
    }
}


template <typename ValueType, typename IndexType, typename AAccessor,
          typename Accumulator>
__device__ void spgemm_multiway_merge_dispatch_short(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    Accumulator &acc)
{
    if (a.size_ <= 2) {
        spgemm_multiway_merge_short<2>(a, b_row_ptrs, b_cols, b_vals, acc);
    } else if (a.size_ <= 4) {
        spgemm_multiway_merge_short<4>(a, b_row_ptrs, b_cols, b_vals, acc);
    } else if (a.size_ <= 8) {
        spgemm_multiway_merge_short<8>(a, b_row_ptrs, b_cols, b_vals, acc);
    } else if (a.size_ <= 16) {
        spgemm_multiway_merge_short<16>(a, b_row_ptrs, b_cols, b_vals, acc);
    } else if (a.size_ <= 32) {
        spgemm_multiway_merge_short<32>(a, b_row_ptrs, b_cols, b_vals, acc);
    } else {
        spgemm_multiway_merge_short<config::warp_size>(a, b_row_ptrs, b_cols,
                                                       b_vals, acc);
    }
}


template <typename ValueType, typename IndexType, typename AAccessor,
          typename Accumulator>
__device__ void spgemm_multiway_merge_dispatch_medium(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    Accumulator &acc)
{
    constexpr auto heap_size = config::warp_size * spgemm_child_count;
    constexpr auto warps_per_block = spgemm_block_size / config::warp_size;
    __shared__ IndexType sh_cols[heap_size * warps_per_block];
    __shared__ IndexType sh_idxs[heap_size * warps_per_block];
    __shared__ IndexType sh_ends[heap_size * warps_per_block];
    __shared__ UninitializedArray<ValueType, heap_size * warps_per_block>
        sh_vals;
    if (a.size_ <= config::warp_size) {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, b_vals,
                                             acc);
    } else {
        spgemm_multiway_merge_medium(a, b_row_ptrs, b_cols, b_vals, sh_cols,
                                     sh_idxs, sh_ends, &sh_vals[0], acc);
    }
}


template <typename ValueType, typename IndexType, typename AAccessor,
          typename Accumulator>
__device__ void spgemm_multiway_merge_dispatch(
    AAccessor a, const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *__restrict__ tmp, ValueType *__restrict__ tmpval,
    Accumulator &acc)
{
    constexpr auto heap_size = config::warp_size * spgemm_child_count;
    constexpr auto warps_per_block = spgemm_block_size / config::warp_size;
    __shared__ IndexType sh_cols[heap_size * warps_per_block];
    __shared__ IndexType sh_idxs[heap_size * warps_per_block];
    __shared__ IndexType sh_ends[heap_size * warps_per_block];
    __shared__ UninitializedArray<ValueType, heap_size * warps_per_block>
        sh_vals;

    if (a.size_ <= config::warp_size) {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, b_vals,
                                             acc);
    } else if (a.size_ <= config::warp_size * (spgemm_child_count + 1)) {
        spgemm_multiway_merge_medium(a, b_row_ptrs, b_cols, b_vals, sh_cols,
                                     sh_idxs, sh_ends, &sh_vals[0], acc);
    } else {
        spgemm_multiway_merge_large(
            a, b_row_ptrs, b_cols, b_vals, sh_cols, sh_idxs, sh_ends,
            &sh_vals[0], tmp, tmp + a.size_, tmp + 2 * a.size_, tmpval, acc);
    }
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void copy_row(AAccessor a, const IndexType *__restrict__ b_row_ptrs,
                         const IndexType *__restrict__ b_cols,
                         const ValueType *__restrict__ b_vals,
                         IndexType c_begin, IndexType *__restrict__ c_cols,
                         ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto a_col = a.col(0);
    auto a_val = a.val(0);
    auto b_begin = b_row_ptrs[a_col];
    auto b_size = b_row_ptrs[a_col + 1] - b_begin;
    for (IndexType i = warp.thread_rank(); i < b_size; i += config::warp_size) {
        c_cols[c_begin + i] = b_cols[b_begin + i];
        c_vals[c_begin + i] = a_val * b_vals[b_begin + i];
    }
}


template <typename IndexType, typename AAccessor>
__device__ IndexType count_merge_2way(AAccessor a,
                                      const IndexType *__restrict__ b_row_ptrs,
                                      const IndexType *__restrict__ b_cols)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto a_col1 = a.col(0);
    const auto a_col2 = a.col(1);
    const auto b_begin1 = b_row_ptrs[a_col1];
    const auto b_begin2 = b_row_ptrs[a_col2];
    const auto b_size1 = b_row_ptrs[a_col1 + 1] - b_begin1;
    const auto b_size2 = b_row_ptrs[a_col2 + 1] - b_begin2;
    IndexType nnz{};
    group_merge<config::warp_size>(
        b_cols + b_begin1, b_size1, b_cols + b_begin2, b_size2, warp,
        [&](IndexType, IndexType b_col1, IndexType, IndexType b_col2, IndexType,
            bool valid) {
            nnz += popcnt(warp.ballot(b_col1 != b_col2 && valid));
            return true;
        });
    return nnz;
}


template <typename ValueType, typename IndexType, typename AAccessor>
__device__ void merge_2way(AAccessor a,
                           const IndexType *__restrict__ b_row_ptrs,
                           const IndexType *__restrict__ b_cols,
                           const ValueType *__restrict__ b_vals, IndexType c_nz,
                           IndexType *__restrict__ c_cols,
                           ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto a_col1 = a.col(0);
    const auto a_col2 = a.col(1);
    const auto a_val1 = a.val(0);
    const auto a_val2 = a.val(1);
    const auto b_begin1 = b_row_ptrs[a_col1];
    const auto b_begin2 = b_row_ptrs[a_col2];
    const auto b_size1 = b_row_ptrs[a_col1 + 1] - b_begin1;
    const auto b_size2 = b_row_ptrs[a_col2 + 1] - b_begin2;
    bool skip_first = false;
    const auto lane = static_cast<IndexType>(warp.thread_rank());
    const auto lanemask_eq = config::lane_mask_type{1} << lane;
    const auto lanemask_lt = lanemask_eq - 1;
    group_merge<config::warp_size>(
        b_cols + b_begin1, b_size1, b_cols + b_begin2, b_size2, warp,
        [&](IndexType b_nz1, IndexType b_col1, IndexType b_nz2,
            IndexType b_col2, IndexType, bool valid) {
            auto c_col = min(b_col1, b_col2);
            auto valid_mask = warp.ballot(valid);
            auto equal_mask = warp.ballot(b_col1 == b_col2) & valid_mask;
            // check if the elements in the previous merge step are
            // equal
            auto prev_equal_mask = equal_mask << 1 | skip_first;
            // store the highest bit for the next group_merge_step
            skip_first = bool(equal_mask >> (config::warp_size - 1));
            auto prev_equal = bool(prev_equal_mask & lanemask_eq);
            // only output an entry if the previous cols weren't equal.
            // if they were equal, they were both handled in the
            // previous step
            if (valid && !prev_equal) {
                auto c_ofs = popcnt(~prev_equal_mask & lanemask_lt);
                c_cols[c_nz + c_ofs] = c_col;
                auto b_val1 = b_col1 <= b_col2 ? b_vals[b_nz1 + b_begin1]
                                               : zero<ValueType>();
                auto b_val2 = b_col2 <= b_col1 ? b_vals[b_nz2 + b_begin2]
                                               : zero<ValueType>();
                c_vals[c_nz + c_ofs] = a_val1 * b_val1 + a_val2 * b_val2;
            }
            // advance by the number of merged elements
            c_nz += popcnt(~prev_equal_mask & valid_mask);
            return true;
        });
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ tmp,
    IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    auto null_val = static_cast<float *>(nullptr);
    memory_row_accessor<float, IndexType> a{null_val, a_cols, a_begin, a_size};
    count_accumulator<IndexType> counter{};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        counter.count_ = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else if (a_size == 2) {
        counter.count_ = count_merge_2way(a, b_row_ptrs, b_cols);
    } else {
        spgemm_multiway_merge_dispatch(a, b_row_ptrs, b_cols, null_val,
                                       tmp + 3 * a_begin, null_val, counter);
    }
    if (write) {
        c_nnz[row] = counter.count_;
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count_short(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    auto null_val = static_cast<float *>(nullptr);
    memory_row_accessor<float, IndexType> a{null_val, a_cols, a_begin, a_size};
    count_accumulator<IndexType> counter{};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        counter.count_ = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else if (a_size == 2) {
        counter.count_ = count_merge_2way(a, b_row_ptrs, b_cols);
    } else {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, null_val,
                                             counter);
    }
    if (write) {
        c_nnz[row] = counter.count_;
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count_merge(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    auto null_val = static_cast<float *>(nullptr);
    merge_row_accessor<float, IndexType> a{a_begin, a_size};
    count_accumulator<IndexType> counter{};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a.col(0);
        counter.count_ = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else if (a_size == 2) {
        counter.count_ = count_merge_2way(a, b_row_ptrs, b_cols);
    } else {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, null_val,
                                             counter);
    }
    if (write) {
        c_nnz[row] = counter.count_;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols, const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ tmp,
    ValueType *__restrict__ tmpval, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    memory_row_accessor<ValueType, IndexType> a{a_vals, a_cols, a_begin,
                                                a_size};

    IndexType c_nz = c_row_ptrs[row];
    output_accumulator<ValueType, IndexType> acc{c_cols, c_vals, c_nz, write};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else if (a_size == 2) {
        merge_2way(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        spgemm_multiway_merge_dispatch(a, b_row_ptrs, b_cols, b_vals,
                                       tmp + a_begin * 3, tmpval + a_begin,
                                       acc);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel_short(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols, const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    memory_row_accessor<ValueType, IndexType> a{a_vals, a_cols, a_begin,
                                                a_size};

    IndexType c_nz = c_row_ptrs[row];
    output_accumulator<ValueType, IndexType> acc{c_cols, c_vals, c_nz, write};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else if (a_size == 2) {
        merge_2way(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, b_vals,
                                             acc);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel_merge(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    if (row >= num_rows) {
        return;
    }

    const auto write = threadIdx.x % config::warp_size == 0;
    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    merge_row_accessor<ValueType, IndexType> a{a_begin, a_size};

    IndexType c_nz = c_row_ptrs[row];
    output_accumulator<ValueType, IndexType> acc{c_cols, c_vals, c_nz, write};
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        copy_row(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else if (a_size == 2) {
        merge_2way(a, b_row_ptrs, b_cols, b_vals, c_nz, c_cols, c_vals);
    } else {
        spgemm_multiway_merge_dispatch_short(a, b_row_ptrs, b_cols, b_vals,
                                             acc);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_merge_counts(
    const IndexType *__restrict__ a_row_ptrs, size_type num_rows,
    int merge_size, IndexType *__restrict__ merge_count)
{
    const auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }

    const auto size = a_row_ptrs[row + 1] - a_row_ptrs[row];
    merge_count[row] = ceildiv(size, merge_size);
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_tall_row_ptrs(
    const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ row_size_prefixsum, size_type num_rows,
    int merge_size, IndexType *__restrict__ tall_row_ptrs)
{
    const auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }

    const auto begin = a_row_ptrs[row];
    const auto end = a_row_ptrs[row + 1];
    auto out_idx = row_size_prefixsum[row];

    for (auto i = begin; i < end; i += merge_size) {
        tall_row_ptrs[out_idx] = i;
        ++out_idx;
    }
    if (row == num_rows - 1) {
        tall_row_ptrs[out_idx] = end;
    }
}
