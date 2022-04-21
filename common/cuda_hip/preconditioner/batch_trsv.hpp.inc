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


/**
 * Exact batch triangular solves with separately stored L and U matrices.
 */
template <typename ValueType>
class BatchExactTrsvSeparate {
public:
    using value_type = ValueType;

    __host__ __device__ static constexpr int dynamic_work_size(
        const int num_rows, const int nnz)
    {
        return 0;
        // for a more parallel variant that needs an intermediate vector:
        // return num_rows;
    }

    /*
     * Set local triangular factors.
     */
    __device__ __forceinline__ void generate(
        const gko::batch_csr::BatchEntry<const ValueType>& l_factor,
        const gko::batch_csr::BatchEntry<const ValueType>& u_factor)
    {
        l_factor_ = l_factor;
        u_factor_ = u_factor;
    }

    __device__ __forceinline__ void apply(const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        auto tile_grp = group::tiled_partition<config::warp_size>(
            group::this_thread_block());
        if (threadIdx.x / config::warp_size > 0) {
            return;
        }
        for (int i = 0; i < l_factor_.num_rows; i++) {
            ValueType sum{};
            for (int iz = l_factor_.row_ptrs[i];
                 iz < l_factor_.row_ptrs[i + 1] - 1; iz += config::warp_size) {
                ValueType val =
                    l_factor_.values[iz] * z[l_factor_.col_idxs[iz]];
                ValueType sumit =
                    reduce(tile_grp, val,
                           [](ValueType a, ValueType b) { return a + b; });
                if (tile_grp.thread_rank() == 0) {
                    sum += sumit;
                }
            }
            if (tile_grp.thread_rank() == 0) {
                z[i] = (r[i] - sum) /
                       l_factor_.values[l_factor_.row_ptrs[i + 1] - 1];
            }
            tile_grp.sync();
        }
        for (int i = u_factor_.num_rows - 1; i >= 0; i--) {
            ValueType sum{};
            for (int iz = u_factor_.row_ptrs[i] + 1;
                 iz < u_factor_.row_ptrs[i + 1]; iz += config::warp_size) {
                ValueType val =
                    u_factor_.values[iz] * z[u_factor_.col_idxs[iz]];
                ValueType sumit =
                    reduce(tile_grp, val,
                           [](ValueType a, ValueType b) { return a + b; });
                if (tile_grp.thread_rank() == 0) {
                    sum += sumit;
                }
            }
            if (tile_grp.thread_rank() == 0) {
                z[i] = (z[i] - sum) / u_factor_.values[u_factor_.row_ptrs[i]];
            }
            tile_grp.sync();
        }
    }

private:
    batch_csr::BatchEntry<const ValueType> l_factor_;
    batch_csr::BatchEntry<const ValueType> u_factor_;
};


template <typename ValueType>
__global__ void apply(const int num_rows, const int l_nnz,
                      const int* const __restrict__ l_row_ptrs,
                      const int* const __restrict__ l_col_idxs,
                      const ValueType* const __restrict__ l_values,
                      const int u_nnz, const int* const __restrict__ u_row_ptrs,
                      const int* const __restrict__ u_col_idxs,
                      const ValueType* const __restrict__ u_values,
                      const ValueType* const __restrict__ r,
                      ValueType* const __restrict__ x)
{}
