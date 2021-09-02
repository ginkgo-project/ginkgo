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


/**
 * (Scalar) Jacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchJacobi final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = batch_config<ValueType>::max_num_rows;

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    __host__ __device__ static constexpr int dynamic_work_size(
        const int num_rows, int)
    {
        return num_rows;
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by work_size or dynamic_work_size.
     */
    __device__ __forceinline__ void generate(
        const gko::batch_csr::BatchEntry<const ValueType>& mat,
        ValueType* const __restrict__ work)
    {
        work_ = work;
        constexpr auto warp_size = config::warp_size;
        const auto tile =
            group::tiled_partition<warp_size>(group::this_thread_block());
        const int tile_rank = threadIdx.x / warp_size;
        const int num_tiles = (blockDim.x - 1) / warp_size + 1;
        for (int irow = tile_rank; irow < mat.num_rows; irow += num_tiles) {
            for (int j = mat.row_ptrs[irow] + tile.thread_rank();
                 j < mat.row_ptrs[irow + 1]; j += warp_size) {
                const int found = (mat.col_idxs[j] == irow);
                if (found) {
                    work_[irow] = one<ValueType>() / mat.values[j];
                }
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void apply(
        const gko::batch_dense::BatchEntry<const ValueType>& r,
        const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        for (int i = threadIdx.x; i < r.num_rows * r.num_rhs; i += blockDim.x) {
            const int row = i / r.num_rhs;
            const int col = i % r.num_rhs;
            z.values[row * z.stride + col] =
                work_[row] * r.values[row * r.stride + col];
        }
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const r,
                                          ValueType* const z) const
    {
        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            z[i] = work_[i] * r[i];
        }
    }

private:
    ValueType* __restrict__ work_;
};


template <typename ValueType>
__global__ void batch_jacobi(
    BatchJacobi<ValueType> prec,
    const gko::batch_csr::UniformBatch<const ValueType> a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    for (size_type batch = 0; batch < a.num_batch; ++batch) {
        const auto a_b = gko::batch::batch_entry(a, batch);
        const auto b_b = gko::batch::batch_entry(b, batch);
        const auto x_b = gko::batch::batch_entry(x, batch);

        //__shared__ ValueType prec_work[BatchJacobi<ValueType>::work_size];
        extern __shared__ char local_mem[];
        const auto prec_work = reinterpret_cast<ValueType*>(local_mem);
        prec.generate(a_b, prec_work);
        prec.apply(b_b, x_b);
    }
}
