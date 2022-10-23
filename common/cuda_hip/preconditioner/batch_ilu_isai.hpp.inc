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


namespace batch_ilu_isai_temp {

// TODO: find a way to use single_matvec_kernel in batch_csr_kernels.hpp.inc
template <typename ValueType>
__device__ __forceinline__ void single_csr_spmv_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    const auto num_rows = a.num_rows;
    const auto val = a.values;
    const auto col = a.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (auto nnz = a.row_ptrs[tidx]; nnz < a.row_ptrs[tidx + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        c[tidx] = temp;
    }
}

}  // namespace batch_ilu_isai_temp

/**
 * Batch ilu-isai preconditioner.
 */
template <typename ValueType>
class batch_ilu_isai final {
public:
    using value_type = ValueType;

    /**
     * @param l_isai_batch  Lower Triangular factor's Incomplete Sparse
     * Approximate Inverse (that was generated externally).
     *
     * @param u_isai_batch  Upper Triangular factor's Incomplete Sparse
     * Approximate Inverse (that was generated externally).
     *
     * @param mult_inv_batch Mutiplication of inverses:  u_isai_batch *
     * l_isai_batch (generated externally).
     *
     * @param apply_type How the preconditioner is to be applied? (0:
     * simple_spmvs, 1: inv_factors_spgemm , 2: relaxation_steps)
     */
    batch_ilu_isai(
        const gko::batch_csr::UniformBatch<const value_type>& l_isai_batch,
        const gko::batch_csr::UniformBatch<const value_type>& u_isai_batch,
        const gko::batch_csr::UniformBatch<const value_type>& mult_inv_batch,
        const int apply_type)
        : l_isai_batch_{l_isai_batch},
          u_isai_batch_{u_isai_batch},
          mult_inv_batch_{mult_inv_batch},
          apply_type_{apply_type}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application (for application, returns the max. of
     * what is required by each of the 3 methods))
     */
    __host__ __device__ static constexpr int dynamic_work_size(int nrows,
                                                               int nnz)
    {
        return nrows;
    }

    /**
     * Complete the precond generation process.
     *
     */
    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        work_ = work;
        l_isai_entry_ = gko::batch::batch_entry(l_isai_batch_, batch_id);
        u_isai_entry_ = gko::batch::batch_entry(u_isai_batch_, batch_id);
        if (apply_type_ == 1) {
            mult_inv_entry_ =
                gko::batch::batch_entry(mult_inv_batch_, batch_id);
        }
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        work_ = work;
        l_isai_entry_ = gko::batch::batch_entry(l_isai_batch_, batch_id);
        u_isai_entry_ = gko::batch::batch_entry(u_isai_batch_, batch_id);
        if (apply_type_ == 1) {
            mult_inv_entry_ =
                gko::batch::batch_entry(mult_inv_batch_, batch_id);
        }
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        work_ = work;
        l_isai_entry_ = gko::batch::batch_entry(l_isai_batch_, batch_id);
        u_isai_entry_ = gko::batch::batch_entry(u_isai_batch_, batch_id);
        if (apply_type_ == 1) {
            mult_inv_entry_ =
                gko::batch::batch_entry(mult_inv_batch_, batch_id);
        }
    }


    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        // z = precond * r  ==> L * U * z = r ==> lai_U * lai_L * L * U * z =
        // lai_U * laiL * r ===> z = lai_U * laiL * r
        if (apply_type_ == 0)  // simple_spmvs
        {
            batch_ilu_isai_temp::single_csr_spmv_kernel(l_isai_entry_, r,
                                                        work_);
            __syncthreads();
            batch_ilu_isai_temp::single_csr_spmv_kernel(u_isai_entry_, work_,
                                                        z);
        } else if (apply_type_ == 1)  // inv_factors_spgemm
        {
            batch_ilu_isai_temp::single_csr_spmv_kernel(mult_inv_entry_, r, z);
        } else if (apply_type_ == 2)  // relaxation steps
        {
            printf("\n Relaxation steps- Not implemented");
        } else {
            printf("\n Not implemented");
        }
    }

private:
    value_type* __restrict__ work_;
    const int apply_type_;
    const gko::batch_csr::UniformBatch<const value_type> l_isai_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_isai_entry_;
    const gko::batch_csr::UniformBatch<const value_type> u_isai_batch_;
    gko::batch_csr::BatchEntry<const value_type> u_isai_entry_;
    const gko::batch_csr::UniformBatch<const value_type> mult_inv_batch_;
    gko::batch_csr::BatchEntry<const value_type> mult_inv_entry_;
};


template <typename ValueType>
__global__ void batch_ilu_isai_apply(batch_ilu_isai<ValueType> prec,
                                     const size_type nbatch, const int nrows,
                                     const ValueType* const b_values,
                                     ValueType* const x_values)
{
    for (size_type batch_id = blockIdx.x; batch_id < nbatch;
         batch_id += gridDim.x) {
        extern __shared__ char sh_mem[];
        ValueType* work = reinterpret_cast<ValueType*>(sh_mem);

        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                      work);
        __syncthreads();
        prec.apply(nrows, b_values + batch_id * nrows,
                   x_values + batch_id * nrows);
    }
}
