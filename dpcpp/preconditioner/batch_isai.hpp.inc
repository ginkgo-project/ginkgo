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


// namespace {

// TODO: find a way to use single_matvec_kernel in batch_csr_kernels.hpp.inc
template <typename ValueType>
__dpct_inline__ void single_csr_spmv_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c,
    sycl::nd_item<3> item_ct1)
{
    const auto num_rows = a.num_rows;
    const auto val = a.values;
    const auto col = a.col_idxs;
    for (int tidx = item_ct1.get_local_linear_id(); tidx < num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (auto nnz = a.row_ptrs[tidx]; nnz < a.row_ptrs[tidx + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        c[tidx] = temp;
    }
}


//}  // namespace


/**
 * Batch isai preconditioner.
 */
template <typename ValueType>
class batch_isai final {
private:
    __dpct_inline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id)
    {
        approx_inv_entry_ =
            gko::batch::batch_entry(approx_inv_batch_, batch_id);
    }

public:
    using value_type = ValueType;

    /**
     *
     * @param approx_inv  Incomplete Sparse Approximate Inverse (that was
     * factored externally).
     */
    batch_isai(const gko::batch_csr::UniformBatch<const value_type>& approx_inv)
        : approx_inv_batch_{approx_inv}
    {}


    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    __dpct_inline__ static constexpr int dynamic_work_size(int nrows, int nnz)
    {
        return 0;
    }

    /**
     * Complete the precond generation process.
     *
     */
    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __dpct_inline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __dpct_inline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&, ValueType* const,
        sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }


    __dpct_inline__ void apply(const int num_rows,
                               const ValueType* const __restrict__ r,
                               ValueType* const __restrict__ z,
                               sycl::nd_item<3> item_ct1) const
    {
        // z = precond * r  ==> z = approx_inv * r (SpMV)
        single_csr_spmv_kernel(approx_inv_entry_, r, z, item_ct1);
    }

private:
    const gko::batch_csr::UniformBatch<const value_type> approx_inv_batch_;
    gko::batch_csr::BatchEntry<const value_type> approx_inv_entry_;
};


template <typename ValueType>
__dpct_inline__ void batch_isai_apply(batch_isai<ValueType> prec,
                                      const size_type nbatch, const int nrows,
                                      const ValueType* const b,
                                      ValueType* const x,
                                      ValueType* const sh_mem,
                                      sycl::nd_item<3> item_ct1)
{
    auto batch_id = item_ct1.get_group_linear_id();
    prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                  sh_mem, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    prec.apply(nrows, b, x, item_ct1);
}
