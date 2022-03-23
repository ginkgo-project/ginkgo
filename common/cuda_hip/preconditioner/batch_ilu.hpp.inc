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
 * Dummy device factorization to use for those factorizations that are only
 * implemented as separate kernel launches.
 */
class BatchDummyFactorization {};


/**
 * Batch preconditioner based on the application of a matrix factorization.
 *
 * \tparam Trsv  The triangular solve method to use.
 * \tparam Factorization  The matrix factorization method to use. This is
 *   currently unused because device-side factorization is not implemented.
 */
template <typename ValueType, typename Trsv,
          typename Factorization = BatchDummyFactorization>
class BatchIlu final {
public:
    using value_type = ValueType;
    using factorization = Factorization;
    using trsv = Trsv;

    /**
     * Set the factorization and triangular solver to use.
     *
     * @param trisolve  The triangular solver to use. Currently, it must be
     *                  pre-generated.
     */
    BatchIlu(const factorization&, const trsv& trisolve) : trsv_{trisolve} {}

    /**
     * The size of the work vector required per batch entry.
     */
    __host__ __device__ static constexpr int dynamic_work_size(int, int)
    {
        return 0;
    }

    // __host__ __device__ static constexpr int dynamic_work_size(
    //     const int num_rows, const int nnz)
    // {
    //     return factorization::dynamic_work_size(num_rows, nnz)
    //         + trsv::dynamic_work_size(num_rows, nnz);
    //     // If generation of trsv can be done in-place, just use max.
    //     // return max(factorization::dynamic_work_size(num_rows, nnz),
    //     //     trsv::dynamic_work_size(num_rows, nnz));
    // }

    /**
     * Generates the preconditioner by calling the device factorization.
     *
     * @param mat  Matrix for which to build an ILU-type preconditioner.
     */
    __device__ __forceinline__ void generate(
        const gko::batch_csr::BatchEntry<const ValueType>&, ValueType*)
    {}

    __device__ __forceinline__ void apply(int, const ValueType* const r,
                                          ValueType* const z) const
    {
        trsv_.apply(r, z);
    }

private:
    trsv trsv_;
};


template <typename Factorization, typename Trsv, typename ValueType>
__global__ void batch_ilu(
    BatchIlu<Factorization, Trsv, ValueType> prec,
    const gko::batch_csr::UniformBatch<const ValueType> a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    for (size_type batch = 0; batch < a.num_batch; ++batch) {
        const auto a_b = gko::batch::batch_entry(a, batch);
        const auto b_b = gko::batch::batch_entry(b, batch);
        const auto x_b = gko::batch::batch_entry(x, batch);

        extern __shared__ char local_mem[];
        const auto prec_work = reinterpret_cast<ValueType*>(local_mem);
        prec.apply(b_b.num_rows, b_b.values, x_b.values);
    }
}
