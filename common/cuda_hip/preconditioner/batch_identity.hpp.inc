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
 *  Identity preconditioner for batch solvers. ( To be able to have
 * unpreconditioned solves )
 */
template <typename ValueType>
class BatchIdentity final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 0;

    /**
     * The size of the work vector required in case of dynamic allocation.
     *
     * For the Identity preconditioner, this is unnecessary, but this function
     * is part of a 'batch preconditioner interface' because other
     * preconditioners may need it.
     */
    __host__ __device__ static constexpr int dynamic_work_size(int, int)
    {
        return 0;
    }


    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     *
     * @param mat  Matrix for which to build an Ideniity preconditioner.
     * @param work  A 'work-vector', which is unneecessary here as no
     * preconditioner values are to be stored.
     */
    __device__ __forceinline__ void generate(
        size_type, const gko::batch_ell::BatchEntry<const ValueType>& mat,
        ValueType*)
    {}


    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     *
     * @param mat  Matrix for which to build an Ideniity preconditioner.
     * @param work  A 'work-vector', which is unneecessary here as no
     * preconditioner values are to be stored.
     */
    __device__ __forceinline__ void generate(
        size_type, const gko::batch_csr::BatchEntry<const ValueType>& mat,
        ValueType*)
    {}

    __device__ __forceinline__ void generate(
        size_type, const gko::batch_dense::BatchEntry<const ValueType>& mat,
        ValueType*)
    {}

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const r,
                                          ValueType* const z) const
    {
        for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
            z[li] = r[li];
        }
    }
};


template <typename ValueType>
__global__ void batch_identity(
    BatchIdentity<ValueType> prec,
    const gko::batch_ell::UniformBatch<const ValueType> a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    for (size_type batch = blockIdx.x; batch < a.num_batch;
         batch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, batch);
        const auto b_b = gko::batch::batch_entry(b, batch);
        const auto x_b = gko::batch::batch_entry(x, batch);

        extern __shared__ char local_mem[];
        const auto prec_work = reinterpret_cast<ValueType*>(local_mem);
        prec.generate(batch, a_b, prec_work);
        prec.apply(b_b.num_rows, b_b.values, x_b.values);
    }
}


template <typename ValueType>
__global__ void batch_identity(
    BatchIdentity<ValueType> prec,
    const gko::batch_csr::UniformBatch<const ValueType> a,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    for (size_type batch = blockIdx.x; batch < a.num_batch;
         batch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, batch);
        const auto b_b = gko::batch::batch_entry(b, batch);
        const auto x_b = gko::batch::batch_entry(x, batch);

        extern __shared__ char local_mem[];
        const auto prec_work = reinterpret_cast<ValueType*>(local_mem);
        prec.generate(batch, a_b, prec_work);
        prec.apply(b_b.num_rows, b_b.values, x_b.values);
    }
}
