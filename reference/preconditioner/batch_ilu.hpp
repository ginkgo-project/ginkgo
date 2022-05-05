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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


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
class batch_ilu_split final {
public:
    using value_type = ValueType;
    using factorization = Factorization;
    using trsv = Trsv;

    /**
     * Set the factorization and triangular solver to use.
     *
     * @param l_factor  Lower-triangular factor that was externally generated.
     * @param u_factor  Upper-triangular factor that was externally generated.
     * @param trisolve  The triangular solver to use. Currently, it must be
     *                  pre-generated.
     */
    batch_ilu_split(
        const gko::batch_csr::UniformBatch<const value_type>& l_factor,
        const gko::batch_csr::UniformBatch<const value_type>& u_factor,
        const trsv& trisolve, const factorization& factors = factorization{})
        : l_factor_{l_factor}, u_factor_{u_factor}, trsv_{trisolve}
    {}

    /**
     * The size of the work vector required per batch entry.
     */
    static constexpr int dynamic_work_size(int, int) { return 0; }

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
    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType*)
    {
        auto batch_L = gko::batch::batch_entry(l_factor_, batch_id);
        auto batch_U = gko::batch::batch_entry(u_factor_, batch_id);
        trsv_.generate(batch_L, batch_U);
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType*)
    {}

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType*)
    {}

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        trsv_.apply(r, z);
    }

private:
    gko::batch_csr::UniformBatch<const value_type> l_factor_;
    gko::batch_csr::UniformBatch<const value_type> u_factor_;
    trsv trsv_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_
