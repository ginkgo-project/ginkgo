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
#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_
#define GKO_DPCPP_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_

#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/matrix/batch_csr_kernels.hpp"
#include "dpcpp/matrix/batch_dense_kernels.hpp"
#include "dpcpp/matrix/batch_ell_kernels.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {

/**
 * (Scalar) Jacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchScalarJacobi final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
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
     *              of memory given by dynamic_work_size.
     */
    __dpct_inline__ void generate(
        const gko::batch_ell::BatchEntry<const ValueType>& mat,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        const auto col = mat.col_idxs;
        const auto val = mat.values;
        work_ = work;
        for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
             tidx += item_ct1.get_local_range().size()) {
            auto temp = one<ValueType>();
            for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
                const auto ind = tidx + idx * mat.stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    const bool found = (col_idx == tidx);
                    if (found) {
                        temp = one<ValueType>() / val[ind];
                    }
                }
            }
            work_[tidx] = temp;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by dynamic_work_size.
     */
    __dpct_inline__ void generate(
        const gko::batch_csr::BatchEntry<const ValueType>& mat,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        work_ = work;
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = sg.get_group_id();
        const int sg_size = sg.get_local_range().size();
        const int num_sg = sg.get_group_range().size();

        for (int irow = sg_id; irow < mat.num_rows; irow += num_sg) {
            for (int j = mat.row_ptrs[irow] + sg.get_local_id();
                 j < mat.row_ptrs[irow + 1]; j += sg_size) {
                const int found = (mat.col_idxs[j] == irow);
                if (found) {
                    work_[irow] = (mat.values[j] != zero<ValueType>())
                                      ? one<ValueType>() / mat.values[j]
                                      : one<ValueType>();
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    /**
     * Sets the input and generates the preconditioner by storing the inverse
     * diagonal entries in the work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the inverse diagonal
     *              entries. It must be allocated with at least the amount
     *              of memory given by dynamic_work_size.
     */
    __dpct_inline__ void generate(
        const gko::batch_dense::BatchEntry<const ValueType>& mat,
        ValueType* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        work_ = work;
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = sg.get_group_id();
        const int sg_size = sg.get_local_range().size();
        const int num_sg = sg.get_group_range().size();

        for (int irow = sg_id; irow < mat.num_rows; irow += num_sg) {
            const int iz = irow * static_cast<int>(mat.stride) + irow;
            work_[irow] = (mat.values[iz] != zero<ValueType>())
                              ? one<ValueType>() / mat.values[iz]
                              : one<ValueType>();
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    __dpct_inline__ void apply(const int num_rows, const ValueType* const r,
                               ValueType* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        for (int i = item_ct1.get_local_linear_id(); i < num_rows;
             i += item_ct1.get_local_range().size()) {
            z[i] = work_[i] * r[i];
        }
    }

private:
    ValueType* __restrict__ work_;
};

// Here: all sys_mat_batch, b_values, a_values are calculated for each batch
// before passing into this function
template <typename BatchMatrixEntryType, typename ValueType>
void batch_scalar_jacobi_apply(BatchScalarJacobi<ValueType> prec,
                               const BatchMatrixEntryType& sys_mat_batch,
                               const int nrows, const ValueType* const b_values,
                               ValueType* const x_values,
                               ValueType* const __restrict__ work,
                               sycl::nd_item<3> item_ct1)
{
    prec.generate(sys_mat_batch, work);
    prec.apply(nrows, b_values, x_values, item_ct1);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif
