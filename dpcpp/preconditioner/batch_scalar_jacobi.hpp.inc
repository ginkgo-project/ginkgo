// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * (Scalar) Jacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class ScalarJacobi final {
public:
    using value_type = ValueType;
    using index_type = int;

    /**
     * The size of the work vector(in bytes) required in case of dynamic
     * allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
    {
        return num_rows * sizeof(value_type);
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
    __dpct_inline__ void generate(size_type,
                                  const batch::matrix::ell::batch_item<
                                      const value_type, const index_type>& mat,
                                  value_type* const __restrict__ work,
                                  sycl::nd_item<3> item_ct1)
    {
        const auto col = mat.col_idxs;
        const auto val = mat.values;
        work_ = work;
        for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
             tidx += item_ct1.get_local_range().size()) {
            auto temp = one<value_type>();
            for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
                const auto ind = tidx + idx * mat.stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    const bool found = (col_idx == tidx);
                    if (found) {
                        temp = one<value_type>() / val[ind];
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
    __dpct_inline__ void generate(size_type,
                                  const batch::matrix::csr::batch_item<
                                      const value_type, const index_type>& mat,
                                  value_type* const __restrict__ work,
                                  sycl::nd_item<3> item_ct1)
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
                    work_[irow] = (mat.values[j] != zero<value_type>())
                                      ? one<value_type>() / mat.values[j]
                                      : one<value_type>();
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
        size_type,
        const batch::matrix::dense::batch_item<const value_type>& mat,
        value_type* const __restrict__ work, sycl::nd_item<3> item_ct1)
    {
        work_ = work;
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = sg.get_group_id();
        const int sg_size = sg.get_local_range().size();
        const int num_sg = sg.get_group_range().size();

        for (int irow = sg_id; irow < mat.num_rows; irow += num_sg) {
            const int iz = irow * static_cast<int>(mat.stride) + irow;
            work_[irow] = (mat.values[iz] != zero<value_type>())
                              ? one<value_type>() / mat.values[iz]
                              : one<value_type>();
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    __dpct_inline__ void apply(const int num_rows, const value_type* const r,
                               value_type* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        for (int i = item_ct1.get_local_linear_id(); i < num_rows;
             i += item_ct1.get_local_range().size()) {
            z[i] = work_[i] * r[i];
        }
    }

private:
    value_type* __restrict__ work_;
};
