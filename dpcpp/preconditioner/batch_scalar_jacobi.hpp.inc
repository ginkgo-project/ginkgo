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
    __dpct_inline__ void generate(size_type,
                                  const batch::matrix::ell::batch_item<
                                      const value_type, const index_type>& mat,
                                  value_type* const __restrict__ work,
                                  sycl::nd_item<3> item_ct1)
    {}

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
    {}

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
    {}

    __dpct_inline__ void apply(const int num_rows, const value_type* const r,
                               value_type* const z,
                               sycl::nd_item<3> item_ct1) const
    {}

private:
    value_type* __restrict__ work_;
};
