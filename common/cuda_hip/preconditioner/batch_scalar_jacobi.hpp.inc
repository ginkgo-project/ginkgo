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
    __host__ __device__ static constexpr int dynamic_work_size(
        const int num_rows, int)
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
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::ell::batch_item<const value_type,
                                                  const index_type>& mat,
        value_type* const __restrict__ work)
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
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::csr::batch_item<const value_type,
                                                  const index_type>& mat,
        value_type* const __restrict__ work)
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
    __device__ __forceinline__ void generate(
        size_type,
        const gko::batch::matrix::dense::batch_item<const value_type>& mat,
        value_type* const __restrict__ work)
    {}

    __device__ __forceinline__ void apply(const int num_rows,
                                          const value_type* const r,
                                          value_type* const z) const
    {}

private:
    value_type* __restrict__ work_;
};
