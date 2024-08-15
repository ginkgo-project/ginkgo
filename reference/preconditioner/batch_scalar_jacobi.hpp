// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace host {
namespace batch_preconditioner {


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
    static int dynamic_work_size(const int nrows, int)
    {
        return nrows * sizeof(ValueType);
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
    void generate(size_type,
                  const gko::batch::matrix::ell::batch_item<
                      const value_type, const index_type>& mat,
                  value_type* const work)
    {
        work_ = work;
        for (int i = 0; i < mat.num_rows; i++) {
            work_[i] = one<value_type>();
            for (int j = 0; j < mat.num_stored_elems_per_row; j++) {
                const auto idx = i + j * mat.stride;
                if (mat.col_idxs[idx] == i) {
                    if (mat.values[idx] != zero<value_type>()) {
                        work_[i] = one<value_type>() / mat.values[idx];
                    }
                    break;
                }
            }
        }
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
    void generate(size_type,
                  const gko::batch::matrix::csr::batch_item<
                      const value_type, const index_type>& mat,
                  value_type* const work)
    {
        work_ = work;
        for (int i = 0; i < mat.num_rows; i++) {
            work_[i] = one<value_type>();
            for (int j = mat.row_ptrs[i]; j < mat.row_ptrs[i + 1]; j++) {
                if (mat.col_idxs[j] == i) {
                    if (mat.values[j] != zero<value_type>()) {
                        work_[i] = one<value_type>() / mat.values[j];
                    }
                    break;
                }
            }
        }
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
    void generate(
        size_type,
        const gko::batch::matrix::dense::batch_item<const value_type>& mat,
        value_type* const work)
    {
        work_ = work;
        for (int i = 0; i < mat.num_rows; i++) {
            work_[i] = one<value_type>() /
                       (mat.values[i * mat.stride + i] == zero<value_type>()
                            ? one<value_type>()
                            : mat.values[i * mat.stride + i]);
        }
    }

    /**
     * Set to unit diagonal for external matrices
     */
    void generate(
        size_type,
        const gko::batch::matrix::external::batch_item<const value_type>& mat,
        value_type* const work)
    {
        work_ = work;
        for (int i = 0; i < mat.num_rows; i++) {
            work_[i] = one<value_type>();
        }
    }

    void apply(const gko::batch::multi_vector::batch_item<const value_type>& r,
               const gko::batch::multi_vector::batch_item<value_type>& z) const
    {
        for (int i = 0; i < r.num_rows; i++) {
            for (int j = 0; j < r.num_rhs; j++) {
                z.values[i * z.stride + j] =
                    work_[i] * r.values[i * r.stride + j];
            }
        }
    }

private:
    value_type* work_ = nullptr;
};


}  // namespace batch_preconditioner
}  // namespace host
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_SCALAR_JACOBI_HPP_
