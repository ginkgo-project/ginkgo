// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_


#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/identity.hpp>


namespace gko {
namespace batch {
namespace matrix {


/**
 * The batch Identity matrix, which represents a batch of Identity matrices.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup batch_identity
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class Identity final : public EnableBatchLinOp<Identity<ValueType>>,
                       public EnableCreateMethod<Identity<ValueType>> {
    friend class EnableCreateMethod<Identity>;
    friend class EnablePolymorphicObject<Identity, BatchLinOp>;

public:
    using value_type = ValueType;
    using index_type = int32;
    using unbatch_type = gko::matrix::Identity<ValueType>;
    using absolute_type = remove_complex<Identity>;
    using complex_type = to_complex<Identity>;

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = I * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    Identity* apply(ptr_param<const MultiVector<value_type>> b,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * Apply the matrix to a multi-vector with a linear combination of the given
     * input vector. Represents the matrix vector multiplication, x = alpha * I
     * * b + beta * x, where x and b are both multi-vectors.
     *
     * @param alpha  the scalar to scale the matrix-vector product with
     * @param b      the multi-vector to be applied to
     * @param beta   the scalar to scale the x vector with
     * @param x      the output multi-vector
     */
    Identity* apply(ptr_param<const MultiVector<value_type>> alpha,
                    ptr_param<const MultiVector<value_type>> b,
                    ptr_param<const MultiVector<value_type>> beta,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const Identity* apply(ptr_param<const MultiVector<value_type>> b,
                          ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const Identity* apply(ptr_param<const MultiVector<value_type>> alpha,
                          ptr_param<const MultiVector<value_type>> b,
                          ptr_param<const MultiVector<value_type>> beta,
                          ptr_param<MultiVector<value_type>> x) const;

private:
    /**
     * Creates an Identity matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     */
    Identity(std::shared_ptr<const Executor> exec,
             const batch_dim<2>& size = batch_dim<2>{});

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_
