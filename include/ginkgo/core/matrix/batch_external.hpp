// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_EXTERNAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_EXTERNAL_HPP_


#include <initializer_list>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace batch {
namespace matrix {
namespace external_apply {


// Use only void* to pass-in data. This is necessary, because
// each backend uses different complex/half types.
using simple_type = void (*)(gko::size_type id, dim<2> size, const void* b,
                             void* x, void* payload);
using advanced_type = void (*)(gko::size_type id, dim<2> size,
                               const void* alpha, const void* b,
                               const void* beta, void* x, void* payload);


}  // namespace external_apply

// struct ExternalOperation {
//     __device__ virtual void simple_apply_cpu(gko::size_type id, dim<2> size,
//                                              const void* b, void* x,
//                                              void* payload)
//     {}
//
//     __device__ virtual void simple_apply_cuda(gko::size_type id, dim<2> size,
//                                               const void* b, void* x,
//                                               void* payload)
//     {}
//
//     __device__ virtual void simple_apply_hip(gko::size_type id, dim<2> size,
//                                              const void* b, void* x,
//                                              void* payload)
//     {}
//
//     __device__ virtual void simple_apply_sycl(gko::size_type id, dim<2> size,
//                                               const void* b, void* x,
//                                               void* payload)
//     {}
//
//     __device__ virtual void advanced_apply_cpu(gko::size_type id, dim<2>
//     size,
//                                                const void* alpha, const void*
//                                                b, const void* beta, void* x,
//                                                void* payload)
//     {}
//
//     __device__ virtual void advanced_apply_cuda(gko::size_type id, dim<2>
//     size,
//                                                 const void* alpha,
//                                                 const void* b, const void*
//                                                 beta, void* x, void* payload)
//     {}
//
//     __device__ virtual void advanced_apply_hip(gko::size_type id, dim<2>
//     size,
//                                                const void* alpha, const void*
//                                                b, const void* beta, void* x,
//                                                void* payload)
//     {}
//
//     __device__ virtual void advanced_apply_sycl(gko::size_type id, dim<2>
//     size,
//                                                 const void* alpha,
//                                                 const void* b, const void*
//                                                 beta, void* x, void* payload)
//     {}
// };


/**
 * Matrix format that uses externally provided function pointer for the
 * individual batch application.
 *
 * @tparam ValueType  value precision of matrix elements
 * @tparam IndexType  index precision of matrix elements
 *
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class External final : public EnableBatchLinOp<External<ValueType>> {
    friend class EnablePolymorphicObject<External, BatchLinOp>;

public:
    using value_type = ValueType;

    template <typename Func>
    struct functor_operation {
        Func cpu_apply = nullptr;
        Func cuda_apply = nullptr;
        Func hip_apply = nullptr;
        Func sycl_apply = nullptr;
    };

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_elems_per_row  the number of elements to be stored in each row
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<External> create(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        functor_operation<external_apply::simple_type> simple_apply,
        functor_operation<external_apply::advanced_type> advanced_apply,
        void* payload);

    /**
     * Apply the matrix to a multi-vector. Represents the matrix vector
     * multiplication, x = A * b, where x and b are both multi-vectors.
     *
     * @param b  the multi-vector to be applied to
     * @param x  the output multi-vector
     */
    External* apply(ptr_param<const MultiVector<value_type>> b,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * Apply the matrix to a multi-vector with a linear combination of the given
     * input vector. Represents the matrix vector multiplication, x = alpha * A
     * * b + beta * x, where x and b are both multi-vectors.
     *
     * @param alpha  the scalar to scale the matrix-vector product with
     * @param b      the multi-vector to be applied to
     * @param beta   the scalar to scale the x vector with
     * @param x      the output multi-vector
     */
    External* apply(ptr_param<const MultiVector<value_type>> alpha,
                    ptr_param<const MultiVector<value_type>> b,
                    ptr_param<const MultiVector<value_type>> beta,
                    ptr_param<MultiVector<value_type>> x);

    /**
     * @copydoc apply(const MultiVector<value_type>*, MultiVector<value_type>*)
     */
    const External* apply(ptr_param<const MultiVector<value_type>> b,
                          ptr_param<MultiVector<value_type>> x) const;

    /**
     * @copydoc apply(const MultiVector<value_type>*, const
     * MultiVector<value_type>*, const MultiVector<value_type>*,
     * MultiVector<value_type>*)
     */
    const External* apply(ptr_param<const MultiVector<value_type>> alpha,
                          ptr_param<const MultiVector<value_type>> b,
                          ptr_param<const MultiVector<value_type>> beta,
                          ptr_param<MultiVector<value_type>> x) const;

    functor_operation<external_apply::simple_type> get_simple_apply_functions()
        const
    {
        return simple_apply_;
    }

    functor_operation<external_apply::advanced_type>
    get_advanced_apply_functions() const
    {
        return advanced_apply_;
    }

    void* get_payload() const { return payload_; }

private:
    External(std::shared_ptr<const Executor> exec);

    External(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             functor_operation<external_apply::simple_type> simple_apply,
             functor_operation<external_apply::advanced_type> advanced_apply,
             void* payload);

    void apply_impl(const MultiVector<value_type>* b,
                    MultiVector<value_type>* x) const;

    void apply_impl(const MultiVector<value_type>* alpha,
                    const MultiVector<value_type>* b,
                    const MultiVector<value_type>* beta,
                    MultiVector<value_type>* x) const;

    functor_operation<external_apply::simple_type> simple_apply_;
    functor_operation<external_apply::advanced_type> advanced_apply_;
    void* payload_;
};


}  // namespace matrix
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_EXTERNAL_HPP_
