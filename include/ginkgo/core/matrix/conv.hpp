// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_CONV_HPP_
#define GKO_PUBLIC_CORE_MATRIX_CONV_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace matrix {


/**
 * This LinOp implements a 1D Convolution
 *
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType>
class Conv : public EnableLinOp<Conv<ValueType>> {
    friend class EnablePolymorphicObject<Conv, LinOp>;

public:
    using EnableLinOp<Conv>::convert_to;
    using EnableLinOp<Conv>::move_to;

    using value_type = ValueType;

    /**
     * Creates an empty Convolution kernel matrix.
     *
     * @param exec  Executor associated to the matrix
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Conv> create(std::shared_ptr<const Executor> exec);

    /**
     * Creates an Convolution kernel matrix.
     *
     * @param array  kernel used by convolution
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Conv> create(std::shared_ptr<const Executor> exec,
                                        const array<ValueType>& array);

protected:
    Conv(std::shared_ptr<const Executor> exec, const array<ValueType>& array);

    Conv(std::shared_ptr<const Executor> exec);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void validate_application_parameters(const LinOp* b,
                                         const LinOp* x) const override;

private:
    array<ValueType> kernel_;
};

/**
 * This LinOp implements a 2D Convolution
 *
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType>
class Conv2d : public EnableLinOp<Conv2d<ValueType>> {
    friend class EnablePolymorphicObject<Conv2d<ValueType>, LinOp>;

public:
    using EnableLinOp<Conv2d>::convert_to;
    using EnableLinOp<Conv2d>::move_to;
    using value_type = ValueType;
    /**
     * Creates an empty Convolution kernel matrix.
     *
     * @param exec  Executor associated to the matrix
     *
     * @return A smart pointer to the newly created matrix.
     */

    static std::unique_ptr<Conv2d> create(std::shared_ptr<const Executor> exec);
    /**
     * Creates an Convolution kernel matrix.
     *
     * @param array  kernel used by convolution
     *
     * @return A smart pointer to the newly created matrix.
     */

    static std::unique_ptr<Conv2d> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<const Dense<ValueType>> kernel);

protected:
    Conv2d(std::shared_ptr<const Executor> exec,
           std::shared_ptr<const Dense<ValueType>> kernel);
    Conv2d(std::shared_ptr<const Executor> exec);

    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void validate_application_parameters(const LinOp* b,
                                         const LinOp* x) const override;

private:
    std::shared_ptr<const Dense<ValueType>> kernel_;
};

/**
 * This LinOp implements a 2D Sparse Convolution
 *
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class Conv2dsparse : public EnableLinOp<Conv2dsparse<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Conv2dsparse, LinOp>;

public:
    using EnableLinOp<Conv2dsparse>::convert_to;
    using EnableLinOp<Conv2dsparse>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Creates an empty Convolution kernel matrix.
     *
     * @param exec  Executor associated to the matrix
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Conv2dsparse> create(
        std::shared_ptr<const Executor> exec);
    /**
     * Creates an Convolution kernel matrix.
     *
     * @param array  kernel used by convolution
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Conv2dsparse> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<const Csr<ValueType, IndexType>> kernel);

protected:
    Conv2dsparse(std::shared_ptr<const Executor> exec,
                 std::shared_ptr<const Csr<ValueType, IndexType>> kernel);
    Conv2dsparse(std::shared_ptr<const Executor> exec);
    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;
    void validate_application_parameters(const LinOp* b,
                                         const LinOp* x) const override;

private:
    std::shared_ptr<const Csr<ValueType, IndexType>> kernel_;
};

}  // namespace matrix
}  // namespace gko
#endif  // GKO_PUBLIC_CORE_MATRIX_FFT_HPP_
