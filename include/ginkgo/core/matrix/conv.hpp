// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_FFT_HPP_
#define GKO_PUBLIC_CORE_MATRIX_FFT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


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


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_FFT_HPP_
