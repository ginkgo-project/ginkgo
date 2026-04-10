// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SKETCH_SKETCH_OPERATOR_HPP_
#define GKO_PUBLIC_CORE_SKETCH_SKETCH_OPERATOR_HPP_


#include <memory>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace sketch {


/**
 * SketchOperator is the abstract base class for randomized sketching
 * operators.
 *
 * A sketch operator S has dimensions (k x m) and supports two operations:
 * - Left-sketch via apply(): x = S * b, where b is (m x n), x is (k x n)
 * - Right-sketch via rapply(): x = b * S^T, where b is (n x m), x is (n x k)
 *
 * @tparam ValueType  precision of matrix elements
 */
template <typename ValueType = default_precision>
class SketchOperator
    : public EnableAbstractPolymorphicObject<SketchOperator<ValueType>,
                                             LinOp> {

public:
    using value_type = ValueType;

    /**
     * Right-applies the sketch operator: x = b * S^T.
     *
     * @param b  input matrix of size (n x m)
     * @param x  output matrix of size (n x k)
     */
    LinOp* rapply(ptr_param<const LinOp> b, ptr_param<LinOp> x)
    {
        GKO_ASSERT_EQUAL_COLS(this, b);
        GKO_ASSERT_EQUAL_ROWS(b, x);
        GKO_ASSERT_REVERSE_CONFORMANT(this, x);
        this->rapply_impl(b.get(), x.get());
        return this;
    }

    /**
     * Right-applies the sketch with scaling: x = alpha * b * S^T + beta * x.
     */
    LinOp* rapply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                  ptr_param<const LinOp> beta, ptr_param<LinOp> x)
    {
        GKO_ASSERT_EQUAL_COLS(this, b);
        GKO_ASSERT_EQUAL_ROWS(b, x);
        GKO_ASSERT_REVERSE_CONFORMANT(this, x);
        this->rapply_impl(alpha.get(), b.get(), beta.get(), x.get());
        return this;
    }

    /** Returns the sketch dimension k. */
    size_type get_sketch_size() const { return this->get_size()[0]; }

    /** Returns the input dimension m. */
    size_type get_input_size() const { return this->get_size()[1]; }

protected:
    /**
     * Subclasses implement this: compute x = S * b.
     * Both b and x are Dense<ValueType> with correct dimensions.
     */
    virtual void apply_sketch_impl(
        const matrix::Dense<ValueType>* b,
        matrix::Dense<ValueType>* x) const = 0;

    /**
     * Subclasses implement this: compute x = b * S^T.
     * Both b and x are Dense<ValueType> with correct dimensions.
     */
    virtual void rapply_sketch_impl(
        const matrix::Dense<ValueType>* b,
        matrix::Dense<ValueType>* x) const = 0;

    SketchOperator(std::shared_ptr<const Executor> exec,
                   const dim<2>& size = {})
        : EnableAbstractPolymorphicObject<SketchOperator, LinOp>(exec, size)
    {}

private:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->apply_sketch_impl(dense_b, dense_x);
            },
            b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                auto exec = this->get_executor();
                // Lazy-allocate cache
                if (!cache_.intermediate ||
                    cache_.intermediate->get_size() != dense_x->get_size()) {
                    cache_.intermediate = matrix::Dense<ValueType>::create(
                        exec, dense_x->get_size());
                }
                this->apply_sketch_impl(dense_b, cache_.intermediate.get());
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, cache_.intermediate);
            },
            alpha, b, beta, x);
    }

    void rapply_impl(const LinOp* b, LinOp* x) const
    {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->rapply_sketch_impl(dense_b, dense_x);
            },
            b, x);
    }

    void rapply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                     LinOp* x) const
    {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                auto exec = this->get_executor();
                if (!cache_.intermediate ||
                    cache_.intermediate->get_size() != dense_x->get_size()) {
                    cache_.intermediate = matrix::Dense<ValueType>::create(
                        exec, dense_x->get_size());
                }
                this->rapply_sketch_impl(dense_b, cache_.intermediate.get());
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, cache_.intermediate);
            },
            alpha, b, beta, x);
    }

    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct&) {}
        cache_struct(cache_struct&&) noexcept {}
        cache_struct& operator=(const cache_struct&) { return *this; }
        cache_struct& operator=(cache_struct&&) noexcept { return *this; }
        std::unique_ptr<matrix::Dense<ValueType>> intermediate{};
    } cache_;
};


}  // namespace sketch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SKETCH_SKETCH_OPERATOR_HPP_
