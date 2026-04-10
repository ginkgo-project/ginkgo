// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SKETCH_GAUSSIAN_SKETCH_HPP_
#define GKO_PUBLIC_CORE_SKETCH_GAUSSIAN_SKETCH_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/sketch/sketch_operator.hpp>


namespace gko {
namespace sketch {


/**
 * GaussianSketch is a sketch operator that uses a dense random matrix
 * with i.i.d. N(0, 1/sqrt(k)) entries.
 *
 * Stores both the (k x m) sketch matrix and its (m x k) transpose
 * for efficient left-sketch and right-sketch operations that delegate
 * to Dense GEMM.
 *
 * @tparam ValueType  precision of matrix elements
 */
template <typename ValueType = default_precision>
class GaussianSketch
    : public EnableLinOp<GaussianSketch<ValueType>,
                         SketchOperator<ValueType>> {
    friend class EnableLinOp<GaussianSketch>;
    friend class EnablePolymorphicObject<GaussianSketch,
                                        SketchOperator<ValueType>>;

public:
    using value_type = ValueType;

    /**
     * Creates a GaussianSketch operator.
     *
     * @param exec  executor where the sketch lives
     * @param sketch_size  sketch dimension k (number of output rows)
     * @param input_size  input dimension m (number of input rows)
     * @param seed  random seed for reproducibility
     */
    static std::unique_ptr<GaussianSketch> create(
        std::shared_ptr<const Executor> exec, size_type sketch_size,
        size_type input_size, uint64 seed);

    /** Returns the random seed used to generate this sketch. */
    uint64 get_seed() const { return seed_; }

    /** Returns a const pointer to the sketch matrix (k x m). */
    const matrix::Dense<ValueType>* get_sketch_matrix() const
    {
        return sketch_matrix_.get();
    }

protected:
    void apply_sketch_impl(const matrix::Dense<ValueType>* b,
                           matrix::Dense<ValueType>* x) const override;

    void rapply_sketch_impl(const matrix::Dense<ValueType>* b,
                            matrix::Dense<ValueType>* x) const override;

    explicit GaussianSketch(std::shared_ptr<const Executor> exec)
        : EnableLinOp<GaussianSketch, SketchOperator<ValueType>>(exec),
          seed_{0}
    {}

    GaussianSketch(std::shared_ptr<const Executor> exec,
                   size_type sketch_size, size_type input_size, uint64 seed);

private:
    std::shared_ptr<matrix::Dense<ValueType>> sketch_matrix_;
    std::shared_ptr<matrix::Dense<ValueType>> sketch_matrix_t_;
    uint64 seed_;
};


}  // namespace sketch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SKETCH_GAUSSIAN_SKETCH_HPP_
