// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SKETCH_SPARSE_STACK_HPP_
#define GKO_PUBLIC_CORE_SKETCH_SPARSE_STACK_HPP_


#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/sketch/sketch_operator.hpp>


namespace gko {
namespace sketch {


/**
 * SparseStack is a matrix-free sketch operator.
 *
 * Each input row i is mapped to zeta output rows hash_map[i * zeta + z]
 * with signs signs[i * zeta + z] for z in [0, zeta).
 * Only O(zeta * m) storage is needed (no explicit matrix).
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  type for hash_map indices
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class SparseStack : public EnableLinOp<SparseStack<ValueType, IndexType>,
                                       SketchOperator<ValueType>> {
    friend class EnableLinOp<SparseStack>;
    friend class EnablePolymorphicObject<SparseStack,
                                         SketchOperator<ValueType>>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Creates a SparseStack operator.
     *
     * @param exec  executor where the sketch lives
     * @param sketch_size  sketch dimension (number of output rows)
     * @param input_size  input dimension (number of input rows)
     * @param zeta  number of non-zeros per column
     * @param seed  random seed for reproducibility
     */
    static std::unique_ptr<SparseStack> create(
        std::shared_ptr<const Executor> exec, size_type sketch_size,
        size_type input_size, size_type zeta, uint64 seed);
    
    /**
     * Creates an uninitialized SparseStack operator.
     *
     * @param exec  executor where the sketch lives
     */
    static std::unique_ptr<SparseStack> create(std::shared_ptr<const Executor> exec);

    /** Returns the random seed. */
    uint64 get_seed() const { return seed_; }

    /** Returns the number of non-zeros per column. */
    size_type get_zeta() const { return zeta_; }

    /** Returns the hash map array (zeta * m entries, each in [0, k)). */
    const array<index_type>& get_hash_map() const { return hash_map_; }

    /** Returns the signs array (zeta * m entries, each +1 or -1). */
    const array<value_type>& get_signs() const { return signs_; }

protected:
    void apply_sketch_impl(const matrix::Dense<ValueType>* b,
                           matrix::Dense<ValueType>* x) const override;

    void rapply_sketch_impl(const matrix::Dense<ValueType>* b,
                            matrix::Dense<ValueType>* x) const override;

    explicit SparseStack(std::shared_ptr<const Executor> exec)
        : EnableLinOp<SparseStack, SketchOperator<ValueType>>(exec),
          hash_map_{exec},
          signs_{exec},
          zeta_{1},
          seed_{0}
    {}

    SparseStack(std::shared_ptr<const Executor> exec, size_type sketch_size,
                size_type input_size, size_type zeta, uint64 seed);

private:
    size_type zeta_;
    array<index_type> hash_map_;
    array<value_type> signs_;
    uint64 seed_;
};


}  // namespace sketch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SKETCH_SPARSE_STACK_HPP_
