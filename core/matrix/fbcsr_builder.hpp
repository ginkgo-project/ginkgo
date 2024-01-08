// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_
#define GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_


#include <ginkgo/core/matrix/fbcsr.hpp>


namespace gko {
namespace matrix {


/**
 * @internal
 *
 * Allows intrusive access to the arrays stored within a @ref Fbcsr matrix.
 *
 * @tparam ValueType  the value type of the matrix
 * @tparam IndexType  the index type of the matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class FbcsrBuilder {
public:
    /**
     * @return The column index array of the matrix.
     */
    array<IndexType>& get_col_idx_array() { return matrix_->col_idxs_; }

    /**
     * @return The value array of the matrix.
     */
    array<ValueType>& get_value_array() { return matrix_->values_; }

    /**
     * @return The (uniform) block size
     */
    int get_block_size() const { return matrix_->bs_; }

    /**
     * @param matrix  An existing FBCSR matrix
     *                for which intrusive access is needed
     */
    explicit FbcsrBuilder(ptr_param<Fbcsr<ValueType, IndexType>> const matrix)
        : matrix_{matrix.get()}
    {}

    ~FbcsrBuilder() = default;

    // make this type non-movable
    FbcsrBuilder(const FbcsrBuilder&) = delete;
    FbcsrBuilder(FbcsrBuilder&&) = delete;
    FbcsrBuilder& operator=(const FbcsrBuilder&) = delete;
    FbcsrBuilder& operator=(FbcsrBuilder&&) = delete;

private:
    Fbcsr<ValueType, IndexType>* matrix_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_FBCSR_BUILDER_HPP_
