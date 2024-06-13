// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_COO_BUILDER_HPP_
#define GKO_CORE_MATRIX_COO_BUILDER_HPP_


#include <ginkgo/core/matrix/coo.hpp>


namespace gko {
namespace matrix {


/**
 * @internal
 *
 * Allows intrusive access to the arrays stored within a @ref Coo matrix.
 *
 * @tparam ValueType  the value type of the matrix
 * @tparam IndexType  the index type of the matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class CooBuilder {
public:
    /**
     * Returns the row index array of the COO matrix.
     */
    array<IndexType>& get_row_idx_array() { return matrix_->row_idxs_; }

    /**
     * Returns the column index array of the COO matrix.
     */
    array<IndexType>& get_col_idx_array() { return matrix_->col_idxs_; }

    /**
     * Returns the value array of the COO matrix.
     */
    array<ValueType>& get_value_array() { return matrix_->values_; }

    /**
     * Initializes a CooBuilder from an existing COO matrix.
     */
    explicit CooBuilder(ptr_param<Coo<ValueType, IndexType>> matrix)
        : matrix_{matrix.get()}
    {}

    // make this type non-movable
    CooBuilder(const CooBuilder&) = delete;
    CooBuilder(CooBuilder&&) = delete;
    CooBuilder& operator=(const CooBuilder&) = delete;
    CooBuilder& operator=(CooBuilder&&) = delete;

private:
    Coo<ValueType, IndexType>* matrix_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_COO_BUILDER_HPP_
