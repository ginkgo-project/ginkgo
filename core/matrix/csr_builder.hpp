// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_BUILDER_HPP_
#define GKO_CORE_MATRIX_CSR_BUILDER_HPP_


#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace matrix {


/**
 * @internal
 *
 * Allows intrusive access to the arrays stored within a @ref Csr matrix.
 *
 * @tparam ValueType  the value type of the matrix
 * @tparam IndexType  the index type of the matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class CsrBuilder {
public:
    /**
     * Returns the column index array of the CSR matrix.
     */
    array<IndexType>& get_col_idx_array() { return matrix_->col_idxs_; }

    /**
     * Returns the value array of the CSR matrix.
     */
    array<ValueType>& get_value_array() { return matrix_->values_; }

    /**
     * Initializes a CsrBuilder from an existing CSR matrix.
     */
    explicit CsrBuilder(ptr_param<Csr<ValueType, IndexType>> matrix)
        : matrix_{matrix.get()}
    {}

    /**
     * Updates the internal matrix data structures at destruction.
     */
    ~CsrBuilder() { matrix_->make_srow(); }

    // make this type non-movable
    CsrBuilder(const CsrBuilder&) = delete;
    CsrBuilder(CsrBuilder&&) = delete;
    CsrBuilder& operator=(const CsrBuilder&) = delete;
    CsrBuilder& operator=(CsrBuilder&&) = delete;

private:
    Csr<ValueType, IndexType>* matrix_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_CSR_BUILDER_HPP_
