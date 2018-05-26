/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_MATRIX_HYB_HPP_
#define GKO_CORE_MATRIX_HYB_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"
#include "core/matrix/coo.hpp"
#include "core/matrix/ell.hpp"

namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

// template <typename ValueType, typename IndexType>
// class Coo;

/**
 * HYB is a matrix format where stride with explicit zeros is used such that
 * all rows have the same number of stored elements. The number of elements
 * stored in each row is the largest number of nonzero elements in any of the
 * rows (obtainable through get_max_nonzeros_per_row() method). This removes
 * the need of a row pointer like in the CSR format, and allows for SIMD
 * processing of the distinct rows. For efficient processing, the nonzero
 * elements and the corresponding column indices are stored in column-major
 * fashion. The columns are padded to the length by user-defined stride
 * parameter whose default value is the number of rows of the matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Hyb : public EnableLinOp<Hyb<ValueType, IndexType>>,
            public EnableCreateMethod<Hyb<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableCreateMethod<Hyb>;
    friend class EnablePolymorphicObject<Hyb, LinOp>;
    friend class Dense<ValueType>;

public:
    using EnableLinOp<Hyb>::convert_to;
    using EnableLinOp<Hyb>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using coo_type = Coo<ValueType, IndexType>;
    using ell_type = Ell<ValueType, IndexType>;
    
    /**
     * This type is used to describe how to get the partition.
     */
    enum partition {
        /**
         * Marks that partition is decided automatically.
         */
        automatically,

        /**
         * Marks that partition is decided by user with percentile.
         */
        percentile,

        /**
         * Marks that partition is decided by user with columns.
         */
        columns
    };

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    /**
     * Returns the values of the Ell part.
     *
     * @return the values of the Ell part.
     */
    value_type *get_ell_values() noexcept { return ell_->get_values(); }

    /**
     * @copydoc Hyb::get_ell_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_ell_values() const noexcept
    {
        return ell_->get_const_values();
    }

    /**
     * Returns the column indexes of the ELL part.
     *
     * @return the column indexes of the ELL part.
     */
    index_type *get_ell_col_idxs() noexcept { return ell_->get_col_idxs(); }

    /**
     * @copydoc Hyb::get_ell_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_ell_col_idxs() const noexcept
    {
        return ell_->get_const_col_idxs();
    }

    /**
     * Returns the maximum number of non-zeros per row of ell part.
     *
     * @return the maximum number of non-zeros per row of ell part.
     */
    size_type get_ell_max_nonzeros_per_row() const noexcept
    {
        return ell_->get_max_nonzeros_per_row();
    }

    /**
     * Returns the stride of the ell part.
     *
     * @return the stride of the ell part.
     */
    size_type get_ell_stride() const noexcept { return ell_->get_stride(); }

    /**
     * Returns the number of elements explicitly stored in the ell part.
     *
     * @return the number of elements explicitly stored in the ell part
     */
    size_type get_ell_num_stored_elements() const noexcept
    {
        return ell_->get_num_stored_elements();
    }

    /**
     * Returns the `idx`-th non-zero element of the `row`-th row in the ell
     * part.
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    value_type &ell_val_at(size_type row, size_type idx) noexcept
    {
        return ell_->val_at(row, idx);
    }

    /**
     * @copydoc Hyb::ell_val_at(size_type, size_type)
     */
    value_type ell_val_at(size_type row, size_type idx) const noexcept
    {
        return ell_->val_at(row, idx);
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row in the ell part.
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    index_type &ell_col_at(size_type row, size_type idx) noexcept
    {
        return ell_->col_at(row, idx);
    }

    /**
     * @copydoc Hyb::ell_col_at(size_type, size_type)
     */
    index_type ell_col_at(size_type row, size_type idx) const noexcept
    {
        return ell_->col_at(row, idx);
    }

    /**
     * Returns the values of the coo part.
     *
     * @return the values of the coo part.
     */
    value_type *get_coo_values() noexcept { return coo_->get_values(); }

    /**
     * @copydoc Hyb::get_coo_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_coo_values() const noexcept
    {
        return coo_->get_const_values();
    }

    /**
     * Returns the column indexes of the ell part.
     *
     * @return the column indexes of the ell part.
     */
    index_type *get_coo_col_idxs() noexcept { return coo_->get_col_idxs(); }

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_coo_col_idxs() const noexcept
    {
        return coo_->get_const_col_idxs();
    }

    /**
     * Returns the row indexes of the matrix.
     *
     * @return the row indexes of the matrix.
     */
    index_type *get_coo_row_idxs() noexcept { return coo_->get_row_idxs(); }

    /**
     * @copydoc Csr::get_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_coo_row_idxs() const noexcept
    {
        return coo_->get_const_row_idxs();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_coo_num_stored_elements() const noexcept
    {
        return coo_->get_num_stored_elements();
    }

protected:
    /**
     * Creates an uninitialized Hyb matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.
     *     The max_nonzeros_per_row is set to the number of cols of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Hyb(std::shared_ptr<const Executor> exec, const dim &size = dim{})
        : Hyb(std::move(exec), size, size.num_cols)
    {}

    /**
     * Creates an uninitialized Hyb matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param max_nonzeros_per_row   maximum number of nonzeros in one row
     */
    Hyb(std::shared_ptr<const Executor> exec, const dim &size,
        size_type max_nonzeros_per_row)
        : Hyb(std::move(exec), size, max_nonzeros_per_row, size.num_rows)
    {}

    /**
     * Creates an uninitialized Hyb matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param max_nonzeros_per_row   maximum number of nonzeros in one row
     * @param stride                stride of the rows
     * @param num_nonzeros  number of nonzeros
     */
    Hyb(std::shared_ptr<const Executor> exec, const dim &size,
        size_type max_nonzeros_per_row, size_type stride,
        size_type num_nonzeros = {})
        : EnableLinOp<Hyb>(exec, size),
          ell_(std::move(ell_type::create(exec, size, max_nonzeros_per_row, stride))),
          coo_(std::move(coo_type::create(exec, size, num_nonzeros)))
    {
        
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;


private:
    std::shared_ptr< ell_type > ell_;
    std::shared_ptr< coo_type > coo_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_HYB_HPP_
