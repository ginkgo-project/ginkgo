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

#ifndef GKO_CORE_MATRIX_SELLP_HPP_
#define GKO_CORE_MATRIX_SELLP_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"


namespace gko {
namespace matrix {


constexpr int default_slice_size = 64;
constexpr int default_stride_factor = 1;


template <typename ValueType>
class Dense;

/**
 * SELL-P is a matrix format similar to ELL format. The difference is that
 * SELL-P format divides rows into smaller slices and store each slice with ELL
 * format.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Sellp : public EnableLinOp<Sellp<ValueType, IndexType>>,
              public EnableCreateMethod<Sellp<ValueType, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public ReadableFromMatrixData<ValueType, IndexType>,
              public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableCreateMethod<Sellp>;
    friend class EnablePolymorphicObject<Sellp, LinOp>;
    friend class Dense<ValueType>;

public:
    using EnableLinOp<Sellp>::convert_to;
    using EnableLinOp<Sellp>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Sellp::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Sellp::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the lengths(columns) of slices.
     *
     * @return the lengths(columns) of slices.
     */
    size_type *get_slice_lengths() noexcept
    {
        return slice_lengths_.get_data();
    }

    /**
     * @copydoc Sellp::get_slice_lengths()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type *get_const_slice_lengths() const noexcept
    {
        return slice_lengths_.get_const_data();
    }

    /**
     * Returns the offsets of slices.
     *
     * @return the offsets of slices.
     */
    size_type *get_slice_sets() noexcept { return slice_sets_.get_data(); }

    /**
     * @copydoc Sellp::get_slice_sets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type *get_const_slice_sets() const noexcept
    {
        return slice_sets_.get_const_data();
    }

    /**
     * Returns the size of a slice.
     *
     * @return the size of a slice.
     */
    size_type get_slice_size() const noexcept { return slice_size_; }

    /**
     * Returns the stride factor(t) of SELL-P.
     *
     * @return the stride factor(t) of SELL-P.
     */
    size_type get_stride_factor() const noexcept { return stride_factor_; }

    /**
     * Returns the total column number.
     *
     * @return the total column number.
     */
    size_type get_total_cols() const noexcept { return total_cols_; }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Returns the `idx`-th non-zero element of the `row`-th row with
     * `slice_set` slice set.
     *
     * @param row  the row of the requested element in the slice
     * @param slice_set  the slice set of the slice
     * @param idx  the idx-th stored element of the row in the slice
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    value_type &val_at(size_type row, size_type slice_set,
                       size_type idx) noexcept
    {
        return values_.get_data()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * @copydoc Sellp::val_at(size_type, size_type, size_type)
     */
    value_type val_at(size_type row, size_type slice_set, size_type idx) const
        noexcept
    {
        return values_
            .get_const_data()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row with `slice_set`
     * slice set.
     *
     * @param row  the row of the requested element in the slice
     * @param slice_set  the slice set of the slice
     * @param idx  the idx-th stored element of the row in the slice
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    index_type &col_at(size_type row, size_type slice_set,
                       size_type idx) noexcept
    {
        return this->get_col_idxs()[this->linearize_index(row, slice_set, idx)];
    }

    /**
     * @copydoc Sellp::col_at(size_type, size_type, size_type)
     */
    index_type col_at(size_type row, size_type slice_set, size_type idx) const
        noexcept
    {
        return this
            ->get_const_col_idxs()[this->linearize_index(row, slice_set, idx)];
    }

protected:
    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *    (The total_cols is set to be the number of slice times the number
     *     of cols of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim &size = dim{})
        : Sellp(std::move(exec), size,
                ceildiv(size.num_rows, default_slice_size) * size.num_cols)
    {}

    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *    (The slice_size and stride_factor are set to the default values.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param total_cols   number of the sum of all cols in every slice.
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim &size,
          size_type total_cols)
        : Sellp(std::move(exec), size, default_slice_size,
                default_stride_factor, total_cols)
    {}

    /**
     * Creates an uninitialized Sellp matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param slice_size  number of rows in each slice
     * @param stride_factor  factor for the stride in each slice (strides
     *                        should be multiples of the stride_factor)
     * @param total_cols   number of the sum of all cols in every slice.
     */
    Sellp(std::shared_ptr<const Executor> exec, const dim &size,
          size_type slice_size, size_type stride_factor, size_type total_cols)
        : EnableLinOp<Sellp>(exec, size),
          values_(exec, slice_size * total_cols),
          col_idxs_(exec, slice_size * total_cols),
          slice_lengths_(exec, (size.num_rows == 0)
                                   ? 0
                                   : ceildiv(size.num_rows, slice_size)),
          slice_sets_(exec, (size.num_rows == 0)
                                ? 0
                                : ceildiv(size.num_rows, slice_size) + 1),
          slice_size_(slice_size),
          stride_factor_(stride_factor),
          total_cols_(total_cols)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    size_type linearize_index(size_type row, size_type slice_set,
                              size_type col) const noexcept
    {
        return (slice_set + col) * slice_size_ + row;
    }

private:
    Array<value_type> values_{0};
    Array<index_type> col_idxs_{0};
    Array<size_type> slice_lengths_{0};
    Array<size_type> slice_sets_{0};
    size_type slice_size_;
    size_type stride_factor_;
    size_type total_cols_{0};
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SELLP_HPP_
