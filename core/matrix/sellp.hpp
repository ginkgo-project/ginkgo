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


constexpr int default_slice_size = 64;
constexpr int default_padding_factor = 1;


namespace gko {
namespace matrix {

template <typename ValueType>
class Dense;

/**
 * Sellp is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format).
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column of each
 * nonzero element.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Sellp : public EnableLinOp<Sellp<ValueType, IndexType>>,
              public EnableCreateMethod<Sellp<ValueType, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public ReadableFromMatrixData<ValueType, IndexType> {
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
    size_type *get_slice_lens() noexcept { return slice_lens_.get_data(); }

    /**
     * @copydoc Sellp::get_slice_lens()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const size_type *get_const_slice_lens() const noexcept
    {
        return slice_lens_.get_const_data();
    }

    /**
     * Returns the offsets of slices.
     *
     * @return the offsets of slices.
     */
    index_type *get_slice_sets() noexcept { return slice_sets_.get_data(); }

    /**
     * @copydoc Sellp::get_slice_sets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_slice_sets() const noexcept
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
     * Returns the padding factor(t) of SELL-P.
     *
     * @return the padding factor(t) of SELL-P.
     */
    size_type get_padding_factor() const noexcept { return padding_factor_; }

    /**
     * Returns the total column number.
     *
     * @return the total column number.
     */
    size_type get_max_total_cols() const noexcept { return max_total_cols_; }

protected:
    Sellp(std::shared_ptr<const Executor> exec, const dim &size = dim{})
        : Sellp(std::move(exec), size,
                ceildiv(size.num_rows, default_slice_size) * size.num_cols)
    {}

    Sellp(std::shared_ptr<const Executor> exec, const dim &size,
          size_type max_total_cols)
        : Sellp(std::move(exec), size, default_slice_size,
                default_padding_factor, max_total_cols)
    {}

    Sellp(std::shared_ptr<const Executor> exec, const dim &size,
          size_type slice_size, size_type padding_factor,
          size_type max_total_cols)
        : EnableLinOp<Sellp>(exec, size),
          values_(exec, slice_size * max_total_cols),
          col_idxs_(exec, slice_size * max_total_cols),
          slice_lens_(exec, (size.num_rows == 0)
                                ? 0
                                : ceildiv(size.num_rows, slice_size)),
          slice_sets_(exec, (size.num_rows == 0)
                                ? 0
                                : ceildiv(size.num_rows, slice_size)),
          slice_size_(slice_size),
          padding_factor_(padding_factor),
          max_total_cols_(max_total_cols)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    size_type linearize_index(size_type row, size_type slice_set,
                              size_type col) const noexcept
    {
        return (slice_set + col) * slice_size_ + row;
    }

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        size_type slice_num = slice_sets_.get_num_elems();
        size_type slice_set = 0;
        for (index_type i = 1; i < slice_num; i++) {
            if (col > slice_sets_.get_const_data()[i]) {
                slice_set = slice_sets_.get_const_data()[i];
            }
        }
        return linearize_index(row, slice_set, col);
    }

    size_type linearize_index(size_type idx) const noexcept
    {
        return linearize_index(idx % slice_size_, idx / slice_size_);
    }

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<size_type> slice_lens_;
    Array<index_type> slice_sets_;
    size_type slice_size_;
    size_type padding_factor_;
    size_type max_total_cols_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SELLP_HPP_