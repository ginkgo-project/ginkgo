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

#ifndef GKO_CORE_MATRIX_SLICED_ELL_HPP_
#define GKO_CORE_MATRIX_SLICED_ELL_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"
#include "core/base/math.hpp"

constexpr int default_slice_size = 32;


namespace gko {
namespace matrix {

template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Sliced_ell;

/**
 * Sliced_ell is a matrix format which stores only the nonzero coefficients by
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
class Sliced_ell : public LinOp,
            public ConvertibleTo<Sliced_ell<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ReadableFromMtx {
    friend class gko::matrix::Dense<ValueType>;

public:
    using value_type = ValueType;

    using index_type = IndexType;

    /**
     * Creates an uninitialized Sliced_ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows      number of rows
     * @param num_cols      number of columns
     * @param num_nonzeros  number of nonzeros
     * 
     */
    static std::unique_ptr<Sliced_ell> create(std::shared_ptr<const Executor> exec,
                                       size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros)
    {
        return std::unique_ptr<Sliced_ell>(
            new Sliced_ell(exec, num_rows, num_cols, num_nonzeros));
    }

    /**
     * Creates an empty Sliced ELL matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    static std::unique_ptr<Sliced_ell> create(std::shared_ptr<const Executor> exec)
    {
        return create(exec, 0, 0, 0);
    }

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(Sliced_ell *other) const override;

    void move_to(Sliced_ell *other) override;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read_from_mtx(const std::string &filename) override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Sliced_ell::get_values()
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
     * @copydoc Sliced_ell::get_col_idxs()
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
    index_type *get_slice_lens() noexcept { return slice_lens_.get_data(); }

    /**
     * @copydoc Sliced_ell::get_slice_lens()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_slice_lens() const noexcept
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
     * @copydoc Sliced_ell::get_slice_sets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_slice_sets() const noexcept
    {
        return slice_sets_.get_const_data();
    }

protected:
    Sliced_ell(std::shared_ptr<const Executor> exec, size_type num_rows,
        size_type num_cols, size_type num_nonzeros)
        : LinOp(exec, num_rows, num_cols, num_nonzeros),
          values_(exec, ceildiv(num_rows, default_slice_size)*default_slice_size*num_cols),
          col_idxs_(exec, ceildiv(num_rows, default_slice_size)*default_slice_size*num_cols),
          slice_lens_(exec, ceildiv(num_rows, default_slice_size)),
          slice_sets_(exec, ceildiv(num_rows, default_slice_size))
    {}

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    // Array<index_type> row_ptrs_;
    Array<index_type> slice_lens_;
    Array<index_type> slice_sets_;

};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_ELL_HPP_
