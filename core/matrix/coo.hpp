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

#ifndef GKO_CORE_MATRIX_COO_HPP_
#define GKO_CORE_MATRIX_COO_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;

/**
 * COO is a matrix format which stores only the nonzero coefficients by
 * compressing the entire 2D coefficient table (coordinate matrix format).
 *
 * The nonzero elements are stored in a 1D array row-wise. Additional row index
 * and column index arrays are used to identify both the row and the column of
 * each nonzero element.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Coo : public BasicLinOp<Coo<ValueType, IndexType>>,
            public ConvertibleTo<Csr<ValueType, IndexType>>,
            public ReadableFromMtx,
            public Transposable {
    friend class BasicLinOp<Coo>;

public:
    using BasicLinOp<Coo>::create;
    using BasicLinOp<Coo>::convert_to;
    using BasicLinOp<Coo>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    void clear() override;

    void convert_to(Csr<ValueType, IndexType> *other) const override;

    void move_to(Csr<ValueType, IndexType> *other) override;

    void read_from_mtx(const std::string &filename) override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Csr::get_values()
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
     * @copydoc Csr::get_col_idxs()
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
     * Returns the row indexes of the matrix.
     *
     * @return the row indexes of the matrix.
     */
    index_type *get_row_idxs() noexcept { return row_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_row_idxs() const noexcept
    {
        return row_idxs_.get_const_data();
    }

protected:
    /**
     * Creates an empty COO matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Coo(std::shared_ptr<const Executor> exec)
        : BasicLinOp<Coo>(exec, 0, 0, 0),
          values_(exec),
          col_idxs_(exec),
          row_idxs_(exec)
    {}

    /**
     * Creates an uninitialized COO matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows      number of rows
     * @param num_cols      number of columns
     * @param num_nonzeros  number of nonzeros
     */
    Coo(std::shared_ptr<const Executor> exec, size_type num_rows,
        size_type num_cols, size_type num_nonzeros)
        : BasicLinOp<Coo>(exec, num_rows, num_cols, num_nonzeros),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          row_idxs_(exec, num_nonzeros)
    {}

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<index_type> row_idxs_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COO_HPP_
