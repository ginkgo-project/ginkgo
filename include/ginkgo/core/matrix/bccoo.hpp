/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType, typename IndexType>
class Coo;


template <typename ValueType, typename IndexType>
class Csr;


template <typename ValueType>
class Dense;

/* // JIAE
template <typename ValueType, typename IndexType>
class BccooBuilder;
*/

/**
 * BCCOO is a matrix format which only stores the nonzeros coeffficients
 * by blocks of consecutive elements.
 *
 * First the elements are sorted by row and column indexes, and, then,
 * only the pairs (column,value) is stored in the 1D array of bytes.
 * The column indexes could be stored as their value or as the difference
 * from the previous element in the same row.
 *
 * Two additional 1-D vectors complete the block structure. One of them
 * contains the starting point of each block in the array of bytes,
 * whereas the second one indicates the row index of the first pair
 * in the block.
 *
 * The BCCOO LinOp supports different operations:
 *
 * ```cpp
 * matrix::BCCOO *A, *B, *C;    // matrices
 * matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
 * matrix::Dense *alpha, *beta; // scalars of dimension 1x1
 * matrix::Identity *I;         // identity matrix
 *
 * // Applying to Dense matrices computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 *
 * // Applying to BCCOO matrices computes a SpGEMM product of two sparse
 * matrices A->apply(B, C)              // C = A*B A->apply(alpha, B, beta, C)
 * // C = alpha*A*B + beta*C
 *
 * // Applying to an Identity matrix computes a SpGEAM sparse matrix addition
 * A->apply(alpha, I, beta, B) // B = alpha*A + beta*B
 * ```
 * Both the SpGEMM and SpGEAM operation require the input matrices to be sorted
 * by column index, otherwise the algorithms will produce incorrect results.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup bccoo
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Bccoo : public EnableLinOp<Bccoo<ValueType, IndexType>>,
              public EnableCreateMethod<Bccoo<ValueType, IndexType>>,
              public ConvertibleTo<Bccoo<next_precision<ValueType>, IndexType>>,
              public ConvertibleTo<Coo<ValueType, IndexType>>,
              public ConvertibleTo<Csr<ValueType, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, IndexType>,
              public WritableToMatrixData<ValueType, IndexType>,
              public EnableAbsoluteComputation<
                  remove_complex<Bccoo<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Bccoo>;
    friend class EnablePolymorphicObject<Bccoo, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Csr<ValueType, IndexType>;
    friend class Dense<ValueType>;
    //    friend class BccooBuilder<ValueType, IndexType>; // JIAE
    friend class Bccoo<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Bccoo>::convert_to;
    using EnableLinOp<Bccoo>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Bccoo>;

    friend class Bccoo<next_precision<ValueType>, IndexType>;

    void convert_to(
        Bccoo<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Bccoo<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(Coo<ValueType, IndexType>* other) const override;

    void move_to(Coo<ValueType, IndexType>* other) override;

    void convert_to(Csr<ValueType, IndexType>* other) const override;

    void move_to(Csr<ValueType, IndexType>* other) override;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void read(const mat_data& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns the row index of the first element of each block.
     *
     * @return the row index of the first element of each block.
     */
    index_type* get_rows() noexcept { return rows_.get_data(); }

    /**
     * @copydoc Bccoo::get_rows()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_rows() const noexcept
    {
        return rows_.get_const_data();
    }

    /**
     * Returns the offset related to the first element of each block.
     *
     * @return the offset related to the first element of each block.
     */
    index_type* get_offsets() noexcept { return offsets_.get_data(); }

    /**
     * @copydoc Bccoo::get_offsets()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_offsets() const noexcept
    {
        return offsets_.get_const_data();
    }

    /**
     * Returns the vector where column indexes and values are stored.
     *
     * @return the vector where column indexes and values are stored.
     */
    uint8* get_data() noexcept { return data_.get_data(); }

    /**
     * @copydoc Bccoo::get_data()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const uint8* get_const_data() const noexcept
    {
        return data_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept { return num_nonzeros_; }

    /**
     * Returns the number of blocks used in the definition of the matrix.
     *
     * @return the number of blocks used in the definition of the matrix.
     */
    size_type get_num_blocks() const noexcept { return rows_.get_num_elems(); }

    /**
     * Returns the number of blocks used in the definition of the matrix.
     *
     * @return the number of blocks used in the definition of the matrix.
     */
    size_type get_num_bytes() const noexcept { return data_.get_num_elems(); }

    /**
     * Applies Bccoo matrix axpy to a vector (or a sequence of vectors).
     *
     * Performs the operation x = Bccoo * b + x
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp* apply2(const LinOp* b, LinOp* x)
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply2(cost LinOp *, LinOp *)
     */
    const LinOp* apply2(const LinOp* b, LinOp* x) const
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Performs the operation x = alpha * Bccoo * b + x.
     *
     * @param alpha  scaling of the result of Bccoo * b
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp* apply2(const LinOp* alpha, const LinOp* b, LinOp* x)
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply2(const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp* apply2(const LinOp* alpha, const LinOp* b, LinOp* x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get(),
                          make_temporary_clone(exec, x).get());
        return this;
    }

protected:
    /**
     * Creates an uninitialized BCCOO matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param num_blocks    number of blocks
     * @param num_bytes     number of bytes
     */
    Bccoo(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
          size_type num_nonzeros = {}, size_type num_blocks = {},
          size_type num_bytes = {})
        : EnableLinOp<Bccoo>(exec, size),
          rows_(exec, num_blocks),
          offsets_(exec, num_blocks + 1),
          data_(exec, num_bytes)
    {}

    /**
     * Creates a BCCOO matrix from already allocated (and initialized) row
     * index, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowIdxArray  type of `row_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_idxs  array of row pointers
     *
     * @note If one of `row_idxs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    /*
    // JIAE
        template <typename ValuesArray, typename ColIdxsArray,
                  typename RowIdxsArray>
        Bccoo(std::shared_ptr<const Executor> exec, const dim<2> &size,
              ValuesArray &&values, ColIdxsArray &&col_idxs,
              RowIdxsArray &&row_idxs)
            : EnableLinOp<Bccoo>(exec, size),
              values_{exec, std::forward<ValuesArray>(values)},
              col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
              row_idxs_{exec, std::forward<RowIdxsArray>(row_idxs)}
        {
            GKO_ASSERT_EQ(values_.get_num_elems(), col_idxs_.get_num_elems());
            GKO_ASSERT_EQ(values_.get_num_elems(), row_idxs_.get_num_elems());
        }
    */
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply2_impl(const LinOp* b, LinOp* x) const;

    void apply2_impl(const LinOp* alpha, const LinOp* b, LinOp* x) const;

private:
    // array<value_type> values_;
    // array<index_type> col_idxs_;
    // array<index_type> row_idxs_;

    array<index_type> rows_;
    array<index_type> offsets_;
    array<uint8> data_;
    size_type num_nonzeros_;
    // size_type num_blocks_;
    // size_type num_bytes_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BCCOO_HPP_
