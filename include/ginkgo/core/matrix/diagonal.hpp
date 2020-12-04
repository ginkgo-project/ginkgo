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

#ifndef GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Dense;


/**
 * This class is a utility which efficiently implements the diagonal matrix (a
 * linear operator which scales a vector row wise).
 *
 * Objects of the Diagonal class always represent a square matrix, and
 * require one array to store their values.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes of a CSR matrix the diagonal
 *                    is applied or converted to.
 *
 * @ingroup diagonal
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Diagonal
    : public EnableLinOp<Diagonal<ValueType>>,
      public EnableCreateMethod<Diagonal<ValueType>>,
      public ConvertibleTo<Csr<ValueType, int32>>,
      public ConvertibleTo<Csr<ValueType, int64>>,
      public Transposable,
      public WritableToMatrixData<ValueType, int32>,
      public WritableToMatrixData<ValueType, int64>,
      public ReadableFromMatrixData<ValueType, int32>,
      public ReadableFromMatrixData<ValueType, int64>,
      public EnableAbsoluteComputation<remove_complex<Diagonal<ValueType>>> {
    friend class EnablePolymorphicObject<Diagonal, LinOp>;
    friend class EnableCreateMethod<Diagonal>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Diagonal<to_complex<ValueType>>;

public:
    using EnableLinOp<Diagonal>::convert_to;
    using EnableLinOp<Diagonal>::move_to;

    using value_type = ValueType;
    using index_type = int64;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<Diagonal>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    void convert_to(Csr<ValueType, int32> *result) const override;

    void move_to(Csr<ValueType, int32> *result) override;

    void convert_to(Csr<ValueType, int64> *result) const override;

    void move_to(Csr<ValueType, int64> *result) override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
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
     * Applies the diagonal matrix from the right side to a matrix b,
     * which means scales the columns of b with the according diagonal entries.
     *
     * @param b  the input vector(s) on which the diagonal matrix is applied
     * @param x  the output vector(s) where the result is stored
     */
    void rapply(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_REVERSE_CONFORMANT(this, b);
        GKO_ASSERT_EQUAL_ROWS(b, x);
        GKO_ASSERT_EQUAL_COLS(this, x);

        this->rapply_impl(b, x);
    }

    void read(const mat_data &data) override;

    void read(const mat_data32 &data) override;

    void write(mat_data &data) const override;

    void write(mat_data32 &data) const override;


protected:
    /**
     * Creates an empty Diagonal matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Diagonal(std::shared_ptr<const Executor> exec)
        : Diagonal(std::move(exec), size_type{})
    {}

    /**
     * Creates an Diagonal matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Diagonal(std::shared_ptr<const Executor> exec, size_type size)
        : EnableLinOp<Diagonal>(exec, dim<2>{size}), values_(exec, size)
    {}

    /**
     * Creates a Diagonal matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Diagonal(std::shared_ptr<const Executor> exec, const size_type size,
             ValuesArray &&values)
        : EnableLinOp<Diagonal>(exec, dim<2>(size)),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        GKO_ENSURE_IN_BOUNDS(size - 1, values_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void rapply_impl(const LinOp *b, LinOp *x) const;


private:
    Array<value_type> values_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DIAGONAL_HPP_
