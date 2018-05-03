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

#ifndef GKO_CORE_BASE_LIN_OP_INTERFACES_HPP_
#define GKO_CORE_BASE_LIN_OP_INTERFACES_HPP_


#include "core/base/matrix_data.hpp"
#include "core/base/types.hpp"


#include <memory>


namespace gko {


class LinOp;


/**
 * Linear operators which support transposition implement the Transposable
 * interface.
 *
 * It provides two functionalities, the normal transpose and the
 * conjugate transpose.
 *
 * The normal transpose returns the transpose of the linear operator without
 * changing any of its elements representing the operation, \f$B = A^{T}\f$.
 *
 * The conjugate transpose returns the conjugate of each of the elements and
 * additionally transposes the linear operator representing the operation, \f$B
 * = A^{H}\f$.
 *
 * Example: Transposing a Csr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Transposing an object of LinOp type.
 * //The object you want to transpose.
 * std::unique_ptr<LinOp> op = matrix::Csr::create(exec);
 * //Transpose the object by first converting it to a transposable type.
 * auto trans = as<Transposable>(op.get())->transpose();
 * //This returns the object of type LinOp, and needs to be cast to the
 * appropriate type for usage.
 * ```
 */
class Transposable {
public:
    virtual ~Transposable() = default;

    /**
     * Returns a LinOp representing the transpose of the Transposable object.
     *
     * @return A pointer to the new transposed object.
     */
    virtual std::unique_ptr<LinOp> transpose() const = 0;

    /**
     * Returns a LinOp representing the conjugate transpose of the Transposable
     * object.
     *
     * @return A pointer to the new conjugate transposed object.
     */
    virtual std::unique_ptr<LinOp> conj_transpose() const = 0;
};


/**
 * A LinOp implementing this interface can read its data from a matrix_data
 * structure.
 */
template <typename ValueType, typename IndexType>
class ReadableFromMatrixData {
public:
    virtual ~ReadableFromMatrixData() = default;

    /**
     * Reads a matrix from a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void read(const matrix_data<ValueType, IndexType> &data) = 0;
};


/**
 * A LinOp implementing this interface can write its data to a matrix_data
 * structure.
 */
template <typename ValueType, typename IndexType>
class WritableToMatrixData {
public:
    virtual ~WritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(matrix_data<ValueType, IndexType> &data) const = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_LIN_OP_INTERFACES_HPP_
