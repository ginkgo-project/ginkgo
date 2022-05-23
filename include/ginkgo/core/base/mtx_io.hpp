/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_MTX_IO_HPP_
#define GKO_PUBLIC_CORE_BASE_MTX_IO_HPP_


#include <istream>


#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {


/**
 * Reads a matrix stored in matrix market format from an input stream.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param is  input stream from which to read the data
 *
 * @return A matrix_data structure containing the matrix. The nonzero elements
 *         are sorted in lexicographic order of their (row, column) indexes.
 *
 * @note This is an advanced routine that will return the raw matrix data
 *       structure. Consider using gko::read instead.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
matrix_data<ValueType, IndexType> read_raw(std::istream& is);


/**
 * Reads a matrix stored in Ginkgo's binary matrix format from an input stream.
 * Note that this format depends on the processor's endianness,
 * so files from a big endian processor can't be read from a little endian
 * processor and vice-versa.
 *
 * The binary format has the following structure (in system endianness):
 * 1. A 32 byte header consisting of 4 uint64_t values:
 *    magic = GINKGO__: The highest two bytes stand for value and index type.
 *                      value type: S (float), D (double),
 *                                  C (complex<float>), Z(complex<double>)
 *                      index type: I (int32), L (int64)
 *    num_rows: Number of rows
 *    num_cols: Number of columns
 *    num_entries: Number of (row, column, value) tuples to follow
 * 2. Following are num_entries blocks of size
 *    sizeof(IndexType) * 2 + sizeof(ValueType).
 *    Each consists of a row index stored as IndexType, followed by
 *    a column index stored as IndexType and a value stored as ValueType.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param is  input stream from which to read the data
 *
 * @return A matrix_data structure containing the matrix. The nonzero elements
 *         are sorted in lexicographic order of their (row, column) indexes.
 *
 * @note This is an advanced routine that will return the raw matrix data
 *       structure. Consider using gko::read_binary instead.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
matrix_data<ValueType, IndexType> read_binary_raw(std::istream& is);


/**
 * Reads a matrix stored in either binary or matrix market format from an input
 * stream.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param is  input stream from which to read the data
 *
 * @return A matrix_data structure containing the matrix. The nonzero elements
 *         are sorted in lexicographic order of their (row, column) indexes.
 *
 * @note This is an advanced routine that will return the raw matrix data
 *       structure. Consider using gko::read_generic instead.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
matrix_data<ValueType, IndexType> read_generic_raw(std::istream& is);


/**
 * Specifies the layout type when writing data in matrix market format.
 */
enum class layout_type {
    /**
     * The matrix should be written as dense matrix in column-major order.
     */
    array,
    /**
     * The matrix should be written as a sparse matrix in coordinate format.
     */
    coordinate
};


/**
 * Writes a matrix_data structure to a stream in matrix market format.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param os  output stream where the data is to be written
 * @param data  the matrix data to write
 * @param layout  the layout used in the output
 *
 * @note This is an advanced routine that writes the raw matrix data structure.
 *       If you are trying to write an existing matrix, consider using
 *       gko::write instead.
 */
template <typename ValueType, typename IndexType>
void write_raw(std::ostream& os, const matrix_data<ValueType, IndexType>& data,
               layout_type layout = layout_type::array);


/**
 * Writes a matrix_data structure to a stream in binary format.
 * Note that this format depends on the processor's endianness,
 * so files from a big endian processor can't be read from a little endian
 * processor and vice-versa.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param os  output stream where the data is to be written
 * @param data  the matrix data to write
 *
 * @note This is an advanced routine that writes the raw matrix data structure.
 *       If you are trying to write an existing matrix, consider using
 *       gko::write_binary instead.
 */
template <typename ValueType, typename IndexType>
void write_binary_raw(std::ostream& os,
                      const matrix_data<ValueType, IndexType>& data);


/**
 * Reads a matrix stored in matrix market format from an input stream.
 *
 * @tparam MatrixType  a ReadableFromMatrixData LinOp type used to store the
 *                     matrix once it's been read from disk.
 * @tparam StreamType  type of stream used to write the data to
 * @tparam MatrixArgs  additional argument types passed to MatrixType
 *                     constructor
 *
 * @param is  input stream from which to read the data
 * @param args  additional arguments passed to MatrixType constructor
 *
 * @return A MatrixType LinOp filled with data from filename
 */
template <typename MatrixType, typename StreamType, typename... MatrixArgs>
inline std::unique_ptr<MatrixType> read(StreamType&& is, MatrixArgs&&... args)
{
    auto mtx = MatrixType::create(std::forward<MatrixArgs>(args)...);
    mtx->read(read_raw<typename MatrixType::value_type,
                       typename MatrixType::index_type>(is));
    return mtx;
}


/**
 * Reads a matrix stored in binary format from an input stream.
 *
 * @tparam MatrixType  a ReadableFromMatrixData LinOp type used to store the
 *                     matrix once it's been read from disk.
 * @tparam StreamType  type of stream used to write the data to
 * @tparam MatrixArgs  additional argument types passed to MatrixType
 *                     constructor
 *
 * @param is  input stream from which to read the data
 * @param args  additional arguments passed to MatrixType constructor
 *
 * @return A MatrixType LinOp filled with data from filename
 */
template <typename MatrixType, typename StreamType, typename... MatrixArgs>
inline std::unique_ptr<MatrixType> read_binary(StreamType&& is,
                                               MatrixArgs&&... args)
{
    auto mtx = MatrixType::create(std::forward<MatrixArgs>(args)...);
    mtx->read(read_binary_raw<typename MatrixType::value_type,
                              typename MatrixType::index_type>(is));
    return mtx;
}


/**
 * Reads a matrix stored either in binary or matrix market format from an input
 * stream.
 *
 * @tparam MatrixType  a ReadableFromMatrixData LinOp type used to store the
 *                     matrix once it's been read from disk.
 * @tparam StreamType  type of stream used to write the data to
 * @tparam MatrixArgs  additional argument types passed to MatrixType
 *                     constructor
 *
 * @param is  input stream from which to read the data
 * @param args  additional arguments passed to MatrixType constructor
 *
 * @return A MatrixType LinOp filled with data from filename
 */
template <typename MatrixType, typename StreamType, typename... MatrixArgs>
inline std::unique_ptr<MatrixType> read_generic(StreamType&& is,
                                                MatrixArgs&&... args)
{
    auto mtx = MatrixType::create(std::forward<MatrixArgs>(args)...);
    mtx->read(read_generic_raw<typename MatrixType::value_type,
                               typename MatrixType::index_type>(is));
    return mtx;
}


namespace matrix {


template <typename ValueType>
class Dense;


class Fft;


class Fft2;


class Fft3;


}  // namespace matrix


namespace detail {


/**
 * @internal
 *
 * Type traits to decide for a default gko::write output layout.
 * It defaults to (sparse) coordinate storage, and has specializations
 * for dense matrix types.
 *
 * @tparam MatrixType  the non cv-qualified matrix type.
 */
template <typename MatrixType>
struct mtx_io_traits {
    static constexpr auto default_layout = layout_type::coordinate;
};


template <typename ValueType>
struct mtx_io_traits<gko::matrix::Dense<ValueType>> {
    static constexpr auto default_layout = layout_type::array;
};


template <>
struct mtx_io_traits<gko::matrix::Fft> {
    static constexpr auto default_layout = layout_type::array;
};


template <>
struct mtx_io_traits<gko::matrix::Fft2> {
    static constexpr auto default_layout = layout_type::array;
};


template <>
struct mtx_io_traits<gko::matrix::Fft3> {
    static constexpr auto default_layout = layout_type::array;
};


}  // namespace detail


/**
 * Writes a matrix into an output stream in matrix market format.
 *
 * @tparam MatrixType  a WritableToMatrixData object providing data to be
 *                     written.
 * @tparam StreamType  type of stream used to write the data to
 *
 * @param os  output stream where the data is to be written
 * @param matrix  the matrix to write
 * @param layout  the layout used in the output
 */
template <typename MatrixType, typename StreamType>
inline void write(
    StreamType&& os, MatrixType* matrix,
    layout_type layout =
        detail::mtx_io_traits<std::remove_const_t<MatrixType>>::default_layout)
{
    matrix_data<typename MatrixType::value_type,
                typename MatrixType::index_type>
        data{};
    matrix->write(data);
    write_raw(os, data, layout);
}


/**
 * Writes a matrix into an output stream in binary format.
 * Note that this format depends on the processor's endianness,
 * so files from a big endian processor can't be read from a little endian
 * processor and vice-versa.
 *
 * @tparam MatrixType  a WritableToMatrixData object providing data to be
 *                     written.
 * @tparam StreamType  type of stream used to write the data to
 *
 * @param os  output stream where the data is to be written
 * @param matrix  the matrix to write
 */
template <typename MatrixType, typename StreamType>
inline void write_binary(StreamType&& os, MatrixType* matrix)
{
    matrix_data<typename MatrixType::value_type,
                typename MatrixType::index_type>
        data{};
    matrix->write(data);
    write_binary_raw(os, data);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MTX_IO_HPP_
