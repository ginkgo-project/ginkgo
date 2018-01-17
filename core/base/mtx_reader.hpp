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

#ifndef GKO_CORE_BASE_MTX_READER_HPP_
#define GKO_CORE_BASE_MTX_READER_HPP_


#include "core/base/array.hpp"


#include <string>
#include <tuple>
#include <vector>


namespace gko {


/**
 * This structure is used as an intermediate data type to store the matrix
 * read from a file in COO-like format.
 *
 * Note that the structure is not optimized for usual access patterns, can only
 * exist on the CPU, and thus should only be used for reading matrices from MTX
 * format.
 *
 * @tparam ValueType  type of matrix values stored in the structure
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <typename ValueType = default_precision, typename IndexType = int32>
struct MtxData {
    /**
     * Total number of rows of the matrix.
     */
    size_type num_rows;
    /**
     * Total number of columns of the matrix.
     */
    size_type num_cols;
    /**
     * A vector of tuples storing the non-zeros of the matrix.
     *
     * The first two elements of the tuple are the row index and the column
     * index of a matrix element, and its third element is the value at that
     * position.
     */
    std::vector<std::tuple<IndexType, IndexType, ValueType>> nonzeros;
};


/**
 * Reads a matrix stored in MTX (matrix market) file.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param filename  filename from which to read the data
 *
 * @return A structure containing the matrix. The nonzero elements are sorted
 *         in lexicographic order of their (row, colum) indexes.
 *
 * @note Prefer using ReadableFromMtx::read_from_mtx interface for existing
 *       Ginkgo types, and use this function only if you want to implement this
 *       interface for your own types.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
MtxData<ValueType, IndexType> read_raw_from_mtx(const std::string &filename);
// TODO: replace filenames with streams


/**
 * A LinOp implementing this interface can read its data from a file stored in
 * matrix market format.
 */
class ReadableFromMtx {
public:
    /**
     * Reads a matrix stored in MTX (matrix market) file.
     *
     * @param filename  filename from which to read the matrix
     */
    virtual void read_from_mtx(const std::string &filename) = 0;

    virtual ~ReadableFromMtx() = default;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MTX_READER_HPP_
