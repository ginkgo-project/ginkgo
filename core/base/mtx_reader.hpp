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


#include "core/base/matrix_data.hpp"


#include <memory>
#include <string>


namespace gko {


// TODO: replace filenames with streams


/**
 * Reads a matrix stored in MTX (matrix market) file.
 *
 * @tparam ValueType  type of matrix values
 * @tparam IndexType  type of matrix indexes
 *
 * @param filename  filename from which to read the data
 *
 * @return A matrix_data structure containing the matrix. The nonzero elements
 *         are sorted in lexicographic order of their (row, colum) indexes.
 *
 * @note This is an advanced routine that will return the raw matrix data
 *       structure. Consider using gko::read instead.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
matrix_data<ValueType, IndexType> read_raw(const std::string &filename);


/**
 * Reads a matrix stored in MTX (matrix market) file.
 *
 * @tparam MatrixType  a ReadableFromMatrixData LinOp type used to store the
 *                     matrix once it's been read from disk.
 * @tparam MatrixArgs  additional argument types passed to MatrixType
 *                     constructor
 *
 * @param filename  filename from which to read the data
 * @param args  additional arguments passed to MatrixType constructor
 *
 * @return A MatrixType LinOp filled with data from filename
 */
template <typename MatrixType, typename... MatrixArgs>
inline std::unique_ptr<MatrixType> read(const std::string &filename,
                                        MatrixArgs &&... args)
{
    auto mtx = MatrixType::create(std::forward<MatrixArgs>(args)...);
    mtx->read(read_raw<typename MatrixType::value_type,
                       typename MatrixType::index_type>(filename));
    return mtx;
}


}  // namespace gko


#endif  // GKO_CORE_BASE_MTX_READER_HPP_
