/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_COO_BUILDER_HPP_
#define GKO_CORE_MATRIX_COO_BUILDER_HPP_


#include <ginkgo/core/matrix/coo.hpp>


namespace gko {
namespace matrix {


/**
 * @internal
 *
 * Allows intrusive access to the arrays stored within a @ref Coo matrix.
 *
 * @tparam ValueType  the value type of the matrix
 * @tparam IndexType  the index type of the matrix
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class CooBuilder {
public:
    /**
     * Returns the row index array of the COO matrix.
     */
    Array<IndexType> &get_row_idx_array() { return matrix_->row_idxs_; }

    /**
     * Returns the column index array of the COO matrix.
     */
    Array<IndexType> &get_col_idx_array() { return matrix_->col_idxs_; }

    /**
     * Returns the value array of the COO matrix.
     */
    Array<ValueType> &get_value_array() { return matrix_->values_; }

    /**
     * Initializes a CooBuilder from an existing COO matrix.
     */
    explicit CooBuilder(Coo<ValueType, IndexType> *matrix) : matrix_{matrix} {}

    // make this type non-movable
    CooBuilder(const CooBuilder &) = delete;
    CooBuilder(CooBuilder &&) = delete;
    CooBuilder &operator=(const CooBuilder &) = delete;
    CooBuilder &operator=(CooBuilder &&) = delete;

private:
    Coo<ValueType, IndexType> *matrix_;
};


}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_COO_BUILDER_HPP_