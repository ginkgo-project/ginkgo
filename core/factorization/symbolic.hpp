/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/elimination_forest.hpp"


namespace gko {
namespace factorization {


/**
 * Computes the symbolic Cholesky factorization of the given matrix.
 *
 * @param mtx  the input matrix
 * @param symmetrize  output the pattern of L + L^T (true) or just L (false)?
 * @param factors  the output factor(s)
 * @param forest  the elimination forest of the input matrix
 */
template <typename ValueType, typename IndexType>
void symbolic_cholesky(
    const matrix::Csr<ValueType, IndexType>* mtx, bool symmetrize,
    std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors,
    std::unique_ptr<elimination_forest<IndexType>>& forest);

/**
 * Computes the symbolic LU factorization of the given matrix.
 *
 * The implementation is based on fill1 algorithm introduced in Rose and Tarjan,
 * "Algorithmic Aspects of Vertex Elimination on Directed Graphs," SIAM J. Appl.
 * Math. 1978. and its formulation in Gaihre et. al,
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs," arXiv 2021
 *
 * @param mtx  the input matrix
 * @param factors  the output factors stored in a combined pattern
 */
template <typename ValueType, typename IndexType>
void symbolic_lu(const matrix::Csr<ValueType, IndexType>* mtx,
                 std::unique_ptr<matrix::Csr<ValueType, IndexType>>& factors);


}  // namespace factorization
}  // namespace gko
