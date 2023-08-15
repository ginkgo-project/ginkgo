// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
