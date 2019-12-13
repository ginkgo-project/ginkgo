/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/solver/lower_trs_isai_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/lower_trs_isai.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The LOWER_TRS_ISAI solver namespace.
 *
 * @ingroup lower_trs_isai
 */
namespace lower_trs_isai {

   template<typename ValueType, typename IndexType>
   void isai_lower_col(
         gko::matrix::Csr<ValueType, IndexType> const* l,
         gko::matrix::Csr<ValueType, IndexType> *linvt,
         IndexType col) {

      // std::string context = "isai_lower_col";

      // L factor
      auto const l_row_ptrs = l->get_const_row_ptrs();
      auto const l_col_idxs = l->get_const_col_idxs();
      auto const l_vals = l->get_const_values();
 
      // M_L precond (transpose)
      auto const col_ptrs = linvt->get_const_row_ptrs();
      auto const row_idxs = linvt->get_const_col_idxs();
      auto vals = linvt->get_values();

      auto colptr_sa = col_ptrs[col];
      auto colptr_en = col_ptrs[col+1]-1;

      // Number of nnz in current column
      int m = col_ptrs[col+1]-col_ptrs[col];

      // Submatrix workspace used to compute column col of precond
      int ldsmat = m;
      gko::size_type smatsz = m*ldsmat;
      // smat is treated as column major
      std::vector<ValueType> smat(smatsz, 0.0); // Init with zeros

      for (auto colptr = colptr_sa; colptr <= colptr_en; ++colptr) {
      
         auto row = row_idxs[colptr];
      
         auto rowptr_sa = l_row_ptrs[row]; // Ptr to first column in current row
         auto rowptr_en = l_row_ptrs[row+1]-1; // Ptr to last column in current row

         // Row ptr to coeff in row of L
         auto rowptr = rowptr_sa;
         // Column ptr to coeff in column of M      
         auto ptr = colptr_sa;

         // Coefficient indexes in sybmatrix
         IndexType ii = colptr-colptr_sa; // Row index
         IndexType jj = 0; // Column index

         while ((rowptr <= rowptr_en) && (ptr <= colptr_en)) {

            auto j = l_col_idxs[rowptr];
            auto k = row_idxs[ptr];

            if (j == k) {
               // Col idx of M and row idx of L match: add element to
               // submatrix

               smat[ii + ldsmat*jj] = l_vals[rowptr];
               ++jj;
               ++ptr;
               ++rowptr;
            }
            else if (j > k) {
               // No matching element in L: add zero coefficient

               ++jj;
               ++ptr;
            }
            else {
               // j < k
               //
               // Current element in L does not match sparsity pattern of column
               // col in M
               ++rowptr;
            }

         }

      
      }

      // Solve triangular subsystem to compute current column of M
      // precond by solver L M_col = e_col

      // FIXME: Require linvt to be initialised with identity?

      // Set diagonal element to 1.0
      vals[colptr_sa] = static_cast<ValueType>(1.0);

      for (IndexType j = 1; j < m; ++j)
         vals[colptr_sa+j] = static_cast<ValueType>(0.0);
   
      // int nrhs = 1;
      // double alpha = 1.0;
      // char fside = 'L';
      // char fuplo = 'L';
      // char ftransa = 'N';
      // char fdiag = 'U';
   
      // dtrsm_(
      //       &fside, &fuplo, &ftransa, &fdiag,
      //       &m, &nrhs, &alpha,
      //       &smat[0], &ldsmat,
      //       &vals[colptr_sa], &m);   

      for (IndexType j = 0; j < m; ++j) {
         vals[colptr_sa+j] /= smat[j*(1+ldsmat)];      
         for (IndexType i = j+1; i < m; ++i) {
            vals[colptr_sa+i] -= vals[colptr_sa+j]*smat[i+j*ldsmat];
         }      
      }
   
   }
   
   template <typename ValueType, typename IndexType>
   void build_isai(
         std::shared_ptr<const ReferenceExecutor> exec,
         matrix::Csr<ValueType, IndexType> const* matrix,
         matrix::Csr<ValueType, IndexType> *isai)
   {

      auto ncol = matrix->get_size()[1];
      using Mtx = gko::matrix::Csr<ValueType, IndexType>;
      auto linvt_linop = matrix->transpose();
      auto linvt  = static_cast<Mtx *>(linvt_linop.get());
      
      // Compute lfactinv for each columns
      for (IndexType col = 0; col < ncol; ++col) {
         isai_lower_col(matrix, linvt, col);
      }

      auto linv_linop = linvt->transpose();
      auto linv = static_cast<Mtx *>(linv_linop.get());
      isai->copy_from(linv);      
   }
   
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_ISAI_BUILD_KERNEL);
   
/**
 * The parameters trans_x and trans_b are used only in the CUDA executor for
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm and hence
 * here essentially unused.
 */
// template <typename ValueType, typename IndexType>
// void solve(std::shared_ptr<const ReferenceExecutor> exec,
//            const matrix::Csr<ValueType, IndexType> *matrix,
//            const solver::SolveStruct *solve_struct,
//            matrix::Dense<ValueType> *trans_b, matrix::Dense<ValueType> *trans_x,
//            const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:lower_trs_isai): change the code imported from solver/lower_trs if needed
//    auto row_ptrs = matrix->get_const_row_ptrs();
//    auto col_idxs = matrix->get_const_col_idxs();
//    auto vals = matrix->get_const_values();
//
//    for (size_type j = 0; j < b->get_size()[1]; ++j) {
//        for (size_type row = 0; row < matrix->get_size()[0]; ++row) {
//            x->at(row, j) = b->at(row, j) / vals[row_ptrs[row + 1] - 1];
//            for (size_type k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
//                auto col = col_idxs[k];
//                if (col < row) {
//                    x->at(row, j) +=
//                        -vals[k] * x->at(col, j) / vals[row_ptrs[row + 1] - 1];
//                }
//            }
//        }
//    }
//}

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_LOWER_TRS_ISAI_SOLVE_KERNEL);


}  // namespace lower_trs_isai
}  // namespace reference
}  // namespace kernels
}  // namespace gko
