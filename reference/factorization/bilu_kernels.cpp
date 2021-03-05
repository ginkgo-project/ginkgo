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

#include "core/factorization/bilu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "reference/components/fixed_block_operations.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The block ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace bilu_factorization {


template <typename IndexType>
static inline ptrdiff_t get_diag_block_position(const IndexType blk_row,
                                                const IndexType *const browptr,
                                                const IndexType *const bcolidx)
{
    const IndexType *const didx = std::find(
        bcolidx + browptr[blk_row], bcolidx + browptr[blk_row + 1], blk_row);
    return didx - bcolidx;
}

template <int bs, typename ValueType, typename IndexType>
static void compute_bilu_inplace_impl(const int nbrows,
                                      const IndexType *const row_ptrs,
                                      const IndexType *const col_idxs,
                                      ValueType *const values)
{
    using Blk_t = blockutils::FixedBlock<ValueType, bs, bs>;
    const auto vals = reinterpret_cast<Blk_t *>(values);

    for (IndexType ibrow = 0; ibrow < nbrows; ibrow++) {
        for (IndexType jbz = row_ptrs[ibrow]; jbz < row_ptrs[ibrow + 1];
             jbz++) {
            const IndexType jcol = col_idxs[jbz];
            if (jcol < ibrow) {
                const auto jdpos =
                    get_diag_block_position(jcol, row_ptrs, col_idxs);
                Blk_t D = vals[jdpos];
                const bool invflag = invert_block_complete(D);
                if (!invflag) {
                    printf(" Could not invert diag block at blk-row %ld!",
                           static_cast<long int>(ibrow));
                }
                Blk_t factor;
                for (int i = 0; i < bs; i++) {
                    for (int j = 0; j < bs; j++) {
                        factor(i, j) = 0.0;
                        for (int k = 0; k < bs; k++) {
                            factor(i, j) += vals[jbz](i, k) * D(k, j);
                        }
                    }
                }

                // match A[i, j+1:] with A[j, j+1:]
                IndexType kbz = jbz + 1, lbz = jdpos + 1;
                while (kbz < row_ptrs[ibrow + 1] && lbz < row_ptrs[jcol + 1]) {
                    const IndexType kcol = col_idxs[kbz];
                    const IndexType lcol = col_idxs[lbz];
                    if (kcol == lcol) {
                        for (int i = 0; i < bs; i++) {
                            for (int j = 0; j < bs; j++) {
                                for (int k = 0; k < bs; k++) {
                                    vals[kbz](i, j) -=
                                        factor(i, k) * vals[lbz](k, j);
                                }
                            }
                        }
                        kbz++;
                        lbz++;
                    } else if (kcol < lcol) {
                        kbz++;
                    } else {
                        lbz++;
                    }
                }

                // Replace (ibrow,jcol)
                for (int i = 0; i < bs; i++)
                    for (int j = 0; j < bs; j++) {
                        vals[jbz](i, j) = factor(i, j);
                    }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void compute_bilu(const std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::Fbcsr<ValueType, IndexType> *const sysmat)
{
    const int bs = sysmat->get_block_size();
    const auto row_ptrs = sysmat->get_const_row_ptrs();
    const auto col_idxs = sysmat->get_const_col_idxs();
    const auto values = sysmat->get_values();
    const IndexType nbrows = sysmat->get_num_block_rows();

    if (bs == 2) {
        compute_bilu_inplace_impl<2>(nbrows, row_ptrs, col_idxs, values);
    }
    if (bs == 3) {
        compute_bilu_inplace_impl<3>(nbrows, row_ptrs, col_idxs, values);
    } else if (bs == 4) {
        compute_bilu_inplace_impl<4>(nbrows, row_ptrs, col_idxs, values);
    } else if (bs == 7) {
        compute_bilu_inplace_impl<7>(nbrows, row_ptrs, col_idxs, values);
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           " block size = " + std::to_string(bs));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BILU_COMPUTE_BLU_KERNEL);


}  // namespace bilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
