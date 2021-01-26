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


template <typename ValueType, typename IndexType, int bs>
static blockutils::FixedBlock<ValueType, bs, bs> extract_diag_block(
    const IndexType blk_row, const IndexType *const browptr,
    const IndexType *const bcolidx,
    const blockutils::FixedBlock<ValueType, bs, bs> *const vals)
{
    const IndexType *const didx = std::find(
        bcolidx + browptr[blk_row], bcolidx + browptr[blk_row + 1], blk_row);
    const ptrdiff_t dbz = didx - bcolidx;
    blockutils::FixedBlock<ValueType, bs, bs> diag;
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            diag(i, j) = vals[dbz](i, j);
        }
    }
    return diag;
}

template <typename ValueType, typename IndexType, int bs>
static void compute_bilu_impl(
    const std::shared_ptr<const ReferenceExecutor> exec,
    matrix::Fbcsr<ValueType, IndexType> *const sysmat)
{
    const IndexType *const col_idxs = sysmat->get_const_col_idxs();
    const IndexType *const row_ptrs = sysmat->get_const_row_ptrs();

    using Blk_t = blockutils::FixedBlock<ValueType, bs, bs>;
    const auto vals = reinterpret_cast<Blk_t *>(sysmat->get_values());

    for (IndexType ibrow = 0;
         ibrow < static_cast<IndexType>(sysmat->get_num_block_rows());
         ++ibrow) {
        for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
             ibz++) {
            const auto jbcol = col_idxs[ibz];

            Blk_t sum;
            ValueType *const sumarr = &sum(0, 0);
            const ValueType *const valarr = &vals[ibz](0, 0);
            for (int i = 0; i < bs * bs; i++) {
                sumarr[i] = valarr[i];
            }

            // Calculate: sum = system_matrix(ibrow, jbcol) -
            //   block_dot(l_factor(ibrow, :), u_factor(:, jbcol))
            for (IndexType kbz = row_ptrs[ibrow]; kbz < row_ptrs[ibrow + 1];
                 kbz++) {
                const auto kcol = col_idxs[kbz];

                // we only cover k s.t. k < j and k < i
                if (kcol >= jbcol || kcol >= ibrow) continue;

                // search the kcol'th row for the jbcol'th column
                IndexType foundbz = -1;
                for (IndexType ubz = row_ptrs[kcol]; ubz < row_ptrs[kcol + 1];
                     ubz++) {
                    // skip entries in the lower-triangular part
                    if (col_idxs[ubz] < kcol) continue;
                    // stop if the matching entry is found
                    else if (col_idxs[ubz] == jbcol) {
                        foundbz = ubz;
                        break;
                    }
                }
                if (foundbz != -1) {
                    for (int i = 0; i < bs; i++) {
                        for (int j = 0; j < bs; j++) {
                            for (int k = 0; k < bs; k++)
                                sum(i, j) -=
                                    vals[kbz](i, k) * vals[foundbz](k, j);
                        }
                    }
                }
            }

            if (ibrow > jbcol) {
                // invert diagonal block
                Blk_t invU =
                    extract_diag_block(jbcol, row_ptrs, col_idxs, vals);

                int perm[bs];
                for (int i = 0; i < bs; i++) perm[i] = i;

                const bool invflag = invert_block<ValueType, bs>(perm, invU);
                if (!invflag)
                    printf(" Could not invert diag block at blk row %ld!",
                           static_cast<long int>(ibrow));
                permute_block(invU, perm);

                for (int i = 0; i < bs; i++)
                    for (int j = 0; j < bs; j++) {
                        vals[ibz](i, j) = 0;
                        for (int k = 0; k < bs; k++) {
                            vals[ibz](i, j) += sum(i, k) * invU(k, j);
                        }
                    }
            } else {
                // modify entry in U
                for (int i = 0; i < bs; i++) {
                    for (int j = 0; j < bs; j++) {
                        vals[ibz](i, j) = sum(i, j);
                    }
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

    if (bs == 2) compute_bilu_impl<ValueType, IndexType, 2>(exec, sysmat);
    if (bs == 3)
        compute_bilu_impl<ValueType, IndexType, 3>(exec, sysmat);
    else if (bs == 4)
        compute_bilu_impl<ValueType, IndexType, 4>(exec, sysmat);
    else
        throw NotSupported(__FILE__, __LINE__, __func__,
                           " block size = " + std::to_string(bs));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BILU_COMPUTE_BLU_KERNEL);


}  // namespace bilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
