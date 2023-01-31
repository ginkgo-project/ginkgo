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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 * BlockBlockJacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchBlockJacobi final {
private:
    inline void common_generate_for_all_system_matrix_types(size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            storage_scheme_.get_batch_offset(num_blocks_, batch_id);
    }

public:
    using value_type = ValueType;

    /**
     *
     * @param num_blocks  Number of diagonal blocks in a matrix
     * @param storage_scheme diagonal blocks storage scheme
     * @param blocks_arr_batch array of diagonal blocks for the batch
     * @param block_ptrs_arr array of block pointers
     *
     */
    BatchBlockJacobi(const uint32, const size_type num_blocks,
                     const gko::preconditioner::batched_blocks_storage_scheme&
                         storage_scheme,
                     const value_type* const blocks_arr_batch,
                     const int* const block_ptrs_arr, const int* const)
        : num_blocks_{num_blocks},
          storage_scheme_{storage_scheme},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr}
    {}

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
    {
        return 0;
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        // Structure-aware SpMV
        for (int bidx = 0; bidx < num_blocks_; bidx++) {
            const int row_st = block_ptrs_arr_[bidx];       // inclusive
            const int row_end = block_ptrs_arr_[bidx + 1];  // exclusive
            const int bsize = row_end - row_st;

            const auto offset = storage_scheme_.get_block_offset(bidx);

            for (int row = row_st; row < row_end; row++) {
                ValueType sum = zero<ValueType>();
                for (int col = 0; col < bsize; col++) {
                    const auto val =
                        blocks_arr_entry_[offset +
                                          (row - row_st) *
                                              storage_scheme_.get_stride() +
                                          col];
                    sum += val * r.values[col + row_st];
                }

                z.values[row] = sum;
            }
        }
    }

private:
    const size_type num_blocks_;
    const gko::preconditioner::batched_blocks_storage_scheme storage_scheme_;
    const value_type* const blocks_arr_batch_;
    const value_type* blocks_arr_entry_;
    const int* const block_ptrs_arr_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
