// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/compressed_coo_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/intrinsics.hpp"
#include "core/components/bitvector.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace compressed_coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::CompactRowCoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    const auto num_cols = b->get_size()[1];
    const auto nnz = a->get_num_stored_elements();
    const auto row_bits = a->get_const_row_bits();
    const auto row_ranks = a->get_const_row_bit_ranks();
    const auto cols = a->get_const_col_idxs();
    const auto vals = a->get_const_values();
    device_bitvector<IndexType> row_bv{row_bits, row_ranks,
                                       static_cast<int64>(nnz)};
    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
        const auto row = row_bv.rank(i);
        const auto col = cols[i];
        const auto val = vals[i];
        for (size_type j = 0; j < num_cols; j++) {
            c->at(row, j) += val * b->at(col, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CRCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::CompactRowCompressedColumnCoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    // TODO
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CRCOCOCOO_SPMV_KERNEL);


template <typename IndexType>
void idxs_to_bits(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* idxs, size_type nnz, uint32* bits,
                  IndexType* ranks)
{
    if (nnz == 0) {
        return;
    }
    const auto num_blocks = ceildiv(nnz, 32);
    std::fill_n(bits, num_blocks, 0);
    for (size_type i = 0; i < nnz - 1; i++) {
        if (idxs[i + 1] > idxs[i]) {
            assert(idxs[i + 1] == idxs[i] + 1);
            bits[i / 32] |= uint32{1} << (i % 32);
        }
    }
    IndexType rank = 0;
    for (int64 i = 0; i < num_blocks; i++) {
        const auto block = bits[i];
        ranks[i] = rank;
        rank += detail::popcount(block);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_IDXS_TO_BITS_KERNEL);


template <typename IndexType>
void bits_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                  const uint32* bits, const IndexType* ranks, size_type nnz,
                  IndexType* idxs)
{
    device_bitvector<IndexType> bv{bits, ranks, static_cast<int64>(nnz)};
    for (size_type i = 0; i < nnz; i++) {
        idxs[i] = bv.rank(i);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_BITS_TO_IDXS_KERNEL);


}  // namespace compressed_coo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
