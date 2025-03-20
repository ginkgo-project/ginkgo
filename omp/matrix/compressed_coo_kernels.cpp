// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/compressed_coo_kernels.hpp"

#include <array>

#include <omp.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/components/bitvector.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace compressed_coo {


template <int block_size, typename ValueType, typename IndexType>
void spmv2_blocked(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::CompactRowCoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   matrix::Dense<ValueType>* c, ValueType scale)
{
    GKO_ASSERT(b->get_size()[1] > block_size);
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto nnz = a->get_num_stored_elements();
    const auto num_rhs = b->get_size()[1];
    const auto rounded_rhs = num_rhs / block_size * block_size;
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto coo_row_bits = a->get_const_row_bits();
    const auto coo_row_ranks = a->get_const_row_bit_ranks();
    device_bitvector<IndexType> coo_row_bv{coo_row_bits, coo_row_ranks,
                                           static_cast<int64>(nnz)};

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first =
                begin > 0 ? coo_row_bv.rank(begin - 1) : sentinel_row;
            const auto last = end < nnz ? coo_row_bv.rank(end) : sentinel_row;
            auto nz = begin;
            std::array<ValueType, block_size> partial_sum;
            if (first != sentinel_row) {
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    // handle row overlap with previous thread: block partial
                    // sums
                    partial_sum.fill(zero<ValueType>());
                    for (auto local_nz = nz;
                         local_nz < end && coo_row_bv.rank(local_nz) == first;
                         local_nz++) {
                        const auto col = coo_row_bv.rank(local_nz);
#pragma unroll
                        for (size_type i = 0; i < block_size; i++) {
                            const auto rhs = i + rhs_base;
                            partial_sum[i] +=
                                scale * coo_val[local_nz] * b->at(col, rhs);
                        }
                    }
                    // handle row overlap with previous thread: block add to
                    // memory
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        atomic_add(c->at(first, rhs), partial_sum[i]);
                    }
                }
                // handle row overlap with previous thread: remainder partial
                // sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end && coo_row_bv.rank(nz) == first; nz++) {
                    const auto row = first;
                    const auto col = coo_col[nz];
                    for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                        partial_sum[rhs - rounded_rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with previous thread: remainder add to
                // memory
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(first, rhs),
                               partial_sum[rhs - rounded_rhs]);
                }
            }
            // handle non-overlapping rows
            for (; nz < end && coo_row_bv.rank(nz) != last; nz++) {
                const auto row = coo_row_bv.rank(nz);
                const auto col = coo_col[nz];
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        c->at(row, rhs) +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
                }
            }
            if (last != sentinel_row) {
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    // handle row overlap with following thread: block partial
                    // sums
                    partial_sum.fill(zero<ValueType>());
                    for (auto local_nz = nz; local_nz < end; local_nz++) {
                        const auto col = coo_col[local_nz];
#pragma unroll
                        for (size_type i = 0; i < block_size; i++) {
                            const auto rhs = i + rhs_base;
                            partial_sum[i] +=
                                scale * coo_val[local_nz] * b->at(col, rhs);
                        }
                    }
                    // handle row overlap with following thread: block add to
                    // memory
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        const auto row = last;
                        atomic_add(c->at(row, rhs), partial_sum[i]);
                    }
                }
                // handle row overlap with following thread: block partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end; nz++) {
                    const auto col = coo_col[nz];
                    for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                        partial_sum[rhs - rounded_rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with following thread: block add to memory
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    const auto row = last;
                    atomic_add(c->at(row, rhs), partial_sum[rhs - rounded_rhs]);
                }
            }
        }
    }
}


template <int num_rhs, typename ValueType, typename IndexType>
void spmv2_small_rhs(std::shared_ptr<const OmpExecutor> exec,
                     const matrix::CompactRowCoo<ValueType, IndexType>* a,
                     const matrix::Dense<ValueType>* b,
                     matrix::Dense<ValueType>* c, ValueType scale)
{
    GKO_ASSERT(b->get_size()[1] == num_rhs);
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto nnz = a->get_num_stored_elements();
    const auto coo_row_bits = a->get_const_row_bits();
    const auto coo_row_ranks = a->get_const_row_bit_ranks();
    device_bitvector<IndexType> coo_row_bv{coo_row_bits, coo_row_ranks,
                                           static_cast<int64>(nnz)};

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first =
                begin > 0 ? coo_row_bv.rank(begin - 1) : sentinel_row;
            const auto last = end < nnz ? coo_row_bv.rank(end) : sentinel_row;
            auto nz = begin;
            std::array<ValueType, num_rhs> partial_sum;
            if (first != sentinel_row) {
                // handle row overlap with previous thread: partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end && coo_row_bv.rank(nz) == first; nz++) {
                    const auto col = coo_row_bv.rank(nz);
#pragma unroll
                    for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                        partial_sum[rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with previous thread: add to memory
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(first, rhs), partial_sum[rhs]);
                }
            }
            // handle non-overlapping rows
            for (; nz < end && coo_row_bv.rank(nz) != last; nz++) {
                const auto row = coo_row_bv.rank(nz);
                const auto col = coo_col[nz];
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
                }
            }
            if (last != sentinel_row) {
                // handle row overlap with following thread: partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end; nz++) {
                    const auto col = coo_col[nz];
#pragma unroll
                    for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                        partial_sum[rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with following thread: add to memory
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    const auto row = last;
                    atomic_add(c->at(row, rhs), partial_sum[rhs]);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void generic_spmv2(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::CompactRowCoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   matrix::Dense<ValueType>* c, ValueType scale)
{
    const auto num_rhs = b->get_size()[1];
    if (num_rhs <= 0) {
        return;
    }
    if (num_rhs == 1) {
        spmv2_small_rhs<1>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 2) {
        spmv2_small_rhs<2>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 3) {
        spmv2_small_rhs<3>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 4) {
        spmv2_small_rhs<4>(exec, a, b, c, scale);
        return;
    }
    spmv2_blocked<4>(exec, a, b, c, scale);
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::CompactRowCoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    generic_spmv2(exec, a, b, c, one<ValueType>());
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


}  // namespace compressed_coo
}  // namespace omp
}  // namespace kernels
}  // namespace gko
