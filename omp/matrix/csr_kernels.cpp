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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "omp/components/csr_spgeam.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) *= vbeta;
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += valpha * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


namespace {


/**
 * @internal
 *
 * Entry in a heap storing a column index and associated non-zero index
 * (and row end) from a matrix.
 *
 * @tparam ValueType  The value type for matrices.
 * @tparam IndexType  The index type for matrices.
 */
template <typename ValueType, typename IndexType>
struct col_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    IndexType idx;
    IndexType end;
    IndexType col;

    ValueType val() const { return zero<ValueType>(); }

    col_heap_element(IndexType idx, IndexType end, IndexType col, ValueType)
        : idx{idx}, end{end}, col{col}
    {}
};


/**
 * @internal
 *
 * Entry in a heap storing an entry (value and column index) and associated
 * non-zero index (and row end) from a matrix.
 *
 * @tparam ValueType  The value type for matrices.
 * @tparam IndexType  The index type for matrices.
 */
template <typename ValueType, typename IndexType>
struct val_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;

    IndexType idx;
    IndexType end;
    IndexType col;
    ValueType val_;

    ValueType val() const { return val_; }

    val_heap_element(IndexType idx, IndexType end, IndexType col, ValueType val)
        : idx{idx}, end{end}, col{col}, val_{val}
    {}
};


/**
 * @internal
 *
 * Restores the binary heap condition downwards from a given index.
 *
 * The heap condition is: col(child) >= col(parent)
 *
 * @param heap  a pointer to the array containing the heap elements.
 * @param idx  the index of the starting heap node that potentially
 *             violates the heap condition.
 * @param size  the number of elements in the heap.
 * @tparam HeapElement  the element type in the heap. See col_heap_element and
 *                      val_heap_element
 */
template <typename HeapElement>
void sift_down(HeapElement *heap, typename HeapElement::index_type idx,
               typename HeapElement::index_type size)
{
    auto curcol = heap[idx].col;
    while (idx * 2 + 1 < size) {
        auto lchild = idx * 2 + 1;
        auto rchild = min(lchild + 1, size - 1);
        auto lcol = heap[lchild].col;
        auto rcol = heap[rchild].col;
        auto mincol = min(lcol, rcol);
        if (mincol >= curcol) {
            break;
        }
        auto minchild = lcol == mincol ? lchild : rchild;
        std::swap(heap[minchild], heap[idx]);
        idx = minchild;
    }
}


/**
 * @internal
 *
 * Generic SpGEMM implementation for a single output row of A * B using binary
 * heap-based multiway merging.
 *
 * @param row  The row for which to compute the SpGEMM
 * @param a  The input matrix A
 * @param b  The input matrix B (its column indices must be sorted within each
 *           row!)
 * @param heap  The heap to use for this implementation. It must have as many
 *              entries as the input row has non-zeros.
 * @param init_cb  function to initialize the state for a single row. Its return
 *                 value will be updated by subsequent calls of other callbacks,
 *                 and then returned by this function. Its signature must be
 *                 compatible with `return_type state = init_cb(row)`.
 * @param step_cb  function that will be called for each accumulation from an
 *                 entry of B into the output state. Its signature must be
 *                 compatible with `step_cb(value, column, state)`.
 * @param col_cb  function that will be called once for each output column after
 *                all accumulations into it are completed. Its signature must be
 *                compatible with `col_cb(column, state)`.
 * @return the value initialized by init_cb and updated by step_cb and col_cb
 * @note If the columns of B are not sorted, the output may have duplicate
 *       column entries.
 *
 * @tparam HeapElement  the heap element type. See col_heap_element and
 *                      val_heap_element
 * @tparam InitCallback  functor type for init_cb
 * @tparam StepCallback  functor type for step_cb
 * @tparam ColCallback  functor type for col_cb
 */
template <typename HeapElement, typename InitCallback, typename StepCallback,
          typename ColCallback>
auto spgemm_multiway_merge(size_type row,
                           const typename HeapElement::matrix_type *a,
                           const typename HeapElement::matrix_type *b,
                           HeapElement *heap, InitCallback init_cb,
                           StepCallback step_cb, ColCallback col_cb)
    -> decltype(init_cb(0))
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_cols = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_cols = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    auto a_begin = a_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];

    using index_type = typename HeapElement::index_type;
    constexpr auto sentinel = std::numeric_limits<index_type>::max();

    auto state = init_cb(row);

    // initialize the heap
    for (auto a_nz = a_begin; a_nz < a_end; ++a_nz) {
        auto b_row = a_cols[a_nz];
        auto b_begin = b_row_ptrs[b_row];
        auto b_end = b_row_ptrs[b_row + 1];
        heap[a_nz] = {b_begin, b_end,
                      checked_load(b_cols, b_begin, b_end, sentinel),
                      a_vals[a_nz]};
    }

    if (a_begin != a_end) {
        // make heap:
        auto a_size = a_end - a_begin;
        for (auto i = (a_size - 2) / 2; i >= 0; --i) {
            sift_down(heap + a_begin, i, a_size);
        }
        auto &top = heap[a_begin];
        auto &bot = heap[a_end - 1];
        auto col = top.col;

        while (top.col != sentinel) {
            step_cb(b_vals[top.idx] * top.val(), top.col, state);
            // move to the next element
            top.idx++;
            top.col = checked_load(b_cols, top.idx, top.end, sentinel);
            // restore heap property
            // pop_heap swaps top and bot, we need to prevent that
            // so that we do a simple sift_down instead
            sift_down(heap + a_begin, index_type{}, a_size);
            if (top.col != col) {
                col_cb(col, state);
            }
            col = top.col;
        }
    }

    return state;
}


}  // namespace


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const OmpExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    auto c_row_ptrs = c->get_row_ptrs();

    Array<col_heap_element<ValueType, IndexType>> col_heap_array(
        exec, a->get_num_stored_elements());

    auto col_heap = col_heap_array.get_data();

    // first sweep: count nnz for each row
#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        c_row_ptrs[a_row] = spgemm_multiway_merge(
            a_row, a, b, col_heap, [](size_type) { return IndexType{}; },
            [](ValueType, IndexType, IndexType &) {},
            [](IndexType, IndexType &nnz) { nnz++; });
    }

    col_heap_array.clear();

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        spgemm_multiway_merge(
            a_row, a, b, heap,
            [&](size_type row) {
                return std::make_pair(zero<ValueType>(), c_row_ptrs[row]);
            },
            [](ValueType val, IndexType,
               std::pair<ValueType, IndexType> &state) { state.first += val; },
            [&](IndexType col, std::pair<ValueType, IndexType> &state) {
                c_col_idxs[state.second] = col;
                c_vals[state.second] = state.first;
                state.first = zero<ValueType>();
                state.second++;
            });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const OmpExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();
    auto d_row_ptrs = d->get_const_row_ptrs();
    auto d_cols = d->get_const_col_idxs();
    auto d_vals = d->get_const_values();

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType> *>(heap);

    // first sweep: count nnz for each row
#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        auto d_nz = d_row_ptrs[a_row];
        auto d_end = d_row_ptrs[a_row + 1];
        auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
        c_row_ptrs[a_row] = spgemm_multiway_merge(
            a_row, a, b, col_heap, [](size_type row) { return IndexType{}; },
            [](ValueType, IndexType, IndexType &) {},
            [&](IndexType col, IndexType &nnz) {
                // skip smaller elements from d
                while (d_col <= col) {
                    d_nz++;
                    nnz += d_col != col;
                    d_col = checked_load(d_cols, d_nz, d_end, sentinel);
                }
                nnz++;
            });
        // handle the remaining columns from d
        c_row_ptrs[a_row] += d_end - d_nz;
    }

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        auto d_nz = d_row_ptrs[a_row];
        auto d_end = d_row_ptrs[a_row + 1];
        auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
        auto d_val = checked_load(d_vals, d_nz, d_end, zero<ValueType>());
        auto c_nz =
            spgemm_multiway_merge(
                a_row, a, b, heap,
                [&](size_type row) {
                    return std::make_pair(zero<ValueType>(), c_row_ptrs[row]);
                },
                [](ValueType val, IndexType,
                   std::pair<ValueType, IndexType> &state) {
                    state.first += val;
                },
                [&](IndexType col, std::pair<ValueType, IndexType> &state) {
                    // handle smaller elements from d
                    ValueType part_d_val{};
                    while (d_col <= col) {
                        if (d_col == col) {
                            part_d_val = d_val;
                        } else {
                            c_col_idxs[state.second] = d_col;
                            c_vals[state.second] = vbeta * d_val;
                            state.second++;
                        }
                        d_nz++;
                        d_col = checked_load(d_cols, d_nz, d_end, sentinel);
                        d_val = checked_load(d_vals, d_nz, d_end,
                                             zero<ValueType>());
                    }
                    c_col_idxs[state.second] = col;
                    c_vals[state.second] =
                        vbeta * part_d_val + valpha * state.first;
                    state.first = zero<ValueType>();
                    state.second++;
                })
                .second;
        // handle remaining elements from d
        while (d_col < sentinel) {
            c_col_idxs[c_nz] = d_col;
            c_vals[c_nz] = vbeta * d_val;
            c_nz++;
            d_nz++;
            d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            d_val = checked_load(d_vals, d_nz, d_end, zero<ValueType>());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const OmpExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    abstract_spgeam(
        a, b, [](IndexType) { return IndexType{}; },
        [](IndexType, IndexType, ValueType, ValueType, IndexType &nnz) {
            ++nnz;
        },
        [&](IndexType row, IndexType nnz) { c_row_ptrs[row] = nnz; });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    abstract_spgeam(
        a, b, [&](IndexType row) { return c_row_ptrs[row]; },
        [&](IndexType, IndexType col, ValueType a_val, ValueType b_val,
            IndexType &nz) {
            c_vals[nz] = valpha * a_val + vbeta * b_val;
            c_col_idxs[nz] = col;
            ++nz;
        },
        [](IndexType, IndexType) {});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_span(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *source, const span &row_span,
    const span &col_span, Array<size_type> *row_nnz) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);

template <typename ValueType, typename IndexType>
void block_approx(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Csr<ValueType, IndexType> *source,
                  matrix::Csr<ValueType, IndexType> *result,
                  Array<size_type> *row_nnz,
                  size_type block_offset) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_BLOCK_APPROX_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto vals = source->get_const_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            result->at(row, col) = zero<ValueType>();
        }
        for (size_type i = row_ptrs[row];
             i < static_cast<size_type>(row_ptrs[row + 1]); ++i) {
            result->at(row, col_idxs[i]) = vals[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_csr_to_csc(size_type num_rows, const IndexType *row_ptrs,
                               const IndexType *col_idxs,
                               const ValueType *csr_vals, IndexType *row_idxs,
                               IndexType *col_ptrs, ValueType *csc_vals,
                               UnaryOperator op)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = row;
            csc_vals[dest_idx] = op(csr_vals[i]);
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::Csr<ValueType, IndexType> *trans,
                             const matrix::Csr<ValueType, IndexType> *orig,
                             UnaryOperator op)
{
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto trans_vals = trans->get_values();
    auto orig_vals = orig->get_const_values();

    auto orig_num_cols = orig->get_size()[1];
    auto orig_num_rows = orig->get_size()[0];
    auto orig_nnz = orig_row_ptrs[orig_num_rows];

    trans_row_ptrs[0] = 0;
    convert_unsorted_idxs_to_ptrs(orig_col_idxs, orig_nnz, trans_row_ptrs + 1,
                                  orig_num_cols);

    convert_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs, orig_vals,
                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const OmpExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const OmpExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_lim = strategy->get_coo_nnz();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    const auto max_nnz_per_row = result->get_ell_num_stored_elements_per_row();

// Initial Hybrid Matrix
#pragma omp parallel for
    for (size_type i = 0; i < max_nnz_per_row; i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = 0;
        }
    }

    const auto csr_row_ptrs = source->get_const_row_ptrs();
    const auto csr_vals = source->get_const_values();
    auto coo_offset = Array<IndexType>(exec, num_rows);
    auto coo_offset_val = coo_offset.get_data();

    coo_offset_val[0] = 0;
#pragma omp parallel for
    for (size_type i = 1; i < num_rows; i++) {
        auto temp = csr_row_ptrs[i] - csr_row_ptrs[i - 1];
        coo_offset_val[i] = (temp > max_nnz_per_row) * (temp - max_nnz_per_row);
    }

    auto workspace = Array<IndexType>(exec, num_rows);
    auto workspace_val = workspace.get_data();
    for (size_type i = 1; i < num_rows; i <<= 1) {
#pragma omp parallel for
        for (size_type j = i; j < num_rows; j++) {
            workspace_val[j] = coo_offset_val[j] + coo_offset_val[j - i];
        }
#pragma omp parallel for
        for (size_type j = i; j < num_rows; j++) {
            coo_offset_val[j] = workspace_val[j];
        }
    }

#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; row++) {
        size_type ell_idx = 0;
        size_type csr_idx = csr_row_ptrs[row];
        size_type coo_idx = coo_offset_val[row];
        while (csr_idx < csr_row_ptrs[row + 1]) {
            const auto val = csr_vals[csr_idx];
            if (ell_idx < ell_lim) {
                result->ell_val_at(row, ell_idx) = val;
                result->ell_col_at(row, ell_idx) =
                    source->get_const_col_idxs()[csr_idx];
                ell_idx++;
            } else {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = source->get_const_col_idxs()[csr_idx];
                coo_row[coo_idx] = row;
                coo_idx++;
            }
            csr_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType *perm,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Csr<ValueType, IndexType> *permuted)
{
    auto in_row_ptrs = orig->get_const_row_ptrs();
    auto in_col_idxs = orig->get_const_col_idxs();
    auto in_vals = orig->get_const_values();
    auto p_row_ptrs = permuted->get_row_ptrs();
    auto p_col_idxs = permuted->get_col_idxs();
    auto p_vals = permuted->get_values();
    size_type num_rows = orig->get_size()[0];

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        p_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum(exec, p_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = p_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        for (IndexType i = 0; i < row_size; ++i) {
            p_col_idxs[dst_begin + i] = perm[in_col_idxs[src_begin + i]];
            p_vals[dst_begin + i] = in_vals[src_begin + i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const OmpExecutor> exec, const IndexType *perm,
                 const matrix::Csr<ValueType, IndexType> *orig,
                 matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto orig_vals = orig->get_const_values();
    auto rp_row_ptrs = row_permuted->get_row_ptrs();
    auto rp_col_idxs = row_permuted->get_col_idxs();
    auto rp_vals = row_permuted->get_values();
    size_type num_rows = orig->get_size()[0];

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = perm[row];
        auto dst_row = row;
        rp_row_ptrs[dst_row] =
            orig_row_ptrs[src_row + 1] - orig_row_ptrs[src_row];
    }
    components::prefix_sum(exec, rp_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = perm[row];
        auto dst_row = row;
        auto src_begin = orig_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = orig_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(orig_col_idxs + src_begin, row_size,
                    rp_col_idxs + dst_begin);
        std::copy_n(orig_vals + src_begin, row_size, rp_vals + dst_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const OmpExecutor> exec,
                         const IndexType *perm,
                         const matrix::Csr<ValueType, IndexType> *orig,
                         matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto orig_vals = orig->get_const_values();
    auto rp_row_ptrs = row_permuted->get_row_ptrs();
    auto rp_col_idxs = row_permuted->get_col_idxs();
    auto rp_vals = row_permuted->get_values();
    size_type num_rows = orig->get_size()[0];

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        rp_row_ptrs[dst_row] =
            orig_row_ptrs[src_row + 1] - orig_row_ptrs[src_row];
    }
    components::prefix_sum(exec, rp_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        auto src_begin = orig_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = orig_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(orig_col_idxs + src_begin, row_size,
                    rp_col_idxs + dst_begin);
        std::copy_n(orig_vals + src_begin, row_size, rp_vals + dst_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const OmpExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    auto row_nnz_val = result->get_data();

#pragma omp parallel for
    for (size_type i = 0; i < result->get_num_elems(); i++) {
        row_nnz_val[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
{
    auto values = to_sort->get_values();
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
#pragma omp parallel for
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        auto helper = detail::IteratorFactory<IndexType, ValueType>(
            col_idxs + start_row_idx, values + start_row_idx, row_nnz);
        std::sort(helper.begin(), helper.end());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    bool local_is_sorted = true;
#pragma omp parallel for reduction(&& : local_is_sorted)
    for (size_type i = 0; i < size[0]; ++i) {
        // Skip comparison if any thread detects that it is not sorted
        if (local_is_sorted) {
            for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
                if (col_idxs[idx - 1] > col_idxs[idx]) {
                    local_is_sorted = false;
                    break;
                }
            }
        }
    }
    *is_sorted = local_is_sorted;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const auto diag_size = diag->get_size()[0];
    auto diag_values = diag->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < diag_size; ++row) {
        for (size_type idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] == row) {
                diag_values[row] = values[idx];
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


}  // namespace csr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
