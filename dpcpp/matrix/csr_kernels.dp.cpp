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
#include <numeric>
#include <utility>


#include <CL/sycl.hpp>


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
#include "dpcpp/components/format_conversion.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b,
          matrix::Dense<ValueType> *c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c) GKO_NOT_IMPLEMENTED;

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
                           const typename HeapElement::index_type *a_row_ptrs,
                           const typename HeapElement::index_type *a_cols,
                           const typename HeapElement::value_type *a_vals,
                           const typename HeapElement::index_type *b_row_ptrs,
                           const typename HeapElement::index_type *b_cols,
                           const typename HeapElement::value_type *b_vals,
                           HeapElement *heap, InitCallback init_cb,
                           StepCallback step_cb, ColCallback col_cb)
    -> decltype(init_cb(0))
{
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
void spgemm(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto a_vals = a->get_const_values();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    const auto b_vals = b->get_const_values();
    auto c_row_ptrs = c->get_row_ptrs();
    auto queue = exec->get_queue();

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType> *>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type) { return IndexType{}; },
                [](ValueType, IndexType, IndexType &) {},
                [](IndexType, IndexType &nnz) { nnz++; });
        });
    });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                heap,
                [&](size_type row) {
                    return std::make_pair(zero<ValueType>(), c_row_ptrs[row]);
                },
                [](ValueType val, IndexType,
                   std::pair<ValueType, IndexType> &state) {
                    state.first += val;
                },
                [&](IndexType col, std::pair<ValueType, IndexType> &state) {
                    c_col_idxs[state.second] = col;
                    c_vals[state.second] = state.first;
                    state.first = zero<ValueType>();
                    state.second++;
                });
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto a_vals = a->get_const_values();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    const auto b_vals = b->get_const_values();
    const auto d_row_ptrs = d->get_const_row_ptrs();
    const auto d_cols = d->get_const_col_idxs();
    const auto d_vals = d->get_const_values();
    auto c_row_ptrs = c->get_row_ptrs();
    const auto alpha_vals = alpha->get_const_values();
    const auto beta_vals = beta->get_const_values();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    auto queue = exec->get_queue();

    // first sweep: count nnz for each row

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType> *>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            auto d_nz = d_row_ptrs[a_row];
            const auto d_end = d_row_ptrs[a_row + 1];
            auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type row) { return IndexType{}; },
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
        });
    });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);

    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            auto d_nz = d_row_ptrs[a_row];
            const auto d_end = d_row_ptrs[a_row + 1];
            auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            auto d_val = checked_load(d_vals, d_nz, d_end, zero<ValueType>());
            const auto valpha = alpha_vals[0];
            const auto vbeta = beta_vals[0];
            auto c_nz =
                spgemm_multiway_merge(
                    a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols,
                    b_vals, heap,
                    [&](size_type row) {
                        return std::make_pair(zero<ValueType>(),
                                              c_row_ptrs[row]);
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
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    auto c_row_ptrs = c->get_row_ptrs();
    auto queue = exec->get_queue();

    // count number of non-zeros per row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            auto a_idx = a_row_ptrs[row];
            const auto a_end = a_row_ptrs[row + 1];
            auto b_idx = b_row_ptrs[row];
            const auto b_end = b_row_ptrs[row + 1];
            IndexType row_nnz{};
            while (a_idx < a_end || b_idx < b_end) {
                const auto a_col = checked_load(a_cols, a_idx, a_end, sentinel);
                const auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                row_nnz++;
                a_idx += (a_col <= b_col) ? 1 : 0;
                b_idx += (b_col <= a_col) ? 1 : 0;
            }
            c_row_ptrs[row] = row_nnz;
        });
    });

    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_cols = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    const auto a_vals = a->get_const_values();
    const auto b_vals = b->get_const_values();
    const auto alpha_vals = alpha->get_const_values();
    const auto beta_vals = beta->get_const_values();

    // count number of non-zeros per row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            auto a_idx = a_row_ptrs[row];
            const auto a_end = a_row_ptrs[row + 1];
            auto b_idx = b_row_ptrs[row];
            const auto b_end = b_row_ptrs[row + 1];
            const auto alpha = alpha_vals[0];
            const auto beta = beta_vals[0];
            auto c_nz = c_row_ptrs[row];
            while (a_idx < a_end || b_idx < b_end) {
                const auto a_col = checked_load(a_cols, a_idx, a_end, sentinel);
                const auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                const bool use_a = a_col <= b_col;
                const bool use_b = b_col <= a_col;
                const auto a_val = use_a ? a_vals[a_idx] : zero<ValueType>();
                const auto b_val = use_b ? b_vals[b_idx] : zero<ValueType>();
                c_cols[c_nz] = std::min(a_col, b_col);
                c_vals[c_nz] = alpha * a_val + beta * b_val;
                c_nz++;
                a_idx += use_a ? 1 : 0;
                b_idx += use_b ? 1 : 0;
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DpcppExecutor> exec,
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
                               UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const DpcppExecutor> exec,
                             matrix::Csr<ValueType, IndexType> *trans,
                             const matrix::Csr<ValueType, IndexType> *orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename IndexType>
void invert_permutation(std::shared_ptr<const DpcppExecutor> exec,
                        size_type size, const IndexType *permutation_indices,
                        IndexType *inv_permutation) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType *perm,
    const matrix::Csr<ValueType, IndexType> *orig,
    matrix::Csr<ValueType, IndexType> *permuted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType *perm,
    const matrix::Csr<ValueType, IndexType> *orig,
    matrix::Csr<ValueType, IndexType> *row_permuted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType *perm,
    const matrix::Csr<ValueType, IndexType> *orig,
    matrix::Csr<ValueType, IndexType> *row_permuted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(
    std::shared_ptr<const DpcppExecutor> exec, const IndexType *perm,
    const matrix::Csr<ValueType, IndexType> *orig,
    matrix::Csr<ValueType, IndexType> *column_permuted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    Array<bool> is_sorted_device_array{exec, {true}};
    const auto num_rows = to_check->get_size()[0];
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto cols = to_check->get_const_col_idxs();
    auto is_sorted_device = is_sorted_device_array.get_data();
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            if (*is_sorted_device) {
                for (auto i = begin; i < end - 1; i++) {
                    if (cols[i] > cols[i + 1]) {
                        *is_sorted_device = false;
                        break;
                    }
                }
            }
        });
    });
    exec->get_master()->copy_from(exec.get(), 1, is_sorted_device, is_sorted);
};

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


}  // namespace csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
