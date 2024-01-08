// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>


#include "core/base/allocator.hpp"
#include "core/base/index_set_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_accessor_helper.hpp"
#include "core/matrix/csr_builder.hpp"
#include "omp/components/csr_spgeam.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Csr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    using arithmetic_type =
        highest_precision<MatrixValueType, InputValueType, OutputValueType>;

    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
    const auto b_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
    auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            auto sum = zero<arithmetic_type>();
            for (size_type k = row_ptrs[row];
                 k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                arithmetic_type val = a_vals(k);
                auto col = col_idxs[k];

                sum += val * b_vals(col, j);
            }
            c_vals(row, j) = sum;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Csr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    using arithmetic_type =
        highest_precision<MatrixValueType, InputValueType, OutputValueType>;

    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    arithmetic_type valpha = alpha->at(0, 0);
    arithmetic_type vbeta = beta->at(0, 0);

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
    const auto b_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
    auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            auto sum = c_vals(row, j) * vbeta;
            for (size_type k = row_ptrs[row];
                 k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                arithmetic_type val = a_vals(k);
                auto col = col_idxs[k];
                sum += valpha * val * b_vals(col, j);
            }
            c_vals(row, j) = sum;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
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
void sift_down(HeapElement* heap, typename HeapElement::index_type idx,
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
                           const typename HeapElement::matrix_type* a,
                           const typename HeapElement::matrix_type* b,
                           HeapElement* heap, InitCallback init_cb,
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
        auto& top = heap[a_begin];
        auto& bot = heap[a_end - 1];
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
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    auto num_rows = a->get_size()[0];
    auto c_row_ptrs = c->get_row_ptrs();

    array<col_heap_element<ValueType, IndexType>> col_heap_array(
        exec, a->get_num_stored_elements());

    auto col_heap = col_heap_array.get_data();

    // first sweep: count nnz for each row
#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        c_row_ptrs[a_row] = spgemm_multiway_merge(
            a_row, a, b, col_heap, [](size_type) { return IndexType{}; },
            [](ValueType, IndexType, IndexType&) {},
            [](IndexType, IndexType& nnz) { nnz++; });
    }

    col_heap_array.clear();

    array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();

    // build row pointers
    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
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
               std::pair<ValueType, IndexType>& state) { state.first += val; },
            [&](IndexType col, std::pair<ValueType, IndexType>& state) {
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
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     const matrix::Dense<ValueType>* beta,
                     const matrix::Csr<ValueType, IndexType>* d,
                     matrix::Csr<ValueType, IndexType>* c)
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

    array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType>*>(heap);

    // first sweep: count nnz for each row
#pragma omp parallel for
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        auto d_nz = d_row_ptrs[a_row];
        auto d_end = d_row_ptrs[a_row + 1];
        auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
        c_row_ptrs[a_row] = spgemm_multiway_merge(
            a_row, a, b, col_heap, [](size_type row) { return IndexType{}; },
            [](ValueType, IndexType, IndexType&) {},
            [&](IndexType col, IndexType& nnz) {
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
    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
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
                   std::pair<ValueType, IndexType>& state) {
                    state.first += val;
                },
                [&](IndexType col, std::pair<ValueType, IndexType>& state) {
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
            const matrix::Dense<ValueType>* alpha,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Dense<ValueType>* beta,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    abstract_spgeam(
        a, b, [](IndexType) { return IndexType{}; },
        [](IndexType, IndexType, ValueType, ValueType, IndexType& nnz) {
            ++nnz;
        },
        [&](IndexType row, IndexType nnz) { c_row_ptrs[row] = nnz; });

    // build row pointers
    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    abstract_spgeam(
        a, b, [&](IndexType row) { return c_row_ptrs[row]; },
        [&](IndexType, IndexType col, ValueType a_val, ValueType b_val,
            IndexType& nz) {
            c_vals[nz] = valpha * a_val + vbeta * b_val;
            c_col_idxs[nz] = col;
            ++nz;
        },
        [](IndexType, IndexType) {});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Csr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto vals = source->get_const_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = row_ptrs[row];
             i < static_cast<size_type>(row_ptrs[row + 1]); ++i) {
            result->at(row, col_idxs[i]) = vals[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source, int bs,
                      array<IndexType>& row_ptrs, array<IndexType>& col_idxs,
                      array<ValueType>& values)
{
    using entry = matrix_data_entry<ValueType, IndexType>;
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto num_block_rows = num_rows / bs;
    const auto num_block_cols = num_cols / bs;
    const auto in_row_ptrs = source->get_const_row_ptrs();
    const auto in_cols = source->get_const_col_idxs();
    const auto in_vals = source->get_const_values();
    const auto nnz = source->get_num_stored_elements();
    auto out_row_ptrs = row_ptrs.get_data();
    array<entry> entry_array{exec, nnz};
    auto entries = entry_array.get_data();
    for (IndexType row = 0; row < num_rows; row++) {
        for (auto nz = in_row_ptrs[row]; nz < in_row_ptrs[row + 1]; nz++) {
            entries[nz] = {row, in_cols[nz], in_vals[nz]};
        }
    }
    auto to_block = [bs](entry a) {
        return std::make_pair(a.row / bs, a.column / bs);
    };
    // sort by block in row-major order
    std::sort(entries, entries + nnz,
              [&](entry a, entry b) { return to_block(a) < to_block(b); });
    // set row pointers by jumps in block row index
    gko::vector<IndexType> col_idx_vec{{exec}};
    gko::vector<ValueType> value_vec{{exec}};
    int64 block_row = -1;
    int64 block_col = -1;
    for (size_type i = 0; i < nnz; i++) {
        const auto entry = entries[i];
        const auto new_block_row = entry.row / bs;
        const auto new_block_col = entry.column / bs;
        while (new_block_row > block_row) {
            // we finished row block_row, so store its end pointer
            out_row_ptrs[block_row + 1] = col_idx_vec.size();
            block_col = -1;
            ++block_row;
        }
        if (new_block_col != block_col) {
            // we encountered a new column, so insert it with block storage
            col_idx_vec.emplace_back(new_block_col);
            value_vec.resize(value_vec.size() + bs * bs);
            block_col = new_block_col;
        }
        const auto local_row = entry.row % bs;
        const auto local_col = entry.column % bs;
        value_vec[value_vec.size() - bs * bs + local_row + local_col * bs] =
            entry.value;
    }
    while (block_row < static_cast<int64>(row_ptrs.get_size() - 1)) {
        // we finished row block_row, so store its end pointer
        out_row_ptrs[block_row + 1] = col_idx_vec.size();
        ++block_row;
    }
    values.resize_and_reset(value_vec.size());
    col_idxs.resize_and_reset(col_idx_vec.size());
    std::copy(value_vec.begin(), value_vec.end(), values.get_data());
    std::copy(col_idx_vec.begin(), col_idx_vec.end(), col_idxs.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_csr_to_csc(size_type num_rows, const IndexType* row_ptrs,
                               const IndexType* col_idxs,
                               const ValueType* csr_vals, IndexType* row_idxs,
                               IndexType* col_ptrs, ValueType* csc_vals,
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
                             matrix::Csr<ValueType, IndexType>* trans,
                             const matrix::Csr<ValueType, IndexType>* orig,
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

    components::fill_array(exec, trans_row_ptrs, orig_num_cols + 1,
                           IndexType{});
    for (size_type i = 0; i < orig_nnz; i++) {
        trans_row_ptrs[orig_col_idxs[i] + 1]++;
    }
    components::prefix_sum_nonnegative(exec, trans_row_ptrs + 1, orig_num_cols);

    convert_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs, orig_vals,
                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* orig,
               matrix::Csr<ValueType, IndexType>* trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* orig,
                    matrix::Csr<ValueType, IndexType>* trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_span(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source, const span& row_span,
    const span& col_span, array<IndexType>* row_nnz)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
#pragma omp parallel for
    for (size_type row = row_span.begin; row < row_span.end; ++row) {
        row_nnz->get_data()[row - row_span.begin] = zero<IndexType>();
        for (auto nnz = row_ptrs[row]; nnz < row_ptrs[row + 1]; ++nnz) {
            if (col_idxs[nnz] >= col_span.begin &&
                col_idxs[nnz] < col_span.end) {
                row_nnz->get_data()[row - row_span.begin]++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set, IndexType* row_nnz)
{
    auto num_row_subsets = row_index_set.get_num_subsets();
    auto num_col_subsets = col_index_set.get_num_subsets();
    auto row_superset_indices = row_index_set.get_superset_indices();
    auto row_subset_begin = row_index_set.get_subsets_begin();
    auto row_subset_end = row_index_set.get_subsets_end();
    auto col_subset_begin = col_index_set.get_subsets_begin();
    auto col_subset_end = col_index_set.get_subsets_end();
    auto src_ptrs = source->get_const_row_ptrs();

#pragma omp parallel for
    for (size_type set = 0; set < num_row_subsets; ++set) {
        size_type res_row = row_superset_indices[set];
        for (auto row = row_subset_begin[set]; row < row_subset_end[set];
             ++row) {
            row_nnz[res_row] = zero<IndexType>();
            for (size_type i = src_ptrs[row]; i < src_ptrs[row + 1]; ++i) {
                auto index = source->get_const_col_idxs()[i];
                if (index >= col_index_set.get_size()) {
                    continue;
                }
                const auto bucket = std::distance(
                    col_subset_begin,
                    std::upper_bound(col_subset_begin,
                                     col_subset_begin + num_col_subsets,
                                     index));
                auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
                if (index < col_subset_end[shifted_bucket] &&
                    index >= col_subset_begin[shifted_bucket]) {
                    row_nnz[res_row]++;
                }
            }
            res_row++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       gko::span row_span, gko::span col_span,
                       matrix::Csr<ValueType, IndexType>* result)
{
    auto row_offset = row_span.begin;
    auto col_offset = col_span.begin;
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto values = source->get_const_values();
    auto res_row_ptrs = result->get_row_ptrs();
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        size_type res_nnz = res_row_ptrs[row];
        for (auto nnz = row_ptrs[row_offset + row];
             nnz < row_ptrs[row_offset + row + 1]; ++nnz) {
            const auto local_col = col_idxs[nnz] - col_offset;
            if (local_col >= 0 && local_col < num_cols) {
                result->get_col_idxs()[res_nnz] = local_col;
                result->get_values()[res_nnz] = values[nnz];
                res_nnz++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix_from_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_row_subsets = row_index_set.get_num_subsets();
    auto row_subset_begin = row_index_set.get_subsets_begin();
    auto row_subset_end = row_index_set.get_subsets_end();
    auto row_superset_indices = row_index_set.get_superset_indices();
    auto res_row_ptrs = result->get_row_ptrs();
    auto res_col_idxs = result->get_col_idxs();
    auto res_values = result->get_values();
    auto num_col_subsets = col_index_set.get_num_subsets();
    auto col_subset_begin = col_index_set.get_subsets_begin();
    auto col_subset_end = col_index_set.get_subsets_end();
    auto col_superset_indices = col_index_set.get_superset_indices();
    const auto src_ptrs = source->get_const_row_ptrs();
    const auto src_col_idxs = source->get_const_col_idxs();
    const auto src_values = source->get_const_values();

#pragma unroll
    for (size_type set = 0; set < num_row_subsets; ++set) {
        for (auto row = row_subset_begin[set]; row < row_subset_end[set];
             ++row) {
            auto local_row =
                row - row_subset_begin[set] + row_superset_indices[set];
            auto res_nnz = res_row_ptrs[local_row];
            for (size_type i = src_ptrs[row]; i < src_ptrs[row + 1]; ++i) {
                auto index = src_col_idxs[i];
                if (index >= col_index_set.get_size()) {
                    continue;
                }
                const auto bucket = std::distance(
                    col_subset_begin,
                    std::upper_bound(col_subset_begin,
                                     col_subset_begin + num_col_subsets,
                                     index));
                auto shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
                if (index < col_subset_end[shifted_bucket] &&
                    (index >= col_subset_begin[shifted_bucket])) {
                    res_col_idxs[res_nnz] =
                        index - col_subset_begin[shifted_bucket] +
                        col_superset_indices[shifted_bucket];
                    res_values[res_nnz] = src_values[i];
                    res_nnz++;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* perm,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Csr<ValueType, IndexType>* permuted)
{
    inv_nonsymm_permute(exec, perm, perm, orig, permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,
                         const IndexType* row_perm,
                         const IndexType* column_perm,
                         const matrix::Csr<ValueType, IndexType>* orig,
                         matrix::Csr<ValueType, IndexType>* permuted)
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
        auto dst_row = row_perm[row];
        p_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum_nonnegative(exec, p_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = row_perm[row];
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = p_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        for (IndexType i = 0; i < row_size; ++i) {
            p_col_idxs[dst_begin + i] = column_perm[in_col_idxs[src_begin + i]];
            p_vals[dst_begin + i] = in_vals[src_begin + i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const OmpExecutor> exec, const IndexType* perm,
                 const matrix::Csr<ValueType, IndexType>* orig,
                 matrix::Csr<ValueType, IndexType>* row_permuted)
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
    components::prefix_sum_nonnegative(exec, rp_row_ptrs, num_rows + 1);
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
void inv_row_permute(std::shared_ptr<const OmpExecutor> exec,
                     const IndexType* perm,
                     const matrix::Csr<ValueType, IndexType>* orig,
                     matrix::Csr<ValueType, IndexType>* row_permuted)
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
    components::prefix_sum_nonnegative(exec, rp_row_ptrs, num_rows + 1);
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
    GKO_DECLARE_CSR_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            const matrix::Csr<ValueType, IndexType>* orig,
                            matrix::Csr<ValueType, IndexType>* permuted)
{
    inv_nonsymm_scale_permute(exec, scale, perm, scale, perm, orig, permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               const matrix::Csr<ValueType, IndexType>* orig,
                               matrix::Csr<ValueType, IndexType>* permuted)
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
        auto dst_row = row_perm[row];
        p_row_ptrs[dst_row] = in_row_ptrs[src_row + 1] - in_row_ptrs[src_row];
    }
    components::prefix_sum_nonnegative(exec, p_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = row_perm[row];
        auto src_begin = in_row_ptrs[src_row];
        auto dst_begin = p_row_ptrs[dst_row];
        auto row_size = in_row_ptrs[src_row + 1] - src_begin;
        for (IndexType i = 0; i < row_size; ++i) {
            const auto dst_col = col_perm[in_col_idxs[src_begin + i]];
            p_col_idxs[dst_begin + i] = dst_col;
            p_vals[dst_begin + i] = in_vals[src_begin + i] /
                                    (row_scale[dst_row] * col_scale[dst_col]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const OmpExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       const matrix::Csr<ValueType, IndexType>* orig,
                       matrix::Csr<ValueType, IndexType>* row_permuted)
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
    components::prefix_sum_nonnegative(exec, rp_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = perm[row];
        auto dst_row = row;
        auto src_begin = orig_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = orig_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(orig_col_idxs + src_begin, row_size,
                    rp_col_idxs + dst_begin);
        for (IndexType i = 0; i < row_size; i++) {
            rp_vals[i + dst_begin] = orig_vals[i + src_begin] * scale[src_row];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const OmpExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           matrix::Csr<ValueType, IndexType>* row_permuted)
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
    components::prefix_sum_nonnegative(exec, rp_row_ptrs, num_rows + 1);
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto src_row = row;
        auto dst_row = perm[row];
        auto src_begin = orig_row_ptrs[src_row];
        auto dst_begin = rp_row_ptrs[dst_row];
        auto row_size = orig_row_ptrs[src_row + 1] - src_begin;
        std::copy_n(orig_col_idxs + src_begin, row_size,
                    rp_col_idxs + dst_begin);
        for (IndexType i = 0; i < row_size; i++) {
            rp_vals[i + dst_begin] = orig_vals[i + src_begin] / scale[dst_row];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::Csr<ValueType, IndexType>* to_sort)
{
    auto values = to_sort->get_values();
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
#pragma omp parallel for
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        auto it = detail::make_zip_iterator(col_idxs + start_row_idx,
                                            values + start_row_idx);
        std::sort(it, it + row_nnz, [](auto t1, auto t2) {
            return std::get<0>(t1) < std::get<0>(t2);
        });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* to_check, bool* is_sorted)
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
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
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


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const mtx, bool& has_all_diags)
{
    bool l_has_all_diags = true;
    const size_type minsize = std::min(mtx->get_size()[0], mtx->get_size()[1]);
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
#pragma omp parallel for reduction(&& : l_has_all_diags)
    for (size_type row = 0; row < minsize; row++) {
        bool row_diag = false;
        for (IndexType iz = row_ptrs[row]; iz < row_ptrs[row + 1]; iz++) {
            if (col_idxs[iz] == row) {
                row_diag = true;
            }
        }
        if (!row_diag) {
            l_has_all_diags = false;
        }
    }
    has_all_diags = l_has_all_diags;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const OmpExecutor> exec,
                         const matrix::Dense<ValueType>* const alpha,
                         const matrix::Dense<ValueType>* const beta,
                         matrix::Csr<ValueType, IndexType>* const mtx)
{
    const auto nrows = static_cast<IndexType>(mtx->get_size()[0]);
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto vals = mtx->get_values();
    const auto beta_val = beta->get_const_values()[0];
    const auto alpha_val = alpha->get_const_values()[0];
#pragma omp parallel for
    for (IndexType row = 0; row < nrows; row++) {
        for (IndexType iz = row_ptrs[row]; iz < row_ptrs[row + 1]; iz++) {
            if (beta_val != one<ValueType>()) {
                vals[iz] *= beta_val;
            }
            if (row == mtx->get_const_col_idxs()[iz] &&
                alpha_val != zero<ValueType>()) {
                vals[iz] += alpha_val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL);


template <typename IndexType>
bool csr_lookup_try_full(IndexType row_len, IndexType col_range,
                         matrix::csr::sparsity_type allowed, int64& row_desc)
{
    using matrix::csr::sparsity_type;
    bool is_allowed = csr_lookup_allowed(allowed, sparsity_type::full);
    if (is_allowed && row_len == col_range) {
        row_desc = static_cast<int64>(sparsity_type::full);
        return true;
    }
    return false;
}


template <typename IndexType>
bool csr_lookup_try_bitmap(IndexType row_len, IndexType col_range,
                           IndexType min_col, IndexType available_storage,
                           matrix::csr::sparsity_type allowed, int64& row_desc,
                           int32* local_storage, const IndexType* cols)
{
    using matrix::csr::sparsity_bitmap_block_size;
    using matrix::csr::sparsity_type;
    bool is_allowed = csr_lookup_allowed(allowed, sparsity_type::bitmap);
    const auto num_blocks = ceildiv(col_range, sparsity_bitmap_block_size);
    if (is_allowed && num_blocks * 2 <= available_storage) {
        row_desc = (static_cast<int64>(num_blocks) << 32) |
                   static_cast<int64>(sparsity_type::bitmap);
        const auto block_ranks = local_storage;
        const auto block_bitmaps =
            reinterpret_cast<uint32*>(block_ranks + num_blocks);
        std::fill_n(block_bitmaps, num_blocks, 0);
        for (auto col_it = cols; col_it < cols + row_len; col_it++) {
            const auto rel_col = *col_it - min_col;
            const auto block = rel_col / sparsity_bitmap_block_size;
            const auto col_in_block = rel_col % sparsity_bitmap_block_size;
            block_bitmaps[block] |= uint32{1} << col_in_block;
        }
        int32 partial_sum{};
        for (int32 block = 0; block < num_blocks; block++) {
            block_ranks[block] = partial_sum;
            partial_sum += gko::detail::popcount(block_bitmaps[block]);
        }
        return true;
    }
    return false;
}


template <typename IndexType>
void csr_lookup_build_hash(IndexType row_len, IndexType available_storage,
                           int64& row_desc, int32* local_storage,
                           const IndexType* cols)
{
    // we need at least one unfilled entry to avoid infinite loops on search
    GKO_ASSERT(row_len < available_storage);
    constexpr double inv_golden_ratio = 0.61803398875;
    // use golden ratio as approximation for hash parameter that spreads
    // consecutive values as far apart as possible. Ensure lowest bit is set
    // otherwise we skip odd hashtable entries
    const auto hash_parameter =
        1u | static_cast<uint32>(available_storage * inv_golden_ratio);
    row_desc = (static_cast<int64>(hash_parameter) << 32) |
               static_cast<int>(matrix::csr::sparsity_type::hash);
    std::fill_n(local_storage, available_storage, invalid_index<int32>());
    for (int32 nz = 0; nz < row_len; nz++) {
        auto hash = (static_cast<typename std::make_unsigned<IndexType>::type>(
                         cols[nz]) *
                     hash_parameter) %
                    static_cast<uint32>(available_storage);
        // linear probing: find the next empty entry
        while (local_storage[hash] != invalid_index<int32>()) {
            hash++;
            if (hash >= available_storage) {
                hash = 0;
            }
        }
        local_storage[hash] = nz;
    }
}


template <typename IndexType>
void build_lookup(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  size_type num_rows, matrix::csr::sparsity_type allowed,
                  const IndexType* storage_offsets, int64* row_desc,
                  int32* storage)
{
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_len = row_ptrs[row + 1] - row_begin;
        const auto storage_begin = storage_offsets[row];
        const auto available_storage = storage_offsets[row + 1] - storage_begin;
        const auto local_storage = storage + storage_begin;
        const auto local_cols = col_idxs + row_begin;
        const auto min_col = row_len > 0 ? local_cols[0] : 0;
        const auto col_range =
            row_len > 0 ? local_cols[row_len - 1] - min_col + 1 : 0;
        bool done =
            csr_lookup_try_full(row_len, col_range, allowed, row_desc[row]);
        if (!done) {
            done = csr_lookup_try_bitmap(
                row_len, col_range, min_col, available_storage, allowed,
                row_desc[row], local_storage, local_cols);
        }
        if (!done) {
            csr_lookup_build_hash(row_len, available_storage, row_desc[row],
                                  local_storage, local_cols);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CSR_BUILD_LOOKUP_KERNEL);


}  // namespace csr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
