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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Csr
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

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
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
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


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const ReferenceExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Csr<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Csr<ValueType, IndexType> *source)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto vals = source->get_const_values();

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
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::Sellp<ValueType, IndexType> *result,
                      const matrix::Csr<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = (result->get_slice_size() == 0)
                          ? matrix::default_slice_size
                          : result->get_slice_size();
    auto stride_factor = (result->get_stride_factor() == 0)
                             ? matrix::default_stride_factor
                             : result->get_stride_factor();

    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_values = source->get_const_values();

    int slice_num = ceildiv(num_rows, slice_size);
    slice_sets[0] = 0;
    for (size_type slice = 0; slice < slice_num; slice++) {
        if (slice > 0) {
            slice_sets[slice] =
                slice_sets[slice - 1] + slice_lengths[slice - 1];
        }
        slice_lengths[slice] = 0;
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            slice_lengths[slice] =
                (slice_lengths[slice] >
                 source_row_ptrs[global_row + 1] - source_row_ptrs[global_row])
                    ? slice_lengths[slice]
                    : source_row_ptrs[global_row + 1] -
                          source_row_ptrs[global_row];
        }
        slice_lengths[slice] =
            stride_factor * ceildiv(slice_lengths[slice], stride_factor);
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            size_type sellp_ind = slice_sets[slice] * slice_size + row;
            for (size_type csr_ind = source_row_ptrs[global_row];
                 csr_ind < source_row_ptrs[global_row + 1]; csr_ind++) {
                vals[sellp_ind] = source_values[csr_ind];
                col_idxs[sellp_ind] = source_col_idxs[csr_ind];
                sellp_ind += slice_size;
            }
            for (size_type i = sellp_ind;
                 i <
                 (slice_sets[slice] + slice_lengths[slice]) * slice_size + row;
                 i += slice_size) {
                col_idxs[i] = 0;
                vals[i] = zero<ValueType>();
            }
        }
    }
    slice_sets[slice_num] =
        slice_sets[slice_num - 1] + slice_lengths[slice_num - 1];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    size_type total_cols = 0;
    const auto num_rows = source->get_size()[0];
    const auto slice_num = ceildiv(num_rows, slice_size);

    const auto row_ptrs = source->get_const_row_ptrs();

    for (size_type slice = 0; slice < slice_num; slice++) {
        IndexType max_nnz_per_row_in_this_slice = 0;
        for (size_type row = 0;
             row < slice_size && row + slice * slice_size < num_rows; row++) {
            size_type global_row = slice * slice_size + row;
            max_nnz_per_row_in_this_slice =
                max(row_ptrs[global_row + 1] - row_ptrs[global_row],
                    max_nnz_per_row_in_this_slice);
        }
        total_cols += ceildiv(max_nnz_per_row_in_this_slice, stride_factor) *
                      stride_factor;
    }

    *result = total_cols;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Ell<ValueType, IndexType> *result,
                    const matrix::Csr<ValueType, IndexType> *source)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto vals = source->get_const_values();
    const auto col_idxs = source->get_const_col_idxs();
    const auto row_ptrs = source->get_const_row_ptrs();

    const auto num_stored_elements_per_row =
        result->get_num_stored_elements_per_row();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            result->val_at(row, i) = zero<ValueType>();
            result->col_at(row, i) = 0;
        }
        for (size_type col_idx = 0; col_idx < row_ptrs[row + 1] - row_ptrs[row];
             col_idx++) {
            result->val_at(row, col_idx) = vals[row_ptrs[row] + col_idx];
            result->col_at(row, col_idx) = col_idxs[row_ptrs[row] + col_idx];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename IndexType, typename ValueType, typename UnaryOperator>
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
void transpose_and_transform(std::shared_ptr<const ReferenceExecutor> exec,
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
    convert_idxs_to_ptrs(orig_col_idxs, orig_nnz, trans_row_ptrs + 1,
                         orig_num_cols);

    convert_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs, orig_vals,
                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::Csr<ValueType, IndexType> *trans,
               const matrix::Csr<ValueType, IndexType> *orig)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *trans,
                    const matrix::Csr<ValueType, IndexType> *orig)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];
    const auto row_ptrs = source->get_const_row_ptrs();
    IndexType max_nnz = 0;

    for (size_type i = 0; i < num_rows; i++) {
        max_nnz = max(row_ptrs[i + 1] - row_ptrs[i], max_nnz);
    }

    *result = max_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       matrix::Hybrid<ValueType, IndexType> *result,
                       const matrix::Csr<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_lim = strategy->get_coo_nnz();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();

    // Initial Hybrid Matrix
    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
         i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = 0;
        }
    }
    for (size_type i = 0; i < result->get_coo_num_stored_elements(); i++) {
        coo_val[i] = zero<ValueType>();
        coo_col[i] = 0;
        coo_row[i] = 0;
    }

    const auto csr_row_ptrs = source->get_const_row_ptrs();
    const auto csr_vals = source->get_const_values();
    size_type csr_idx = 0;
    size_type coo_idx = 0;
    for (IndexType row = 0; row < num_rows; row++) {
        size_type ell_idx = 0;
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
void calculate_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    auto row_nnz_val = result->get_data();
    for (size_type i = 0; i < result->get_num_elems(); i++) {
        row_nnz_val[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


// TODO move to own file
/**
 * @internal
 * @brief This class is used to sort two distinct, arrays of the same size
 * according to the values of one.
 *
 * Stores the pointers of two arrays, one storing the type `ValueType`, and one
 * storing the type `IndexType`. This class also provides an iterator class,
 * which can be used for `std::sort`. Without a custom iterator, memory copies
 * would be necessary to create, for example, one array of `std::pairs` to use
 * `std::sort`, or a self written sort function.
 */
template <typename ValueType, typename IndexType>
class CustomSortHelper {
public:
    /**
     * Helper struct, needed for the default construction and assignment inside
     * `std::sort` through CustomReference
     */
    struct custom_element {
        ValueType value;
        IndexType index;
    };

    /**
     * This class is used as a reference to a sorting target, which abstracts
     * the existence of two distinct arrays.
     * It meets the requirements of `MoveAssignable`, `MoveConstructible`
     * In all comparisons, only the values of `indices_` matter, while the
     * corresponding value will always be copied / moved / swapped to the
     * same place.
     */
    class CustomReference {
    public:
        using array_index_type = int64;

        CustomReference() = delete;
        CustomReference(CustomSortHelper &parent, array_index_type array_index)
            : parent_(parent), arr_index_(array_index)
        {}
        // Since it must be `MoveConstructible`
        CustomReference(CustomReference &&other)
            : parent_(other.parent_), arr_index_(std::move(other.arr_index_))
        {}
        CustomReference(const CustomReference &other)
            : parent_(other.parent_), arr_index_(other.index)
        {}

        CustomReference &operator=(custom_element other)
        {
            value() = other.value;
            index() = other.index;
            return *this;
        }
        CustomReference &operator=(const CustomReference &other)
        {
            value() = other.value();
            index() = other.index();
            return *this;
        }
        // Since it must be `MoveAssignable`
        CustomReference &operator=(CustomReference &&other)
        {
            parent_.values_[arr_index_] =
                std::move(other.parent_.values_[other.arr_index_]);
            parent_.indices_[arr_index_] =
                std::move(other.parent_.indices_[other.arr_index_]);
            return *this;
        }
        // Conversion operator to `custom_element`
        operator custom_element() const
        {
            return custom_element{value(), index()};
        }
        friend void swap(CustomReference a, CustomReference b)
        {
            std::swap(a.value(), b.value());
            std::swap(a.index(), b.index());
        }
        friend bool operator<(const CustomReference &left,
                              const CustomReference &right)
        {
            return left.index() < right.index();
        }
        friend bool operator<(const CustomReference &left,
                              const custom_element &right)
        {
            return left.index() < right.index;
        }
        friend bool operator<(const custom_element &left,
                              const CustomReference &right)
        {
            return left.index < right.index();
        }

        ValueType &value() { return parent_.values_[arr_index_]; }
        ValueType value() const { return parent_.values_[arr_index_]; }
        IndexType &index() { return parent_.indices_[arr_index_]; }
        IndexType index() const { return parent_.indices_[arr_index_]; }

    private:
        CustomSortHelper &parent_;
        array_index_type arr_index_;
    };
    /**
     * The iterator that can be used for `std::sort`. It meets the requirements
     * of `LegacyRandomAccessIterator` and `ValueSwappable`.
     * For performance reasons, it is expected that all iterators that are
     * compared / used with each other have the same `parent`.
     * This class encapsuls the array index to both arrays.
     */
    class CustomIterator {
    public:
        // Needed to count as a `LegacyRandomAccessIterator`
        using difference_type = typename CustomReference::array_index_type;
        using value_type = custom_element;
        using pointer = CustomReference;
        using reference = CustomReference;
        using iterator_category = std::random_access_iterator_tag;

        CustomIterator(CustomSortHelper &parent, difference_type array_index)
            : parent_(parent), arr_index_(array_index)
        {}

        CustomIterator(const CustomIterator &other)
            : parent_(other.parent_), arr_index_(other.arr_index_)
        {}

        CustomIterator &operator=(const CustomIterator &other)
        {
            arr_index_ = other.arr_index_;
            return *this;
        }

        // Operators needed for std::sort requirement of
        // `LegacyRandomAccessIterator`
        CustomIterator &operator+=(difference_type i)
        {
            arr_index_ += i;
            return *this;
        }
        CustomIterator &operator-=(difference_type i)
        {
            arr_index_ -= i;
            return *this;
        }
        CustomIterator &operator++()
        {  // Prefix increment (++i)
            ++arr_index_;
            return *this;
        }
        CustomIterator operator++(int)
        {  // Postfix increment (i++)
            CustomIterator temp(*this);
            ++arr_index_;
            return temp;
        }
        CustomIterator &operator--()
        {  // Prefix decrement (--i)
            --arr_index_;
            return *this;
        }
        CustomIterator operator--(int)
        {  // Postfix decrement (i--)
            CustomIterator temp(*this);
            --arr_index_;
            return temp;
        }
        CustomIterator operator+(difference_type i) const
        {
            return CustomIterator{parent_, arr_index_ + i};
        }
        friend CustomIterator operator+(difference_type i,
                                        const CustomIterator &iter)
        {
            return CustomIterator{iter.parent_, iter.arr_index_ + i};
        }
        CustomIterator operator-(difference_type i) const
        {
            return CustomIterator{parent_, arr_index_ - i};
        }
        difference_type operator-(const CustomIterator &other) const
        {
            return arr_index_ - other.arr_index_;
        }

        CustomReference operator*() const
        {
            return CustomReference(parent_, arr_index_);
        }

        // Comparable operators
        bool operator==(const CustomIterator &other)
        {
            return arr_index_ == other.arr_index_;
        }
        bool operator!=(const CustomIterator &other)
        {
            return arr_index_ != other.arr_index_;
        }
        bool operator<(const CustomIterator &other) const
        {
            return arr_index_ < other.arr_index_;
        }
        bool operator<=(const CustomIterator &other) const
        {
            return arr_index_ <= other.arr_index_;
        }
        bool operator>(const CustomIterator &other) const
        {
            return arr_index_ > other.arr_index_;
        }
        bool operator>=(const CustomIterator &other) const
        {
            return arr_index_ >= other.arr_index_;
        }

    private:
        CustomSortHelper &parent_;
        difference_type arr_index_;
    };

public:
    CustomSortHelper(ValueType *values, IndexType *indices, size_type size)
        : values_(values), indices_(indices), size_(size)
    {}
    /**
     * Creates an iterator pointing to the beginning of both arrays
     * @returns  an iterator pointing to the beginning of both arrays
     */
    CustomIterator begin() { return CustomIterator(*this, 0); }
    /**
     * Creates an iterator pointing to the (excluding) end of both arrays
     * @returns  an iterator pointing to the (excluding) end of both arrays
     */
    CustomIterator end() { return CustomIterator(*this, size_); }

public:
    ValueType *values_;
    IndexType *indices_;
    size_type size_;
};


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
{
    auto values = to_sort->get_values();
    auto row_ptrs = to_sort->get_row_ptrs();
    auto cols = to_sort->get_col_idxs();
    auto number_rows = to_sort->get_size()[0];
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        auto helper = CustomSortHelper<ValueType, IndexType>(
            values + start_row_idx, cols + start_row_idx, row_nnz);
        std::sort(helper.begin(), helper.end());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *to_sort,
                               bool *is_sorted)
{
    const auto rows = to_sort->get_const_row_ptrs();
    const auto cols = to_sort->get_const_col_idxs();
    const auto size = to_sort->get_size();
    for (size_type i = 0; i < size[0]; ++i) {
        for (auto idx = rows[i] + 1; idx < rows[i + 1]; ++idx) {
            if (cols[idx - 1] > cols[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
