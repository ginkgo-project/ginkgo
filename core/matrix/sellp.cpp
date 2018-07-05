/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/sellp.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/sellp_kernels.hpp"


#include <vector>


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, sellp::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, sellp::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense,
                           sellp::convert_to_dense<TplArgs...>);
};


template <typename ValueType, typename IndexType>
size_type calculate_total_cols(const matrix_data<ValueType, IndexType> &data,
                               const size_type slice_size,
                               const size_type stride_factor,
                               std::vector<size_type> &slice_lengths)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    IndexType current_slice = 0;
    size_type total_cols = 0;
    for (const auto &elem : data.nonzeros) {
        if (elem.row / slice_size != current_slice) {
            slice_lengths[current_slice] =
                stride_factor *
                ceildiv(slice_lengths[current_slice], stride_factor);
            total_cols += slice_lengths[current_slice];
            current_slice = elem.row / slice_size;
        }
        if (elem.row != current_row) {
            current_row = elem.row;
            slice_lengths[current_slice] =
                max(slice_lengths[current_slice], nnz);
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    slice_lengths[current_slice] = max(slice_lengths[current_slice], nnz);
    slice_lengths[current_slice] =
        stride_factor * ceildiv(slice_lengths[current_slice], stride_factor);
    total_cols += slice_lengths[current_slice];
    return total_cols;
}


}  // namespace


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                             const LinOp *beta, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::read(const mat_data &data)
{
    // Make sure that slice_size and stride factor are not zero.
    auto slice_size = (this->get_slice_size() == 0) ? default_slice_size
                                                    : this->get_slice_size();
    auto stride_factor = (this->get_stride_factor() == 0)
                             ? default_stride_factor
                             : this->get_stride_factor();

    // Allocate space for slice_cols.
    size_type slice_num = static_cast<index_type>(
        (data.size.num_rows + slice_size - 1) / slice_size);
    std::vector<size_type> slice_lengths(slice_num, 0);

    // Get the number of maximum columns for every slice.
    auto total_cols =
        calculate_total_cols(data, slice_size, stride_factor, slice_lengths);

    // Create an SELL-P format matrix based on the sizes.
    auto tmp = Sellp::create(this->get_executor()->get_master(), data.size,
                             slice_size, stride_factor, total_cols);

    // Get slice length, slice set, matrix values and column indexes.
    index_type slice_set = 0;
    size_type ind = 0;
    auto n = data.nonzeros.size();
    for (size_type slice = 0; slice < slice_num; slice++) {
        tmp->get_slice_lengths()[slice] = slice_lengths[slice];
        tmp->get_slice_sets()[slice] = slice_set;
        slice_set += tmp->get_slice_lengths()[slice];
        for (size_type row_in_slice = 0; row_in_slice < slice_size;
             row_in_slice++) {
            size_type col = 0;
            size_type row = slice * slice_size + row_in_slice;
            while (ind < n && data.nonzeros[ind].row == row) {
                auto val = data.nonzeros[ind].value;
                auto sellp_ind =
                    (tmp->get_slice_sets()[slice] + col) * slice_size + row;
                if (val != zero<ValueType>()) {
                    tmp->get_values()[sellp_ind] = val;
                    tmp->get_col_idxs()[sellp_ind] = data.nonzeros[ind].column;
                    col++;
                }
                ind++;
            }
            for (auto i = col; i < tmp->get_slice_lengths()[slice]; i++) {
                auto sellp_ind =
                    (tmp->get_slice_sets()[slice] + i) * slice_size + row;
                tmp->get_values()[sellp_ind] = zero<ValueType>();
                tmp->get_col_idxs()[sellp_ind] = 0;
            }
        }
    }
    tmp->get_slice_sets()[slice_num] = slice_set;

    // Return the matrix.
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Sellp *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Sellp *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    auto slice_size = tmp->get_slice_size();
    size_type slice_num = static_cast<index_type>(
        (tmp->get_size().num_rows + slice_size - 1) / slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row_in_slice = 0; row_in_slice < slice_size;
             row_in_slice++) {
            auto row = slice * slice_size + row_in_slice;
            for (size_type i = 0; i < tmp->get_const_slice_lengths()[slice];
                 i++) {
                const auto val = tmp->val_at(
                    row_in_slice, tmp->get_const_slice_sets()[slice], i);
                if (val != zero<ValueType>() &&
                    row < tmp->get_size().num_rows) {
                    const auto col = tmp->col_at(
                        row_in_slice, tmp->get_const_slice_sets()[slice], i);
                    data.nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }
}


#define DECLARE_SELLP_MATRIX(ValueType, IndexType) \
    class Sellp<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SELLP_MATRIX);
#undef DECLARE_SELLP_MATRIX


}  // namespace matrix
}  // namespace gko