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

#include "core/matrix/sliced_ell.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/sliced_ell_kernels.hpp"
#include "core/matrix/dense.hpp"
#include <iostream>

namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, sliced_ell::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, sliced_ell::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense, sliced_ell::convert_to_dense<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_dense, sliced_ell::move_to_dense<TplArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::copy_from(const LinOp *other)
{
    as<ConvertibleTo<Sliced_ell<ValueType, IndexType>>>(other)->convert_to(this);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    as<ConvertibleTo<Sliced_ell<ValueType, IndexType>>>(other.get())->move_to(this);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::apply(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Sliced_ell<ValueType, IndexType>::clone_type() const
{
    return std::unique_ptr<LinOp>(
        new Sliced_ell(this->get_executor(), this->get_num_rows(),
                       this->get_num_cols(), this->get_num_stored_elements(),
                       this->get_slice_size(), this->get_padding_factor(),
                       this->get_total_cols()));
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::clear()
{
    this->set_dimensions(0, 0, 0);
    values_.clear();
    col_idxs_.clear();
    slice_lens_.clear();
    slice_sets_.clear();
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::convert_to(Sliced_ell *other) const
{
    other->set_dimensions(this);
    other->values_ = values_;
    other->col_idxs_ = col_idxs_;
    other->slice_lens_ = std::move(slice_lens_);
    other->slice_sets_ = std::move(slice_sets_);
    other->slice_size_ = std::move(slice_size_);
    other->padding_factor_ = std::move(padding_factor_);
    other->total_cols_ = std::move(total_cols_);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::move_to(Sliced_ell *other)
{
    other->set_dimensions(this);
    other->values_ = std::move(values_);
    other->col_idxs_ = std::move(col_idxs_);
    other->slice_lens_ = std::move(slice_lens_);
    other->slice_sets_ = std::move(slice_sets_);
    other->slice_size_ = std::move(slice_size_);
    other->padding_factor_ = std::move(padding_factor_);
    other->total_cols_ = std::move(total_cols_);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(
        exec, this->get_num_rows(), this->get_num_cols(), this->get_num_cols());
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(
        exec, this->get_num_rows(), this->get_num_cols(), this->get_num_cols());
    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_move_to_dense_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Sliced_ell<ValueType, IndexType>::read_from_mtx( \
                                       const std::string &filename)
{
    // Define variables
    auto data = read_raw_from_mtx<ValueType, IndexType>(filename);
    size_type nnz = 0;
    std::vector<index_type> nnz_row(data.num_rows, 0);
    auto slice_size = this->get_slice_size();
    // Make sure that slice_size is not zero
    slice_size = (slice_size == 0)?default_slice_size:slice_size;
    auto padding_factor = this->get_padding_factor();
    // Make sure that padding factor is not zero
    padding_factor = (padding_factor == 0)?default_padding_factor:padding_factor;
    index_type slice_num = \
        static_cast<index_type>((data.num_rows+slice_size-1) / slice_size);
    std::vector<index_type> slice_cols(slice_num, 0);

    // Count number of nonzeros in every row
    for (const auto &elem : data.nonzeros) {
        nnz += (std::get<2>(elem) != zero<ValueType>());
        nnz_row.at(std::get<0>(elem))++;
    }

    // Find longest column for each slice
    for (size_type row = 0; row < data.num_rows; row++) {
        index_type slice_id = static_cast<index_type>(row / slice_size);
        slice_cols[slice_id] = std::max(slice_cols[slice_id], nnz_row[row]);
    }

    // Find total column length
    size_type total_cols = 0;
    for (size_type slice = 0; slice < slice_num; slice++) {
        total_cols += slice_cols[slice];
    }
    total_cols = ceildiv(total_cols, padding_factor) * padding_factor;

    auto tmp = create(this->get_executor()->get_master(), data.num_rows,
                      data.num_cols, nnz, slice_size, padding_factor, total_cols);

    // Setup slice_lens and slice_sets
    index_type start_col = 0;
    for (index_type slice = 0; slice < slice_num; slice++) {
        tmp->get_slice_lens()[slice] = \
            padding_factor * ceildiv(slice_cols[slice], padding_factor);
        tmp->get_slice_sets()[slice] = start_col;
        start_col += tmp->get_slice_lens()[slice];
    }

    // Get values and column idxs
    size_type ind = 0;
    int n = data.nonzeros.size();
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type col = 0;
            for (; ind < n; ind++) {
                if (std::get<0>(data.nonzeros[ind]) > slice * slice_size + row) {
                    break;
                }
                auto val = std::get<2>(data.nonzeros[ind]);
                auto sliced_ell_ind = row + (tmp->get_slice_sets()[slice] + col) * slice_size;
                if (val != zero<ValueType>()) {
                    tmp->get_values()[sliced_ell_ind] = val;
                    tmp->get_col_idxs()[sliced_ell_ind] = std::get<1>(data.nonzeros[ind]);
                    col++;
                }
            }
            for (auto i = col; i < tmp->get_slice_lens()[slice]; i++) {
                auto sliced_ell_ind = row + (tmp->get_slice_sets()[slice] + i) * slice_size;
                tmp->get_values()[sliced_ell_ind] = 0;
                tmp->get_col_idxs()[sliced_ell_ind] = tmp->get_col_idxs()[sliced_ell_ind-slice_size];
            }
        }
    }

    tmp->move_to(this);
}


#define DECLARE_SLICED_ELL_MATRIX(ValueType, IndexType) class Sliced_ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SLICED_ELL_MATRIX);
#undef DECLARE_SLICED_ELL_MATRIX


}  // namespace matrix
}  // namespace gko
