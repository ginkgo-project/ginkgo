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

#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/base/batch_multi_vector_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_multi_vector {


GKO_REGISTER_OPERATION(scale, batch_multi_vector::scale);
GKO_REGISTER_OPERATION(add_scaled, batch_multi_vector::add_scaled);
GKO_REGISTER_OPERATION(compute_dot, batch_multi_vector::compute_dot);
GKO_REGISTER_OPERATION(compute_norm2, batch_multi_vector::compute_norm2);
GKO_REGISTER_OPERATION(copy, batch_multi_vector::copy);


}  // namespace batch_multi_vector


template <typename ValueType>
void BatchMultiVector<ValueType>::scale_impl(
    const BatchMultiVector<ValueType>* alpha)
{
    GKO_ASSERT_BATCH_EQUAL_ROWS(
        alpha, batch_dim<2>(this->get_num_batch_entries(), dim<2>(1, 1)));
    for (size_type b = 0; b < alpha->get_num_batch_entries(); ++b) {
        if (alpha->get_common_size()[1] != 1) {
            // different alpha for each column
            GKO_ASSERT_BATCH_EQUAL_COLS(this, alpha);
        }
    }
    this->get_executor()->run(batch_multi_vector::make_scale(alpha, this));
}


template <typename ValueType>
void BatchMultiVector<ValueType>::add_scaled_impl(
    const BatchMultiVector<ValueType>* alpha,
    const BatchMultiVector<ValueType>* b)
{
    GKO_ASSERT_BATCH_EQUAL_ROWS(
        alpha, batch_dim<2>(this->get_num_batch_entries(), dim<2>(1, 1)));
    for (size_type b = 0; b < alpha->get_num_batch_entries(); ++b) {
        if (alpha->get_common_size()[1] != 1) {
            // different alpha for each column
            GKO_ASSERT_BATCH_EQUAL_COLS(this, alpha);
        }
    }
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, b);

    this->get_executor()->run(
        batch_multi_vector::make_add_scaled(alpha, b, this));
}


inline const batch_dim<2> get_col_sizes(const batch_dim<2>& sizes)
{
    return batch_dim<2>(sizes.get_num_batch_entries(), dim<2>(1, sizes[1]));
}


template <typename ValueType>
void BatchMultiVector<ValueType>::compute_dot_impl(
    const BatchMultiVector<ValueType>* b,
    BatchMultiVector<ValueType>* result) const
{
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(batch_result,
                                      get_col_sizes(this->get_size()));
    this->get_executor()->run(
        batch_multi_vector::make_compute_dot(this, b, result));
}


template <typename ValueType>
void BatchMultiVector<ValueType>::compute_norm2_impl(
    BatchMultiVector<remove_complex<ValueType>>* result) const
{
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(result, get_col_sizes(this->get_size()));
    this->get_executor()->run(batch_multi_vector::make_compute_norm2(
        as<BatchMultiVector<ValueType>>(this), result));
}


template <typename ValueType>
void BatchMultiVector<ValueType>::convert_to(
    BatchMultiVector<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void BatchMultiVector<ValueType>::move_to(
    BatchMultiVector<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType* mtx, const std::vector<MatrixData>& data)
{
    auto batch_sizes = std::vector<dim<2>>(data.size());
    size_type ind = 0;
    for (const auto& b : data) {
        batch_sizes[ind] = b.size;
        ++ind;
    }
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(),
                                  batch_dim<2>(batch_sizes));
    for (size_type b = 0; b < data.size(); ++b) {
        size_type ind = 0;
        for (size_type row = 0; row < data[b].size[0]; ++row) {
            for (size_type col = 0; col < data[b].size[1]; ++col) {
                if (ind < data[b].nonzeros.size() &&
                    data[b].nonzeros[ind].row == row &&
                    data[b].nonzeros[ind].column == col) {
                    tmp->at(b, row, col) = data[b].nonzeros[ind].value;
                    ++ind;
                } else {
                    tmp->at(b, row, col) =
                        zero<typename MatrixType::value_type>();
                }
            }
        }
    }
    tmp->move_to(mtx);
}


template <typename ValueType>
void BatchMultiVector<ValueType>::read(const std::vector<mat_data>& data)
{
    read_impl(this, data);
}


template <typename ValueType>
void BatchMultiVector<ValueType>::read(const std::vector<mat_data32>& data)
{
    read_impl(this, data);
}


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType* mtx, std::vector<MatrixData>& data)
{
    std::unique_ptr<const BatchMultiVector<ValueType>> op{};
    const MatrixType* tmp{};
    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType*>(op.get());
    } else {
        tmp = mtx;
    }

    data = std::vector<MatrixData>(mtx->get_num_batch_entries());
    for (size_type b = 0; b < mtx->get_num_batch_entries(); ++b) {
        data[b] = {mtx->get_size().at(b), {}};
        for (size_type row = 0; row < data[b].size[0]; ++row) {
            for (size_type col = 0; col < data[b].size[1]; ++col) {
                if (tmp->at(b, row, col) !=
                    zero<typename MatrixType::value_type>()) {
                    data[b].nonzeros.emplace_back(row, col,
                                                  tmp->at(b, row, col));
                }
            }
        }
    }
}


template <typename ValueType>
void BatchMultiVector<ValueType>::write(std::vector<mat_data>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void BatchMultiVector<ValueType>::write(std::vector<mat_data32>& data) const
{
    write_impl(this, data);
}


#define GKO_DECLARE_BATCH_MULTI_VECTOR_MATRIX(_type) \
    class BatchMultiVector<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_MATRIX);


}  // namespace matrix


}  // namespace gko
