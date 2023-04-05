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

#include <ginkgo/core/matrix/batch_band.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_band_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_band {}  // namespace batch_band


template <typename ValueType>
void BatchBand<ValueType>::apply_impl(const BatchLinOp* b,
                                      BatchLinOp* x) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchBand<ValueType>::apply_impl(const BatchLinOp* alpha,
                                      const BatchLinOp* b,
                                      const BatchLinOp* beta,
                                      BatchLinOp* x) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchBand<ValueType>::convert_to(
    BatchBand<next_precision<ValueType>>* result) const
{
    result->KL_ = this->KL_;
    result->KU_ = this->KU_;
    result->band_array_col_major_ = this->band_array_col_major_;
    result->num_elems_per_batch_cumul_ = this->num_elems_per_batch_cumul_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void BatchBand<ValueType>::move_to(BatchBand<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void BatchBand<ValueType>::convert_to(
    BatchCsr<ValueType, int32>* const result) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchBand<ValueType>::move_to(BatchCsr<ValueType, int32>* const result)
    GKO_NOT_IMPLEMENTED;

template <typename ValueType>
void BatchBand<ValueType>::convert_to(BatchDense<ValueType>* const result) const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchBand<ValueType>::move_to(BatchDense<ValueType>* const result)
    GKO_NOT_IMPLEMENTED;


namespace {

template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType* mtx, const std::vector<MatrixData>& data,
                      const batch_stride& KLs, const batch_stride& KUs)
{
    using ValueType = typename MatrixType::value_type;
    auto batch_sizes = std::vector<dim<2>>(data.size());
    size_type ind = 0;
    for (const auto& b : data) {
        batch_sizes[ind] = b.size;
        ++ind;
    }

    auto tmp = MatrixType::create(mtx->get_executor()->get_master(),
                                  batch_dim<2>(batch_sizes), batch_stride(KLs),
                                  batch_stride(KUs));

    for (size_type batch_entry_idx = 0; batch_entry_idx < data.size();
         batch_entry_idx++) {
        assert(data[batch_entry_idx].size[0] == data[batch_entry_idx].size[1]);
        const auto size = data[batch_entry_idx].size[0];
        const auto KL = KLs.at(batch_entry_idx);
        const auto KU = KUs.at(batch_entry_idx);

        for (size_type i = 0; i < size * (2 * KL + KU + 1); i++) {
            tmp->get_band_array(batch_entry_idx)[i] = gko::nan<ValueType>();
        }

        // initialize the band region with zeroes
        // Think of band in the dense A layout and access the corresponding
        // elements in the band array
        for (size_type dense_col = 0; dense_col < size; dense_col++) {
            for (size_type dense_row =
                     std::max(int{0}, static_cast<int>(dense_col) -
                                          static_cast<int>(KU));
                 dense_row <= std::min(size - 1, dense_col + KL); dense_row++) {
                tmp->at_in_reference_to_dense_layout(
                    batch_entry_idx, dense_row, dense_col) = zero<ValueType>();
            }
        }

        size_type ind = 0;
        while (ind < data[batch_entry_idx].nonzeros.size()) {
            const auto row = data[batch_entry_idx].nonzeros[ind].row;
            const auto col = data[batch_entry_idx].nonzeros[ind].column;
            const auto val = data[batch_entry_idx].nonzeros[ind].value;

            tmp->at_in_reference_to_dense_layout(batch_entry_idx, row, col) =
                val;

            ++ind;
        }
    }

    tmp->move_to(mtx);
}


template <typename MatrixData>
std::pair<size_type, size_type> infer_KL_and_KU(const MatrixData& data)
{
    typename MatrixData::index_type current_row = 0;
    int kl_acc_to_curr_row = 0;
    int ku_acc_to_curr_row = 0;
    int KL = 0;
    int KU = 0;

    kl_acc_to_curr_row = data.nonzeros[0].row - data.nonzeros[0].column;
    KL = kl_acc_to_curr_row;
    const auto nnz = data.nonzeros.size();
    for (size_type i = 0; i < nnz; i++) {
        const auto& elem = data.nonzeros[i];

        if (elem.row != current_row) {
            // first element of the next row
            current_row = elem.row;
            kl_acc_to_curr_row = elem.row - elem.column;
            KL = std::max(kl_acc_to_curr_row, KL);
            // last element of the current row
            const auto& prev_ele = data.nonzeros[i - 1];
            ku_acc_to_curr_row = prev_ele.column - prev_ele.row;
            KU = std::max(ku_acc_to_curr_row, KU);
        }
    }

    ku_acc_to_curr_row =
        data.nonzeros[nnz - 1].column - data.nonzeros[nnz - 1].row;
    KU = std::max(ku_acc_to_curr_row, KU);

    return std::pair<size_type, size_type>(KL, KU);
}


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType* mtx, const std::vector<MatrixData>& data)
{
    auto KLs = std::vector<size_type>(data.size());
    auto KUs = std::vector<size_type>(data.size());

    for (size_type batch_entry_idx = 0; batch_entry_idx < data.size();
         batch_entry_idx++) {
        auto KL_KU_pair = infer_KL_and_KU(data[batch_entry_idx]);
        KLs[batch_entry_idx] = KL_KU_pair.first;
        KUs[batch_entry_idx] = KL_KU_pair.second;
    }

    read_impl(mtx, data, KLs, KUs);
}


}  // namespace


template <typename ValueType>
void BatchBand<ValueType>::read(const std::vector<mat_data>& data)
{
    read_impl(this, data);
}

template <typename ValueType>
void BatchBand<ValueType>::read(const std::vector<mat_data32>& data)
{
    read_impl(this, data);
}

template <typename ValueType>
void BatchBand<ValueType>::read(const std::vector<mat_data>& data,
                                const batch_stride& KLs,
                                const batch_stride& KUs)
{
    read_impl(this, data, KLs, KUs);
}

template <typename ValueType>
void BatchBand<ValueType>::read(const std::vector<mat_data32>& data,
                                const batch_stride& KLs,
                                const batch_stride& KUs)
{
    read_impl(this, data, KLs, KUs);
}

namespace {

template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType* mtx, std::vector<MatrixData>& data)
{
    std::unique_ptr<const BatchLinOp> op{};
    const MatrixType* tmp{};

    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType*>(op.get());
    } else {
        tmp = mtx;
    }

    data = std::vector<MatrixData>(mtx->get_num_batch_entries());

    for (size_type batch_entry_idx = 0;
         batch_entry_idx < mtx->get_num_batch_entries(); ++batch_entry_idx) {
        assert(mtx->get_size().at(batch_entry_idx)[0] ==
               mtx->get_size().at(batch_entry_idx)[1]);

        const auto size = mtx->get_size().at(batch_entry_idx)[0];
        data[batch_entry_idx] = {mtx->get_size().at(batch_entry_idx), {}};

        const auto kl = tmp->get_num_lower_diagonals().at(batch_entry_idx);
        const auto ku = tmp->get_num_upper_diagonals().at(batch_entry_idx);

        for (size_type row = 0; row < data[batch_entry_idx].size[0]; ++row) {
            for (size_type col = static_cast<size_type>(std::max(
                     int{0}, static_cast<int>(row) - static_cast<int>(kl)));
                 col <= std::min(size - 1, row + ku); ++col) {
                auto val = tmp->at_in_reference_to_dense_layout(batch_entry_idx,
                                                                row, col);
                if (val != zero<typename MatrixType::value_type>()) {
                    data[batch_entry_idx].nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }
}

}  // namespace

template <typename ValueType>
void BatchBand<ValueType>::write(std::vector<mat_data>& data) const
{
    write_impl(this, data);
}

template <typename ValueType>
void BatchBand<ValueType>::write(std::vector<mat_data32>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchBand<ValueType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchBand<ValueType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void BatchBand<ValueType>::add_scaled_identity_impl(
    const BatchLinOp* const a, const BatchLinOp* const b) GKO_NOT_IMPLEMENTED;


#define GKO_DECLARE_BATCH_BAND_MATRIX(ValueType) class BatchBand<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BAND_MATRIX);


}  // namespace matrix
}  // namespace gko
