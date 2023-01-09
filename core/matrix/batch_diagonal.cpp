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

#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/matrix/batch_diagonal_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_diagonal {


GKO_REGISTER_OPERATION(apply, batch_diagonal::apply);
GKO_REGISTER_OPERATION(pre_diag_transform_csr,
                       batch_csr::pre_diag_transform_system);
GKO_REGISTER_OPERATION(pre_diag_scale_csr, batch_csr::batch_scale);
GKO_REGISTER_OPERATION(pre_diag_scale_dense, batch_dense::batch_scale);
GKO_REGISTER_OPERATION(conj_transpose, batch_diagonal::conj_transpose);
GKO_REGISTER_OPERATION(fill, components::fill_array);


}  // namespace batch_diagonal


template <typename ValueType>
void BatchDiagonal<ValueType>::apply_impl(const BatchLinOp* b,
                                          BatchLinOp* x) const
{
    if (!this->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    if (auto xd = dynamic_cast<BatchDense<ValueType>*>(x)) {
        auto bd = as<const BatchDense<ValueType>>(b);
        this->get_executor()->run(batch_diagonal::make_apply(this, bd, xd));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void BatchDiagonal<ValueType>::apply_impl(const BatchLinOp* alpha,
                                          const BatchLinOp* b,
                                          const BatchLinOp* beta,
                                          BatchLinOp* x) const
{
    if (!x->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    auto dense_x = as<matrix::BatchDense<value_type>>(x);
    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
void BatchDiagonal<ValueType>::convert_to(
    BatchDiagonal<next_precision<ValueType>>* const result) const
{
    result->values_ = this->values_;
    result->num_elems_per_batch_cumul_ = this->num_elems_per_batch_cumul_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void BatchDiagonal<ValueType>::move_to(
    BatchDiagonal<next_precision<ValueType>>* const result)
{
    this->convert_to(result);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType* mtx, const std::vector<MatrixData>& data)
{
    using value_type = typename MatrixType::value_type;
    auto batch_sizes = std::vector<dim<2>>(data.size());
    size_type ind = 0;
    for (const auto& b : data) {
        batch_sizes[ind] = b.size;
        ++ind;
    }
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(),
                                  batch_dim<2>(batch_sizes));
    mtx->get_executor()->run(batch_diagonal::make_fill(
        tmp->get_values(), tmp->get_num_stored_elements(), zero<value_type>()));
    for (size_type b = 0; b < data.size(); ++b) {
        for (auto nnz : data[b].nonzeros) {
            if (nnz.row == nnz.column) {
                tmp->at(b, nnz.row) = nnz.value;
            }
        }
    }
    tmp->move_to(mtx);
}


}  // namespace


template <typename ValueType>
void BatchDiagonal<ValueType>::read(const std::vector<mat_data>& data)
{
    read_impl(this, data);
}


template <typename ValueType>
void BatchDiagonal<ValueType>::read(const std::vector<mat_data32>& data)
{
    read_impl(this, data);
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
    for (size_type b = 0; b < mtx->get_num_batch_entries(); ++b) {
        data[b] = {mtx->get_size().at(b), {}};
        const auto nstored = std::min(data[b].size[0], data[b].size[1]);
        for (size_type row = 0; row < nstored; ++row) {
            if (tmp->at(b, row) != zero<typename MatrixType::value_type>()) {
                data[b].nonzeros.emplace_back(row, row, tmp->at(b, row));
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void BatchDiagonal<ValueType>::write(std::vector<mat_data>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void BatchDiagonal<ValueType>::write(std::vector<mat_data32>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDiagonal<ValueType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        BatchDiagonal::create(exec, gko::transpose(this->get_size()));

    exec->copy(this->get_num_stored_elements(), this->get_const_values(),
               trans_cpy->get_values());

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDiagonal<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        BatchDiagonal::create(exec, gko::transpose(this->get_size()));
    exec->run(batch_diagonal::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


#define GKO_DECLARE_BATCH_DIAGONAL_MATRIX(_type) class BatchDiagonal<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIAGONAL_MATRIX);


template <typename ValueType>
void two_sided_batch_system_transform(
    const std::shared_ptr<const Executor> exec,
    const BatchDiagonal<ValueType>* const left,
    const BatchDiagonal<ValueType>* const right, BatchLinOp* const mtx,
    BatchDense<ValueType>* const rhs)
{
    if (auto csrmtx = dynamic_cast<BatchCsr<ValueType>*>(mtx)) {
        exec->run(batch_diagonal::make_pre_diag_transform_csr(left, right,
                                                              csrmtx, rhs));
    } else {
        GKO_NOT_SUPPORTED(mtx);
    }
}


#define GKO_DECLARE_TWO_SIDED_BATCH_SYSTEM_TRANSFORM(_type)   \
    void two_sided_batch_system_transform(                    \
        std::shared_ptr<const Executor> exec,                 \
        const BatchDiagonal<_type>* left_op,                  \
        const BatchDiagonal<_type>* rght_op, BatchLinOp* mtx, \
        BatchDense<_type>* rhs)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_TWO_SIDED_BATCH_SYSTEM_TRANSFORM);


template <typename ValueType>
void two_sided_batch_transform(const std::shared_ptr<const Executor> exec,
                               const BatchDiagonal<ValueType>* const left,
                               const BatchDiagonal<ValueType>* const right,
                               BatchLinOp* const mtx)
{
    if (auto csrmtx = dynamic_cast<BatchCsr<ValueType>*>(mtx)) {
        exec->run(batch_diagonal::make_pre_diag_scale_csr(left, right, csrmtx));
    } else if (auto dmtx = dynamic_cast<BatchDense<ValueType>*>(mtx)) {
        exec->run(batch_diagonal::make_pre_diag_scale_dense(left, right, dmtx));
    } else {
        GKO_NOT_SUPPORTED(mtx);
    }
}


#define GKO_DECLARE_TWO_SIDED_BATCH_TRANSFORM(_type)                     \
    void two_sided_batch_transform(std::shared_ptr<const Executor> exec, \
                                   const BatchDiagonal<_type>* left_op,  \
                                   const BatchDiagonal<_type>* rght_op,  \
                                   BatchLinOp* mtx)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TWO_SIDED_BATCH_TRANSFORM);


}  // namespace matrix


}  // namespace gko
