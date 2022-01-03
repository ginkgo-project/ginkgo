/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/matrix/hybrid.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array_kernels.hpp"
#include "core/components/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"


namespace gko {
namespace matrix {
namespace hybrid {
namespace {


GKO_REGISTER_OPERATION(build_row_ptrs, components::build_row_ptrs);
GKO_REGISTER_OPERATION(compute_row_nnz, hybrid::compute_row_nnz);
GKO_REGISTER_OPERATION(split_matrix_data, hybrid::split_matrix_data);
GKO_REGISTER_OPERATION(convert_to_dense, hybrid::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_csr, hybrid::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros, hybrid::count_nonzeros);
GKO_REGISTER_OPERATION(extract_coo_diagonal, coo::extract_diagonal);
GKO_REGISTER_OPERATION(extract_ell_diagonal, ell::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace hybrid


namespace {


template <typename ValueType, typename IndexType>
void get_each_row_nnz(const matrix_data<ValueType, IndexType>& data,
                      Array<size_type>& row_nnz)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    auto row_nnz_val = row_nnz.get_data();
    for (size_type i = 0; i < row_nnz.get_num_elems(); i++) {
        row_nnz_val[i] = zero<size_type>();
    }
    for (const auto& elem : data.nonzeros) {
        if (elem.row != current_row) {
            row_nnz_val[current_row] = nnz;
            current_row = elem.row;
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    row_nnz_val[current_row] = nnz;
}


}  // namespace


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto ell_mtx = this->get_ell();
            auto coo_mtx = this->get_coo();
            ell_mtx->apply(dense_b, dense_x);
            coo_mtx->apply2(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto ell_mtx = this->get_ell();
            auto coo_mtx = this->get_coo();
            ell_mtx->apply(dense_alpha, dense_b, dense_beta, dense_x);
            coo_mtx->apply2(dense_alpha, dense_b, dense_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(
    Hybrid<next_precision<ValueType>, IndexType>* result) const
{
    this->ell_->convert_to(result->ell_.get());
    this->coo_->convert_to(result->coo_.get());
    // TODO set strategy correctly
    // There is no way to correctly clone the strategy like in
    // Csr::convert_to
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(
    Hybrid<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(hybrid::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();

    size_type num_stored_elements = 0;
    exec->run(hybrid::make_count_nonzeros(this, &num_stored_elements));

    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), num_stored_elements, result->get_strategy());
    exec->run(hybrid::make_convert_to_csr(this, tmp.get()));

    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    auto local_data = make_temporary_clone(exec, &data.nonzeros);
    Array<int64> row_ptrs{exec, data.size[0] + 1};
    exec->run(hybrid::make_build_row_ptrs(*local_data, data.size[0],
                                          row_ptrs.get_data()));
    Array<size_type> row_nnz{exec, data.size[0]};
    exec->run(hybrid::make_compute_row_nnz(row_ptrs, row_nnz.get_data()));
    size_type ell_max_nnz{};
    size_type coo_nnz{};
    this->get_strategy()->compute_hybrid_config(row_nnz, &ell_max_nnz,
                                                &coo_nnz);
    auto ell_nnz = data.nonzeros.get_num_elems() - coo_nnz;
    device_mat_data ell_data{exec, data.size, ell_nnz};
    device_mat_data coo_data{exec, data.size, coo_nnz};
    exec->run(hybrid::make_split_matrix_data(
        data.nonzeros, row_ptrs.get_const_data(), ell_max_nnz, data.size[0],
        ell_data.nonzeros, coo_data.nonzeros));
    this->set_size(data.size);
    ell_->read(ell_data);
    coo_->read(coo_data);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_view_from_host(
        this->get_executor(), const_cast<mat_data&>(data)));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::write(mat_data& data) const
{
    std::unique_ptr<const LinOp> op{};
    auto tmp_clone =
        make_temporary_clone(this->get_executor()->get_master(), this);
    auto tmp = tmp_clone.get();
    data = {tmp->get_size(), {}};
    size_type coo_ind = 0;
    auto coo_nnz = tmp->get_coo_num_stored_elements();
    auto coo_vals = tmp->get_const_coo_values();
    auto coo_col_idxs = tmp->get_const_coo_col_idxs();
    auto coo_row_idxs = tmp->get_const_coo_row_idxs();
    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        for (size_type i = 0; i < tmp->get_ell_num_stored_elements_per_row();
             ++i) {
            const auto val = tmp->ell_val_at(row, i);
            if (val != zero<ValueType>()) {
                const auto col = tmp->ell_col_at(row, i);
                data.nonzeros.emplace_back(row, col, val);
            }
        }

        while (coo_ind < coo_nnz && coo_row_idxs[coo_ind] == row) {
            if (coo_vals[coo_ind] != zero<ValueType>()) {
                data.nonzeros.emplace_back(row, coo_col_idxs[coo_ind],
                                           coo_vals[coo_ind]);
            }
            coo_ind++;
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Hybrid<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(hybrid::make_fill_array(diag->get_values(), diag->get_size()[0],
                                      zero<ValueType>()));
    exec->run(hybrid::make_extract_ell_diagonal(this->get_ell(), lend(diag)));
    exec->run(hybrid::make_extract_coo_diagonal(this->get_coo(), lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(hybrid::make_inplace_absolute_array(
        this->get_ell_values(), this->get_ell_num_stored_elements()));
    exec->run(hybrid::make_inplace_absolute_array(
        this->get_coo_values(), this->get_coo_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Hybrid<ValueType, IndexType>::absolute_type>
Hybrid<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_hybrid = absolute_type::create(
        exec, this->get_size(), this->get_strategy<absolute_type>());

    abs_hybrid->ell_->copy_from(ell_->compute_absolute());
    abs_hybrid->coo_->copy_from(coo_->compute_absolute());

    return abs_hybrid;
}


#define GKO_DECLARE_HYBRID_MATRIX(ValueType, IndexType) \
    class Hybrid<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYBRID_MATRIX);


}  // namespace matrix
}  // namespace gko
