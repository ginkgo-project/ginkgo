// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <limits>
#include <map>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/fbcsr_kernels.hpp"


namespace gko {
namespace matrix {
namespace fbcsr {
namespace {


GKO_REGISTER_OPERATION(spmv, fbcsr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, fbcsr::advanced_spmv);
GKO_REGISTER_OPERATION(fill_in_matrix_data, fbcsr::fill_in_matrix_data);
GKO_REGISTER_OPERATION(convert_to_csr, fbcsr::convert_to_csr);
GKO_REGISTER_OPERATION(fill_in_dense, fbcsr::fill_in_dense);
GKO_REGISTER_OPERATION(transpose, fbcsr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, fbcsr::conj_transpose);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       fbcsr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(sort_by_column_index, fbcsr::sort_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, fbcsr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace fbcsr


template <typename ValueType, typename IndexType>
Fbcsr<ValueType, IndexType>& Fbcsr<ValueType, IndexType>::operator=(
    const Fbcsr& other)
{
    if (&other != this) {
        EnableLinOp<Fbcsr>::operator=(other);
        // block size is immutable except for assignment
        bs_ = other.bs_;
        values_ = other.values_;
        col_idxs_ = other.col_idxs_;
        row_ptrs_ = other.row_ptrs_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Fbcsr<ValueType, IndexType>& Fbcsr<ValueType, IndexType>::operator=(
    Fbcsr&& other)
{
    if (&other != this) {
        EnableLinOp<Fbcsr>::operator=(std::move(other));
        // block size is immutable except for assignment
        bs_ = other.bs_;
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        row_ptrs_ = std::move(other.row_ptrs_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Fbcsr<ValueType, IndexType>::Fbcsr(const Fbcsr& other)
    : Fbcsr{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Fbcsr<ValueType, IndexType>::Fbcsr(Fbcsr&& other) : Fbcsr{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::apply_impl(const LinOp* const b,
                                             LinOp* const x) const
{
    if (auto b_fbcsr = dynamic_cast<const Fbcsr<ValueType, IndexType>*>(b)) {
        // if b is a FBCSR matrix, we need an SpGeMM
        GKO_NOT_SUPPORTED(b_fbcsr);
    } else {
        // otherwise we assume that b is dense and compute a SpMV/SpMM
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->get_executor()->run(
                    fbcsr::make_spmv(this, dense_b, dense_x));
            },
            b, x);
    }
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::apply_impl(const LinOp* const alpha,
                                             const LinOp* const b,
                                             const LinOp* const beta,
                                             LinOp* const x) const
{
    if (auto b_fbcsr = dynamic_cast<const Fbcsr<ValueType, IndexType>*>(b)) {
        // if b is a FBCSR matrix, we need an SpGeMM
        GKO_NOT_SUPPORTED(b_fbcsr);
    } else if (auto b_ident = dynamic_cast<const Identity<ValueType>*>(b)) {
        // if b is an identity matrix, we need an SpGEAM
        GKO_NOT_SUPPORTED(b_ident);
    } else {
        // otherwise we assume that b is dense and compute a SpMV/SpMM
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                this->get_executor()->run(fbcsr::make_advanced_spmv(
                    dense_alpha, this, dense_b, dense_beta, dense_x));
            },
            alpha, b, beta, x);
    }
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Fbcsr<next_precision<ValueType>, IndexType>* const result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    // block sizes are immutable except for assignment/conversion
    result->bs_ = this->bs_;
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    Fbcsr<next_precision<ValueType>, IndexType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Dense<ValueType>* const result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(fbcsr::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(Dense<ValueType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* const result) const
{
    auto exec = this->get_executor();
    {
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
        tmp->col_idxs_.resize_and_reset(this->get_num_stored_elements());
        tmp->values_.resize_and_reset(this->get_num_stored_elements());
        tmp->set_size(this->get_size());
        exec->run(fbcsr::make_convert_to_csr(this, tmp.get()));
    }
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    Csr<ValueType, IndexType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    SparsityCsr<ValueType, IndexType>* const result) const
{
    result->set_size(
        gko::dim<2>{static_cast<size_type>(this->get_num_block_rows()),
                    static_cast<size_type>(this->get_num_block_cols())});
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->value_ =
        array<ValueType>(result->get_executor(), {one<ValueType>()});
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    SparsityCsr<ValueType, IndexType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::read(const device_mat_data& data)
{
    // make a copy, read the data in
    this->read(device_mat_data{this->get_executor(), data});
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::read(device_mat_data&& data)
{
    const auto row_blocks = detail::get_num_blocks(bs_, data.get_size()[0]);
    const auto col_blocks = detail::get_num_blocks(bs_, data.get_size()[1]);
    this->set_size(data.get_size());
    row_ptrs_.resize_and_reset(row_blocks + 1);
    auto exec = this->get_executor();
    {
        auto local_data = make_temporary_clone(exec, &data);
        exec->run(fbcsr::make_fill_in_matrix_data(*local_data, bs_, row_ptrs_,
                                                  col_idxs_, values_));
    }
    // this needs to happen after the temporary clone copy-back
    data.empty_out();
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {tmp->get_size(), {}};

    const size_type nbnz = tmp->get_num_stored_blocks();
    const acc::range<acc::block_col_major<const value_type, 3>> vblocks(
        std::array<acc::size_type, 3>{static_cast<acc::size_type>(nbnz),
                                      static_cast<acc::size_type>(bs_),
                                      static_cast<acc::size_type>(bs_)},
        tmp->values_.get_const_data());

    for (size_type brow = 0; brow < tmp->get_num_block_rows(); ++brow) {
        const auto start = tmp->row_ptrs_.get_const_data()[brow];
        const auto end = tmp->row_ptrs_.get_const_data()[brow + 1];

        for (int ib = 0; ib < bs_; ib++) {
            const auto row = brow * bs_ + ib;
            for (auto inz = start; inz < end; ++inz) {
                for (int jb = 0; jb < bs_; jb++) {
                    const auto col =
                        tmp->col_idxs_.get_const_data()[inz] * bs_ + jb;
                    const auto val = vblocks(inz, ib, jb);
                    data.nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Fbcsr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Fbcsr::create(exec, gko::transpose(this->get_size()),
                                   this->get_num_stored_elements(), bs_);

    exec->run(fbcsr::make_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Fbcsr<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Fbcsr::create(exec, gko::transpose(this->get_size()),
                                   this->get_num_stored_elements(), bs_);

    exec->run(fbcsr::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::sort_by_column_index()
{
    auto exec = this->get_executor();
    exec->run(fbcsr::make_sort_by_column_index(this));
}


template <typename ValueType, typename IndexType>
bool Fbcsr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(fbcsr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Fbcsr<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(fbcsr::make_fill_array(diag->get_values(), diag->get_size()[0],
                                     zero<ValueType>()));
    exec->run(fbcsr::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(fbcsr::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Fbcsr<ValueType, IndexType>::absolute_type>
Fbcsr<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_fbcsr = absolute_type::create(exec, this->get_size(),
                                           this->get_num_stored_elements(),
                                           this->get_block_size());

    abs_fbcsr->col_idxs_ = col_idxs_;
    abs_fbcsr->row_ptrs_ = row_ptrs_;
    exec->run(fbcsr::make_outplace_absolute_array(
        this->get_const_values(), this->get_num_stored_elements(),
        abs_fbcsr->get_values()));

    return abs_fbcsr;
}


#define GKO_DECLARE_FBCSR_MATRIX(ValueType, IndexType) \
    class Fbcsr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_MATRIX);


}  // namespace matrix
}  // namespace gko
