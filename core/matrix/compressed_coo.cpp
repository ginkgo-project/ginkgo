// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/compressed_coo.hpp"

#include <algorithm>
#include <numeric>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/device_matrix_data_kernels.hpp"
#include "core/matrix/compressed_coo_kernels.hpp"


namespace gko {
namespace matrix {
namespace compressed_coo {
namespace {


GKO_REGISTER_OPERATION(spmv, compressed_coo::spmv);
GKO_REGISTER_OPERATION(idxs_to_bits, compressed_coo::idxs_to_bits);
GKO_REGISTER_OPERATION(bits_to_idxs, compressed_coo::bits_to_idxs);
GKO_REGISTER_OPERATION(aos_to_soa, components::aos_to_soa);


}  // anonymous namespace
}  // namespace compressed_coo


template <typename ValueType, typename IndexType>
std::unique_ptr<CompactRowCoo<ValueType, IndexType>>
CompactRowCoo<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    size_type num_nonzeros)
{
    return std::unique_ptr<CompactRowCoo>{
        new CompactRowCoo{exec, size, num_nonzeros}};
}


template <typename ValueType, typename IndexType>
CompactRowCoo<ValueType, IndexType>::CompactRowCoo(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    size_type num_nonzeros)
    : EnableLinOp<CompactRowCoo>(exec, size),
      values_(exec, num_nonzeros),
      col_idxs_(exec, num_nonzeros),
      row_bits_(exec, static_cast<size_type>(ceildiv(num_nonzeros, 32))),
      row_ranks_(exec, row_bits_.get_size())
{}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                     LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                compressed_coo::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta,
    LinOp* x) const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::resize(dim<2> new_size, size_type nnz)
{
    this->set_size(new_size);
    const auto num_blocks = static_cast<size_type>(ceildiv(nnz, 32));
    this->row_bits_.resize_and_reset(num_blocks);
    this->row_ranks_.resize_and_reset(num_blocks);
    this->col_idxs_.resize_and_reset(nnz);
    this->values_.resize_and_reset(nnz);
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::row_bits_from_idxs(
    const array<IndexType> row_idxs)
{
    const auto nnz = row_idxs.get_size();
    auto exec = this->get_executor();
    const auto num_blocks = static_cast<size_type>(ceildiv(nnz, 32));
    this->row_bits_.resize_and_reset(num_blocks);
    this->row_ranks_.resize_and_reset(num_blocks);
    exec->run(compressed_coo::make_idxs_to_bits(
        make_temporary_clone(exec, &row_idxs)->get_const_data(), nnz,
        this->row_bits_.get_data(), this->row_ranks_.get_data()));
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::read(const mat_data& data)
{
    auto size = data.size;
    auto exec = this->get_executor();
    this->set_size(size);
    const auto nnz = data.nonzeros.size();
    gko::array<IndexType> row_idxs{exec, nnz};
    this->col_idxs_.resize_and_reset(nnz);
    this->values_.resize_and_reset(nnz);
    device_mat_data view{exec, size, row_idxs.as_view(),
                         this->col_idxs_.as_view(), this->values_.as_view()};
    const auto host_data =
        make_array_view(exec->get_master(), data.nonzeros.size(),
                        const_cast<matrix_data_entry<ValueType, IndexType>*>(
                            data.nonzeros.data()));
    exec->run(compressed_coo::make_aos_to_soa(
        *make_temporary_clone(exec, &host_data), view));
    this->row_bits_from_idxs(row_idxs);
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::read(const device_mat_data& data)
{
    this->set_size(data.get_size());
    auto exec = this->get_executor();
    const auto nnz = data.get_num_stored_elements();
    // copy the arrays from device matrix data into the arrays of
    // this. Compared to the read(device_mat_data&&) version, the internal
    // arrays keep their current ownership status
    this->values_ = make_const_array_view(data.get_executor(),
                                          data.get_num_stored_elements(),
                                          data.get_const_values());
    this->col_idxs_ = make_const_array_view(data.get_executor(),
                                            data.get_num_stored_elements(),
                                            data.get_const_col_idxs());
    auto row_idxs = array_const_cast(make_const_array_view(
        data.get_executor(), data.get_num_stored_elements(),
        data.get_const_row_idxs()));
    this->row_bits_from_idxs(row_idxs);
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->set_size(data.get_size());
    auto arrays = data.empty_out();
    this->values_ = std::move(arrays.values);
    this->col_idxs_ = std::move(arrays.col_idxs);
    auto row_idxs = std::move(arrays.row_idxs);
    this->row_bits_from_idxs(row_idxs);
}


template <typename ValueType, typename IndexType>
void CompactRowCoo<ValueType, IndexType>::write(mat_data& data) const
{
    const auto nnz = this->get_num_stored_elements();
    const auto exec = this->get_executor();
    array<IndexType> row_idxs{exec, nnz};
    exec->run(compressed_coo::make_bits_to_idxs(
        this->row_bits_.get_const_data(), this->row_ranks_.get_const_data(),
        nnz, row_idxs.get_data()));
    const device_matrix_data<ValueType, IndexType> ddata{
        exec, this->get_size(), std::move(row_idxs),
        array_const_cast(this->col_idxs_.as_const_view()),
        array_const_cast(this->values_.as_const_view())};
    data = ddata.copy_to_host();
}


#define GKO_DECLARE_CRCOO_MATRIX(ValueType, IndexType) \
    class CompactRowCoo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CRCOO_MATRIX);


}  // namespace matrix
}  // namespace gko
