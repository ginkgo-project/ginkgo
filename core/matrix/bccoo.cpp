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

#include <ginkgo/core/matrix/bccoo.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/bccoo_kernels.hpp"


namespace gko {
namespace matrix {
namespace bccoo {


GKO_REGISTER_OPERATION(get_default_block_size, bccoo::get_default_block_size);
GKO_REGISTER_OPERATION(spmv, bccoo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, bccoo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, bccoo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, bccoo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_to_compression, bccoo::convert_to_compression);
GKO_REGISTER_OPERATION(convert_to_next_precision,
                       bccoo::convert_to_next_precision);
GKO_REGISTER_OPERATION(convert_to_coo, bccoo::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, bccoo::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_dense, bccoo::convert_to_dense);
GKO_REGISTER_OPERATION(extract_diagonal, bccoo::extract_diagonal);
GKO_REGISTER_OPERATION(compute_absolute_inplace,
                       bccoo::compute_absolute_inplace);
GKO_REGISTER_OPERATION(compute_absolute, bccoo::compute_absolute);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace bccoo


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(bccoo::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(bccoo::make_advanced_spmv(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply2_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                bccoo::make_spmv2(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply2_impl(const LinOp* alpha,
                                              const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_x) {
            this->get_executor()->run(bccoo::make_advanced_spmv2(
                dense_alpha, this, dense_b, dense_x));
        },
        alpha, b, x);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Bccoo<ValueType, IndexType>* result) const
{
    gko::matrix::bccoo::compression compression_src = this->get_compression();
    gko::matrix::bccoo::compression compression_res = result->get_compression();

    /*
    if (result_executor is on GPU) {
            compression_res = gko::matrix::bccoo::compression::block;
    }
    */
    if (compression_src == compression_res) {
        // result = this->clone();
    } else if (compression_src == gko::matrix::bccoo::compression::element) {
        // convert_from_element_to_block(this, result);
    } else {
        // convert_from_block_to_element(this, result);
        /*
            gko::matrix::bccoo::compression compression_res =
           resource->get_compression(); auto exec = this->get_executor();
            gko::matrix::bccoo::compression compression =
           this->get_compression(); size_type block_size =
           this->get_block_size(); size_type num_nonzeros =
           this->get_num_stored_elements(); size_type num_bytes =
           this->get_num_bytes(); num_bytes += num_nonzeros *
           sizeof(new_precision); num_bytes -= num_nonzeros * sizeof(ValueType);
            auto tmp = Bccoo<new_precision, IndexType>::create(
                exec, this->get_size(), num_nonzeros, block_size, num_bytes,
                compression);
            exec->run(bccoo::make_convert_to_next_precision(this, tmp.get()));
            tmp->move_to(result);
        */
    }
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Bccoo<next_precision<ValueType>, IndexType>* result) const
{
    using new_precision = next_precision<ValueType>;

    auto exec = this->get_executor();
    gko::matrix::bccoo::compression compression = this->get_compression();
    size_type block_size = this->get_block_size();
    size_type num_nonzeros = this->get_num_stored_elements();
    size_type num_bytes = this->get_num_bytes();
    num_bytes += num_nonzeros * sizeof(new_precision);
    num_bytes -= num_nonzeros * sizeof(ValueType);
    auto tmp = Bccoo<new_precision, IndexType>::create(exec, this->get_size(),
                                                       num_nonzeros, block_size,
                                                       num_bytes, compression);
    exec->run(bccoo::make_convert_to_next_precision(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(
    Bccoo<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = Coo<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements());
    exec->run(bccoo::make_convert_to_coo(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Coo<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    exec->run(bccoo::make_convert_to_csr(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(bccoo::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::read(const mat_data& data)
{
    // Computation of nnz
    size_type nnz = 0;
    for (const auto& elem : data.nonzeros) {
        nnz += (elem.value != zero<ValueType>());
    }

    // Definition of executor
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // Block partitioning
    size_type block_size = 0;
    size_type num_blocks = 0;
    exec->run(bccoo::make_get_default_block_size(&block_size));
    num_blocks = ceildiv(nnz, block_size);

    // Creation of some components of Bccoo
    array<IndexType> rows(exec_master, num_blocks);
    array<IndexType> offsets(exec_master, num_blocks + 1);

    // Computation of rows, offsets and m (mem_size)
    IndexType* rows_data = rows.get_data();
    IndexType* offsets_data = offsets.get_data();
    size_type nblk = 0;
    size_type blk = 0;
    size_type col = 0;
    size_type row = 0;
    size_type shf = 0;
    offsets_data[0] = 0;
    for (const auto& elem : data.nonzeros) {
        if (elem.value != zero<ValueType>()) {
            put_detect_newblock(rows_data, nblk, blk, row, elem.row - row, col);
            size_type colRS = cnt_position_newrow_mat_data(
                elem.row, elem.column, shf, row, col);
            cnt_next_position_value(colRS, shf, col, elem.value, nblk);
            put_detect_endblock(offsets_data, shf, block_size, nblk, blk);
        }
    }

    // Creation of chunk
    array<uint8> chunk(exec_master, shf);
    uint8* chunk_data = chunk.get_data();

    // Computation of chunk
    nblk = 0;
    blk = 0;
    col = 0;
    row = 0;
    shf = 0;
    offsets_data[0] = 0;
    for (const auto& elem : data.nonzeros) {
        if (elem.value != zero<ValueType>()) {
            put_detect_newblock(rows_data, nblk, blk, row, elem.row - row, col);
            size_type colRS = put_position_newrow_mat_data(
                elem.row, elem.column, chunk_data, shf, row, col);
            put_next_position_value(chunk_data, nblk, colRS, shf, col,
                                    elem.value);
            put_detect_endblock(offsets_data, shf, block_size, nblk, blk);
        }
    }

    // Creation of the Bccoo object
    auto tmp =
        Bccoo::create(exec_master, data.size, std::move(chunk),
                      std::move(offsets), std::move(rows), nnz, block_size);
    this->copy_from(std::move(tmp));
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::write(mat_data& data) const
{
    const IndexType* rows_data = this->get_const_rows();
    const IndexType* offsets_data = this->get_const_offsets();
    const uint8* chunk_data = this->get_const_chunk();
    const size_type num_stored_elements = this->get_num_stored_elements();
    const size_type block_size = this->get_block_size();
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    std::unique_ptr<const LinOp> op{};
    const Bccoo* tmp{};
    if (exec_master != exec) {
        op = this->clone(exec_master);
        tmp = static_cast<const Bccoo*>(op.get());
    } else {
        tmp = this;
    }

    // Creation of the data vector
    data = {this->get_size(), {}};

    // Computation of chunk
    size_type nblk = 0, blk = 0, col = 0, row = 0, shf = 0;
    ValueType val;
    for (size_type i = 0; i < num_stored_elements; i++) {
        get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row, col);
        uint8 ind = get_position_newrow(chunk_data, shf, row, col);
        get_next_position_value(chunk_data, nblk, ind, shf, col, val);
        data.nonzeros.emplace_back(row, col, val);
        get_detect_endblock(block_size, nblk, blk);
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Bccoo<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();
    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(bccoo::make_fill_array(diag->get_values(), diag->get_size()[0],
                                     zero<ValueType>()));
    exec->run(bccoo::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();
    exec->run(bccoo::make_compute_absolute_inplace(this));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Bccoo<ValueType, IndexType>::absolute_type>
Bccoo<ValueType, IndexType>::compute_absolute() const
{
    size_type block_size = this->get_block_size();
    size_type num_nonzeros = this->get_num_stored_elements();
    size_type num_bytes = this->get_num_bytes();
    gko::matrix::bccoo::compression compression = this->get_compression();
    auto exec = this->get_executor();
    auto abs_bccoo = absolute_type::create(exec, this->get_size(), num_nonzeros,
                                           block_size, num_bytes, compression);
    exec->run(bccoo::make_compute_absolute(this, abs_bccoo.get()));
    return abs_bccoo;
}


#define GKO_DECLARE_BCCOO_MATRIX(ValueType, IndexType) \
    class Bccoo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_MATRIX);


}  // namespace matrix
}  // namespace gko
