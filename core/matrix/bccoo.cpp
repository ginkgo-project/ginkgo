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
GKO_REGISTER_OPERATION(get_default_compression, bccoo::get_default_compression);
GKO_REGISTER_OPERATION(spmv, bccoo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, bccoo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, bccoo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, bccoo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_to_bccoo, bccoo::convert_to_bccoo);
GKO_REGISTER_OPERATION(convert_to_next_precision,
                       bccoo::convert_to_next_precision);
GKO_REGISTER_OPERATION(convert_to_coo, bccoo::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, bccoo::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_dense, bccoo::convert_to_dense);
GKO_REGISTER_OPERATION(extract_diagonal, bccoo::extract_diagonal);
GKO_REGISTER_OPERATION(compute_absolute_inplace,
                       bccoo::compute_absolute_inplace);
GKO_REGISTER_OPERATION(compute_absolute, bccoo::compute_absolute);
GKO_REGISTER_OPERATION(mem_size_bccoo, bccoo::mem_size_bccoo);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace bccoo


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

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
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

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
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

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
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    bccoo::compression compress_src = this->get_compression();
    size_type block_size_src = this->get_block_size();
    bccoo::compression compress_res = result->get_compression();
    size_type block_size_res = result->get_block_size();

    // For non initialized result objects, the compression and block_size
    // values are copied from "this"
    if ((compress_res == matrix::bccoo::compression::def_value) &&
        			(block_size_res == 0)) {
        block_size_res = block_size_src;
        compress_res = compress_src;
    } else {
        // For partial non initialized result objects, compression or
        // block_size defaults are used
        if (compress_res == matrix::bccoo::compression::def_value) {
            exec->run(bccoo::make_get_default_compression(&compress_res));
        }
        if (block_size_res == 0) {
            exec->run(bccoo::make_get_default_block_size(&block_size_res));
        }
    }

    auto num_stored_elements = this->get_num_stored_elements();
    size_type mem_size_res{};
    // If the compression and block_size values are the same in "this" and
    // result objects, a raw copy is applied
    if ((compress_res == this->get_compression()) &&
        (block_size_res == this->get_block_size())) {
        *result = *this;
    } else if (exec == exec_master) {
        // The standard copy calculates the size in bytes of the chunk,
        // before of the object creation and the conversion
        exec->run(bccoo::make_mem_size_bccoo(this, compress_res, block_size_res,
                                             &mem_size_res));
        auto tmp = Bccoo<ValueType, IndexType>::create(
            exec, this->get_size(), num_stored_elements, block_size_res,
            mem_size_res, compress_res);
        exec->run(bccoo::make_convert_to_bccoo(this, tmp.get()));
        *result = *tmp;
    } else {
        // If the executor of "this" is not the master, the conversion is made
        // in master, and then is moved to the corresponding executor
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        exec_master->run(bccoo::make_mem_size_bccoo(
            host_bccoo.get(), compress_res, block_size_res, &mem_size_res));
        auto tmp = Bccoo<ValueType, IndexType>::create(
            exec_master, host_bccoo->get_size(), num_stored_elements,
            block_size_res, mem_size_res, compress_res);
        exec_master->run(
            bccoo::make_convert_to_bccoo(host_bccoo.get(), tmp.get()));
        *result = *tmp;
    }
    // Other alternative could that make_mem_size_bccoo inicializes
    // the internal vectors whose size is num_blocks_res:
    // const auto num_blocks_res = ceildiv(num_stored_elements, block_size_res);
    // array<IndexType> rows(exec, num_blocks_res);
    // array<IndexType> cols(exec,
    //		(compress_src == matrix::bccoo::compression::block) *
    // num_blocks_res);
    // array<uint8> types(exec,
    // 		(compress_src == matrix::bccoo::compression::block) *
    // num_blocks_res); array<IndexType> offsets(exec, num_blocks_res + 1);
    // And use an alternative definition of make_mem_size_bccoo:
    // exec->run(csr::make_mem_size_bccoo(this, rows.get_data(),
    // 		cols.get_data(), types.get_data(), offsets.get_data(),
    // 		num_blocks_res, block_size_res, &mem_size_res));
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Bccoo<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Bccoo<next_precision<ValueType>, IndexType>* result) const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    using new_precision = next_precision<ValueType>;

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    bccoo::compression compress_src = this->get_compression();
    size_type block_size_src = this->get_block_size();
    bccoo::compression compress_res = result->get_compression();
    size_type block_size_res = result->get_block_size();

    if ((compress_res == matrix::bccoo::compression::def_value) &&
        (block_size_res == 0)) {
        block_size_res = block_size_src;
        compress_res = compress_src;
    } else {
        if (compress_res == matrix::bccoo::compression::def_value) {
            exec->run(bccoo::make_get_default_compression(&compress_res));
        }

        if (block_size_res == 0) {
            exec->run(bccoo::make_get_default_block_size(&block_size_res));
        }
    }

    auto num_stored_elements = this->get_num_stored_elements();

    size_type mem_size_res{};
    if (exec == exec_master) {
        exec->run(bccoo::make_mem_size_bccoo(this, compress_res, block_size_res,
                                             &mem_size_res));
        mem_size_res += num_stored_elements * sizeof(new_precision);
        mem_size_res -= num_stored_elements * sizeof(ValueType);
        auto tmp = Bccoo<new_precision, IndexType>::create(
            exec, this->get_size(), num_stored_elements, block_size_res,
            mem_size_res, compress_res);
        exec->run(bccoo::make_convert_to_next_precision(this, tmp.get()));
        *result = *tmp;
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        exec_master->run(bccoo::make_mem_size_bccoo(
            host_bccoo.get(), compress_res, block_size_res, &mem_size_res));
        mem_size_res += num_stored_elements * sizeof(new_precision);
        mem_size_res -= num_stored_elements * sizeof(ValueType);
        auto tmp = Bccoo<new_precision, IndexType>::create(
            exec_master, host_bccoo->get_size(), num_stored_elements,
            block_size_res, mem_size_res, compress_res);
        exec_master->run(
            bccoo::make_convert_to_next_precision(host_bccoo.get(), tmp.get()));
        *result = *tmp;
    }
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
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        auto tmp = Coo<ValueType, IndexType>::create(
            exec, this->get_size(), this->get_num_stored_elements());
        exec->run(bccoo::make_convert_to_coo(this, tmp.get()));
        tmp->move_to(result);
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        auto tmp = Coo<ValueType, IndexType>::create(
            exec_master, host_bccoo->get_size(),
            host_bccoo->get_num_stored_elements());
        exec_master->run(
            bccoo::make_convert_to_coo(host_bccoo.get(), tmp.get()));
        tmp->move_to(result);
    }
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
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        auto tmp = Csr<ValueType, IndexType>::create(
            exec, this->get_size(), this->get_num_stored_elements(),
            result->get_strategy());
        exec->run(bccoo::make_convert_to_csr(this, tmp.get()));
        tmp->make_srow();
        tmp->move_to(result);
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        auto tmp = Csr<ValueType, IndexType>::create(
            exec_master, host_bccoo->get_size(),
            host_bccoo->get_num_stored_elements(), result->get_strategy());
        exec_master->run(
            bccoo::make_convert_to_csr(host_bccoo.get(), tmp.get()));
        tmp->make_srow();
        tmp->move_to(result);
    }
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        auto tmp = Dense<ValueType>::create(exec, this->get_size());
        exec->run(bccoo::make_convert_to_dense(this, tmp.get()));
        tmp->move_to(result);
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        auto tmp =
            Dense<ValueType>::create(exec_master, host_bccoo->get_size());
        exec_master->run(
            bccoo::make_convert_to_dense(host_bccoo.get(), tmp.get()));
        tmp->move_to(result);
    }
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

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // Block partitioning. If the initial value is 0, the default is chosen
    size_type block_size = this->get_block_size();
    if (block_size == 0) {
        exec->run(bccoo::make_get_default_block_size(&block_size));
    }
    size_type num_blocks = ceildiv(nnz, block_size);

    // Compression. If the initial value is def_value, the default is chosen
    bccoo::compression compress = this->get_compression();
    if (compress == matrix::bccoo::compression::def_value) {
        exec->run(bccoo::make_get_default_compression(&compress));
    }

    if (compress == matrix::bccoo::compression::element) {
        // Creation of some components of Bccoo
        array<IndexType> rows(exec_master, num_blocks);
        array<IndexType> offsets(exec_master, num_blocks + 1);

        // Computation of mem_size (idxs.shf)
        IndexType* rows_data = rows.get_data();
        IndexType* offsets_data = offsets.get_data();
        compr_idxs idxs = {};
        offsets_data[0] = 0;
        for (const auto& elem : data.nonzeros) {
            if (elem.value != zero<ValueType>()) {
                put_detect_newblock(rows_data, idxs.nblk, idxs.blk, idxs.row,
                                    elem.row - idxs.row, idxs.col);
                size_type col_src_res = cnt_position_newrow_mat_data(
                    elem.row, elem.column, idxs.shf, idxs.row, idxs.col);
                cnt_next_position_value(col_src_res, idxs.shf, idxs.col,
                                        elem.value, idxs.nblk);
                put_detect_endblock(offsets_data, idxs.shf, block_size,
                                    idxs.nblk, idxs.blk);
            }
        }

        // Creation of chunk
        array<uint8> chunk(exec_master, idxs.shf);
        uint8* chunk_data = chunk.get_data();

        // Computation of chunk
        idxs = {};
        offsets_data[0] = 0;
        for (const auto& elem : data.nonzeros) {
            if (elem.value != zero<ValueType>()) {
                put_detect_newblock(rows_data, idxs.nblk, idxs.blk, idxs.row,
                                    elem.row - idxs.row, idxs.col);
                size_type col_src_res = put_position_newrow_mat_data(
                    elem.row, elem.column, chunk_data, idxs.shf, idxs.row,
                    idxs.col);
                put_next_position_value(chunk_data, idxs.nblk, col_src_res,
                                        idxs.shf, idxs.col, elem.value);
                put_detect_endblock(offsets_data, idxs.shf, block_size,
                                    idxs.nblk, idxs.blk);
            }
        }
        if (idxs.nblk > 0) {
            offsets_data[idxs.blk + 1] = idxs.shf;
        }

        // Creation of the Bccoo object
        auto tmp =
            Bccoo::create(exec_master, data.size, std::move(chunk),
                          std::move(offsets), std::move(rows), nnz, block_size);
        this->copy_from(std::move(tmp));
    } else {
        // Creation of some components of Bccoo
        array<IndexType> rows(exec_master, num_blocks);
        array<IndexType> cols(exec_master, num_blocks);
        array<uint8> types(exec_master, num_blocks);
        array<IndexType> offsets(exec_master, num_blocks + 1);

        IndexType* rows_data = rows.get_data();
        IndexType* cols_data = cols.get_data();
        uint8* types_data = types.get_data();
        IndexType* offsets_data = offsets.get_data();

        // Computation of mem_size (idxs.shf)
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        for (const auto& elem : data.nonzeros) {
            if (elem.value != zero<ValueType>()) {
                proc_block_indices(elem.row, elem.column, idxs, blk_idxs);
                idxs.nblk++;
                if (idxs.nblk == block_size) {
                    // Counting bytes to write block on result
                    cnt_block_indices<ValueType>(block_size, blk_idxs, idxs);
                    idxs.blk++;
                    idxs.nblk = 0;
                }
            }
        }
        if (idxs.nblk > 0) {
            // Counting bytes to write block on result
            cnt_block_indices<ValueType>(block_size, blk_idxs, idxs);
            idxs.blk++;
            idxs.nblk = 0;
        }

        // Creation of chunk
        array<uint8> chunk(exec_master, idxs.shf);
        uint8* chunk_data = chunk.get_data();

        // Creation of auxiliary vectors and scalar
        array<IndexType> rows_blk(exec, block_size);
        array<IndexType> cols_blk(exec, block_size);
        array<ValueType> vals_blk(exec, block_size);
        uint8 type_blk = {};

        // Computation of chunk
        idxs = {};
        blk_idxs = {};
        offsets_data[0] = 0;
        for (const auto& elem : data.nonzeros) {
            if (elem.value != zero<ValueType>()) {
                proc_block_indices(elem.row, elem.column, idxs, blk_idxs);
                rows_blk.get_data()[idxs.nblk] = elem.row;
                cols_blk.get_data()[idxs.nblk] = elem.column;
                vals_blk.get_data()[idxs.nblk] = elem.value;
                idxs.nblk++;
                if (idxs.nblk == block_size) {
                    type_blk =
                        write_chunk_blk_type(idxs, blk_idxs, rows_blk, cols_blk,
                                             vals_blk, chunk_data);
                    rows_data[idxs.blk] = blk_idxs.row_frs;
                    cols_data[idxs.blk] = blk_idxs.col_frs;
                    types_data[idxs.blk] = type_blk;
                    offsets_data[idxs.blk + 1] = idxs.shf;

                    idxs.blk++;
                    idxs.nblk = 0;
                }
            }
        }
        if (idxs.nblk > 0) {
            type_blk = write_chunk_blk_type(idxs, blk_idxs, rows_blk, cols_blk,
                                            vals_blk, chunk_data);
            rows_data[idxs.blk] = blk_idxs.row_frs;
            cols_data[idxs.blk] = blk_idxs.col_frs;
            types_data[idxs.blk] = type_blk;
            offsets_data[idxs.blk + 1] = idxs.shf;

            idxs.blk++;
            idxs.nblk = 0;
        }

        // Creation of the Bccoo object
        auto tmp =
            Bccoo::create(exec_master, data.size, std::move(chunk),
                          std::move(offsets), std::move(types), std::move(cols),
                          std::move(rows), nnz, block_size);
        this->copy_from(std::move(tmp));
    }
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::write(mat_data& data) const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // If the executor of the object is not master, a copy on
    // master is done
    std::unique_ptr<const LinOp> op{};
    const Bccoo* tmp{};
    if (exec_master != exec) {
        op = this->clone(exec_master);
        tmp = static_cast<const Bccoo*>(op.get());
    } else {
        tmp = this;
    }

    // Getting data from the object to be written
    const IndexType* rows_data = tmp->get_const_rows();
    const IndexType* cols_data = tmp->get_const_cols();
    const uint8* types_data = tmp->get_const_types();
    const IndexType* offsets_data = tmp->get_const_offsets();
    const uint8* chunk_data = tmp->get_const_chunk();

    const size_type num_stored_elements = tmp->get_num_stored_elements();
    const size_type block_size = tmp->get_block_size();

    // Creation of the data vector
    data = {this->get_size(), {}};

    if (tmp->use_element_compression()) {
        compr_idxs idxs = {};
        ValueType val;
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            // Reading (row,col,val) from source
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            // Writing (row,col,val) to result
            data.nonzeros.emplace_back(idxs.row, idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
        }
    } else {
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        ValueType val;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from source
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Writing (row,col,val) to result
                data.nonzeros.emplace_back(idxs.row, idxs.col, val);
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Bccoo<ValueType, IndexType>::extract_diagonal() const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // Creation and initialization of the result
    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(bccoo::make_fill_array(diag->get_values(), diag->get_size()[0],
                                     zero<ValueType>()));

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        exec->run(bccoo::make_extract_diagonal(this, lend(diag)));
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        auto tmp = Diagonal<ValueType>::create(exec_master, diag_size);
        exec_master->run(
            bccoo::make_extract_diagonal(host_bccoo.get(), lend(tmp)));
        diag->copy_from(std::move(tmp));
    }
    return diag;
}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::compute_absolute_inplace()
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        exec->run(bccoo::make_compute_absolute_inplace(this));
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        exec_master->run(
            bccoo::make_compute_absolute_inplace(host_bccoo.get()));
        *this = *host_bccoo;
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Bccoo<ValueType, IndexType>::absolute_type>
Bccoo<ValueType, IndexType>::compute_absolute() const
{
    // This routine doesn't make sense for non initialized objects
    GKO_ASSERT(this->use_default_compression());

    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // Gettinh information from the original object
    size_type block_size = this->get_block_size();
    size_type num_nonzeros = this->get_num_stored_elements();
    size_type num_bytes = this->get_num_bytes();
    bccoo::compression compress = this->get_compression();

    // The size of chunk related to the new object is computed from
    // the size of original object
    num_bytes += (sizeof(remove_complex<ValueType>) * block_size) -
                 (sizeof(ValueType) * block_size);
    auto abs_bccoo = absolute_type::create(exec, this->get_size(), num_nonzeros,
                                           block_size, num_bytes, compress);

    // If the block compression is used, the conversion could be directly
    // done on all executors
    if ((exec == exec_master) || this->use_block_compression()) {
        exec->run(bccoo::make_compute_absolute(this, abs_bccoo.get()));
    } else {
        auto host_bccoo = Bccoo<ValueType, IndexType>::create(exec_master);
        *host_bccoo = *this;
        auto tmp =
            absolute_type::create(exec_master, this->get_size(), num_nonzeros,
                                  block_size, num_bytes, compress);
        exec_master->run(
            bccoo::make_compute_absolute(host_bccoo.get(), tmp.get()));
        *abs_bccoo = *tmp;
    }
    return abs_bccoo;
}


#define GKO_DECLARE_BCCOO_MATRIX(ValueType, IndexType) \
    class Bccoo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_MATRIX);


}  // namespace matrix
}  // namespace gko
