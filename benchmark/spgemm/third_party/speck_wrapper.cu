/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "benchmark_wrappers.hpp"


#include <Multiply.h>
#include <dCSR.h>
#include <spECKConfig.h>


#include "core/matrix/csr_builder.hpp"

namespace gko {


template <typename ValueType>
void SpeckCsr<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = as<CudaExecutor>(this->get_executor());

    auto b_csr = as<csr>(b);
    auto x_csr = as<csr>(x);
    dCSR<ValueType> this_wrap{};
    dCSR<ValueType> b_wrap{};
    dCSR<ValueType> x_wrap{};
    this_wrap.rows = this->get_size()[0];
    this_wrap.cols = this->get_size()[1];
    this_wrap.nnz = this->csr_->get_num_stored_elements();
    this_wrap.row_offsets = reinterpret_cast<uint32 *>(
        const_cast<int32 *>(this->csr_->get_const_row_ptrs()));
    this_wrap.col_ids = reinterpret_cast<uint32 *>(
        const_cast<int32 *>(this->csr_->get_const_col_idxs()));
    this_wrap.data = const_cast<ValueType *>(this->csr_->get_const_values());

    b_wrap.rows = b_csr->get_size()[0];
    b_wrap.cols = b_csr->get_size()[1];
    b_wrap.nnz = b_csr->get_num_stored_elements();
    b_wrap.row_offsets = reinterpret_cast<uint32 *>(
        const_cast<int32 *>(b_csr->get_const_row_ptrs()));
    b_wrap.col_ids = reinterpret_cast<uint32 *>(
        const_cast<int32 *>(b_csr->get_const_col_idxs()));
    b_wrap.data = const_cast<ValueType *>(b_csr->get_const_values());

    x_wrap.rows = x_csr->get_size()[0];
    x_wrap.cols = x_csr->get_size()[1];
    x_wrap.nnz = 0;
    x_wrap.row_offsets = exec->template alloc<uint32>(x_wrap.rows + 1);
    x_wrap.col_ids = nullptr;
    x_wrap.data = nullptr;
    auto config = spECK::spECKConfig::initialize(exec->get_device_id());
    Timings stats{};

    // separately log the runtimes to eliminate allocation and init overheads
    this->get_executor()->run(
        "spgemm", [] {},
        [&] {
            spECK::MultiplyspECK<ValueType, 4, 1024,
                                 spECK_DYNAMIC_MEM_PER_BLOCK,
                                 spECK_STATIC_MEM_PER_BLOCK>(
                this_wrap, b_wrap, x_wrap, config, stats);
        },
        [] {});

    matrix::CsrBuilder<ValueType, int32> x_builder{x_csr};
    x_builder.get_col_idx_array() = Array<int32>(
        exec, x_wrap.nnz, reinterpret_cast<int32 *>(x_wrap.col_ids));
    x_builder.get_value_array() =
        Array<ValueType>(exec, x_wrap.nnz, x_wrap.data);

    exec->copy(x_wrap.rows + 1, reinterpret_cast<int32 *>(x_wrap.row_offsets),
               x_csr->get_row_ptrs());
    // prevent data from being deleted
    this_wrap.row_offsets = nullptr;
    this_wrap.col_ids = nullptr;
    this_wrap.data = nullptr;
    b_wrap.row_offsets = nullptr;
    b_wrap.col_ids = nullptr;
    b_wrap.data = nullptr;
    x_wrap.col_ids = nullptr;
    x_wrap.data = nullptr;
}


template class SpeckCsr<float>;
template class SpeckCsr<double>;

}  // namespace gko
