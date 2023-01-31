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

#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace batch_jacobi {
namespace {


GKO_REGISTER_OPERATION(find_blocks, jacobi::find_blocks);
GKO_REGISTER_OPERATION(extract_common_blocks_pattern,
                       batch_jacobi::extract_common_blocks_pattern);
GKO_REGISTER_OPERATION(compute_block_jacobi,
                       batch_jacobi::compute_block_jacobi);
GKO_REGISTER_OPERATION(transpose_block_jacobi,
                       batch_jacobi::transpose_block_jacobi);


}  // namespace
}  // namespace batch_jacobi


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchJacobi<ValueType, IndexType>::transpose() const
{
    if (parameters_.max_block_size == 1) {
        return this->clone();
        // Since batch scalar jacobi preconditioner does nothing in the external
        // genarate step
    } else {
        auto res = std::unique_ptr<BatchJacobi<ValueType, IndexType>>(
            new BatchJacobi<ValueType, IndexType>(this->get_executor()));
        // BatchJacobi enforces square matrices, so no dim transposition
        // necessary
        res->set_size(this->get_size());
        res->storage_scheme_ = storage_scheme_;
        res->num_blocks_ = num_blocks_;
        res->row_part_of_which_block_info_ = row_part_of_which_block_info_;
        res->blocks_.resize_and_reset(blocks_.get_num_elems());
        res->parameters_ = parameters_;

        const bool to_conjugate = false;

        this->get_executor()->run(batch_jacobi::make_transpose_block_jacobi(
            this->get_num_batch_entries(), this->get_size().at(0)[0],
            num_blocks_, parameters_.max_block_size,
            parameters_.block_pointers.get_const_data(),
            blocks_.get_const_data(), storage_scheme_,
            row_part_of_which_block_info_.get_const_data(),
            res->blocks_.get_data(), to_conjugate));

        return std::move(res);
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchJacobi<ValueType, IndexType>::conj_transpose()
    const
{
    if (parameters_.max_block_size == 1) {
        return this->clone();
        // Since batch scalar jacobi preconditioner does nothing in the external
        // genarate step
    } else {
        auto res = std::unique_ptr<BatchJacobi<ValueType, IndexType>>(
            new BatchJacobi<ValueType, IndexType>(this->get_executor()));
        // BatchJacobi enforces square matrices, so no dim transposition
        // necessary
        res->set_size(this->get_size());
        res->storage_scheme_ = storage_scheme_;
        res->num_blocks_ = num_blocks_;
        res->row_part_of_which_block_info_ = row_part_of_which_block_info_;
        res->blocks_.resize_and_reset(blocks_.get_num_elems());
        res->parameters_ = parameters_;

        const bool to_conjugate = true;

        this->get_executor()->run(batch_jacobi::make_transpose_block_jacobi(
            this->get_num_batch_entries(), this->get_size().at(0)[0],
            num_blocks_, parameters_.max_block_size,
            parameters_.block_pointers.get_const_data(),
            blocks_.get_const_data(), storage_scheme_,
            row_part_of_which_block_info_.get_const_data(),
            res->blocks_.get_data(), to_conjugate));

        return std::move(res);
    }
}

template <typename ValueType, typename IndexType>
void BatchJacobi<ValueType, IndexType>::detect_blocks(
    const size_type num_batch,
    const matrix::Csr<ValueType, IndexType>* first_system)
{
    parameters_.block_pointers.resize_and_reset(first_system->get_size()[0] +
                                                1);
    this->get_executor()->run(batch_jacobi::make_find_blocks(
        first_system, parameters_.max_block_size, num_blocks_,
        parameters_.block_pointers));
    blocks_.resize_and_reset(
        storage_scheme_.compute_storage_space(num_batch, num_blocks_));
}

namespace detail {

template <typename IndexType>
void find_row_is_part_of_which_block(
    const size_type num_blocks, const size_type num_rows,
    const gko::array<IndexType>& block_pointers,
    gko::array<IndexType>& row_part_of_which_block_info)
{
    auto exec = block_pointers.get_executor();
    gko::array<IndexType> block_pointers_ref(exec->get_master());
    block_pointers_ref = block_pointers;
    gko::array<IndexType> row_part_of_which_block_info_ref(exec->get_master(),
                                                           num_rows);
    for (size_type block_idx = 0; block_idx < num_blocks; block_idx++) {
        for (IndexType i = block_pointers_ref.get_const_data()[block_idx];
             i < block_pointers_ref.get_const_data()[block_idx + 1]; i++) {
            row_part_of_which_block_info_ref.get_data()[i] = block_idx;
        }
    }
    row_part_of_which_block_info = row_part_of_which_block_info_ref;
}

}  // namespace detail

template <typename ValueType, typename IndexType>
void BatchJacobi<ValueType, IndexType>::generate_precond(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!system_matrix->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();

    if (parameters_.max_block_size == 1u) {
        // External generate does nothing in case of scalar block jacobi (as the
        // whole generation is done inside the solver kernel)
        num_blocks_ = system_matrix->get_size().at(0)[0];
        blocks_ = gko::array<ValueType>(exec);
        parameters_.block_pointers = gko::array<IndexType>(exec);
        return;
    }

    std::shared_ptr<matrix_type> sys_csr;

    if (auto temp_csr = dynamic_cast<const matrix_type*>(system_matrix)) {
        sys_csr = gko::share(gko::clone(exec, temp_csr));
    } else {
        sys_csr = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(system_matrix)
            ->convert_to(sys_csr.get());
    }

    const auto num_batch = sys_csr->get_num_batch_entries();
    const auto num_rows = sys_csr->get_size().at(0)[0];
    const auto num_nz = sys_csr->get_num_stored_elements() / num_batch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, sys_csr->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, sys_csr->get_const_col_idxs());
    auto sys_vals_view =
        array<ValueType>::const_view(exec, num_nz, sys_csr->get_const_values());
    auto first_sys_csr = gko::share(unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view)));

    if (parameters_.block_pointers.get_data() == nullptr) {
        this->detect_blocks(num_batch, first_sys_csr.get());
    }

    detail::find_row_is_part_of_which_block(num_blocks_, num_rows,
                                            parameters_.block_pointers,
                                            row_part_of_which_block_info_);

    // Note: Storing each block in the same sized matrix and with same stride
    // makes implementation(mainly accessing elements) easy with almost no
    // effect on performance. Note: Row-major order offers advanatge in terms of
    // performance in both preconditioner generation and application for both
    // reference and cuda backend. Note: The pattern blocks in block_pattern are
    // also stored in a similar way.

    // array for storing the common pattern of the diagonal blocks
    gko::array<IndexType> blocks_pattern(
        exec, storage_scheme_.compute_storage_space(1, this->num_blocks_));
    blocks_pattern.fill(static_cast<IndexType>(-1));

    // Since all the matrices in the batch have the same sparisty pattern, it is
    // advantageous to extract the blocks only once instead of repeating
    // computations for each matrix entry. Thus, first, a common pattern for the
    // blocks (corresponding to a batch entry) is extracted and then blocks
    // corresponding to different batch entries are obtained by just filling in
    // values based on the common pattern.

    exec->run(batch_jacobi::make_extract_common_blocks_pattern(
        first_sys_csr.get(), num_blocks_, storage_scheme_,
        parameters_.block_pointers.get_const_data(),
        row_part_of_which_block_info_.get_const_data(),
        blocks_pattern.get_data()));

    exec->run(batch_jacobi::make_compute_block_jacobi(
        sys_csr.get(), parameters_.max_block_size, num_blocks_, storage_scheme_,
        parameters_.block_pointers.get_const_data(),
        blocks_pattern.get_const_data(), blocks_.get_data()));
}


#define GKO_DECLARE_BATCH_JACOBI(ValueType) class BatchJacobi<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_JACOBI);


}  // namespace preconditioner
}  // namespace gko
