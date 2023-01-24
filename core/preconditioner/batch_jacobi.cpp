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


}  // namespace
}  // namespace batch_jacobi


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchJacobi<ValueType, IndexType>::transpose() const
{
    if (parameters_.max_block_size == 1) {
        return this->clone();
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    // auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
    //     new Jacobi<ValueType, IndexType>(this->get_executor()));
    // // Jacobi enforces square matrices, so no dim transposition necessary
    // res->set_size(this->get_size());
    // res->storage_scheme_ = storage_scheme_;
    // res->num_blocks_ = num_blocks_;
    // res->blocks_.resize_and_reset(blocks_.get_num_elems());
    // res->conditioning_ = conditioning_;
    // res->parameters_ = parameters_;
    // if (parameters_.max_block_size == 1) {
    //     res->blocks_ = blocks_;
    // } else {
    //     this->get_executor()->run(jacobi::make_transpose_jacobi(
    //         num_blocks_, parameters_.max_block_size,
    //         parameters_.storage_optimization.block_wise,
    //         parameters_.block_pointers, blocks_, storage_scheme_,
    //         res->blocks_));
    // }

    // return std::move(res);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchJacobi<ValueType, IndexType>::conj_transpose()
    const
{
    if (parameters_.max_block_size == 1) {
        // Since this preconditioner does nothing in its genarate step,
        //  conjugate transpose only depends on the matrix being
        //  conjugate-transposed.
        return this->clone();
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    // auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
    //     new Jacobi<ValueType, IndexType>(this->get_executor()));
    // // Jacobi enforces square matrices, so no dim transposition necessary
    // res->set_size(this->get_size());
    // res->storage_scheme_ = storage_scheme_;
    // res->num_blocks_ = num_blocks_;
    // res->blocks_.resize_and_reset(blocks_.get_num_elems());
    // res->conditioning_ = conditioning_;
    // res->parameters_ = parameters_;
    // if (parameters_.max_block_size == 1) {
    //     this->get_executor()->run(
    //         jacobi::make_scalar_conj(this->blocks_, res->blocks_));
    // } else {
    //     this->get_executor()->run(jacobi::make_conj_transpose_jacobi(
    //         num_blocks_, parameters_.max_block_size,
    //         parameters_.storage_optimization.block_wise,
    //         parameters_.block_pointers, blocks_, storage_scheme_,
    //         res->blocks_));
    // }

    // return std::move(res);
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

    if (parameters_.skip_sorting != true) {
        sys_csr->sort_by_column_index();
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

    /*  TODO:
       treat row_part_of_which_block_info_
    */
    gko::array<index_type> block_pointers_ref(exec->get_master());
    block_pointers_ref = parameters_.block_pointers;
    gko::array<index_type> row_part_of_which_block_info_ref(exec->get_master(),
                                                            num_rows);
    for (size_type block_idx = 0; block_idx < num_blocks_; block_idx++) {
        for (index_type i = block_pointers_ref.get_const_data()[block_idx];
             i < block_pointers_ref.get_const_data()[block_idx + 1]; i++) {
            row_part_of_which_block_info_ref.get_data()[i] = block_idx;
        }
    }
    row_part_of_which_block_info_ = row_part_of_which_block_info_ref;


    // Note: Storing each block in the same size and stride matrix makes
    // accessing elements/implementation easy with
    // no effect on performance.
    // Note: Row-major order offers advanatge in terms of performnace in both
    // preconditioner generation and application
    // for both reference and cuda backend.
    // Note: The pattern blocks in block_pattern are also stored in the similar
    // way.

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

    // Just for printing
    gko::array<IndexType> blocks_pattern_ref(exec->get_master());
    blocks_pattern_ref = blocks_pattern;

    for (int i = 0; i < num_blocks_; i++) {
        std::cout << "block idx: " << i << std::endl;
        const auto bsize = block_pointers_ref.get_const_data()[i + 1] -
                           block_pointers_ref.get_const_data()[i];
        for (int r = 0; r < bsize; r++) {
            for (int c = 0; c < bsize; c++) {
                std::cout << "block_pattern[" << r << "," << c << "]:"
                          << blocks_pattern_ref.get_const_data()
                                 [storage_scheme_.get_block_offset(i) +
                                  r * storage_scheme_.get_stride() + c]
                          << std::endl;
            }
        }
    }


    exec->run(batch_jacobi::make_compute_block_jacobi(
        sys_csr.get(), num_blocks_, storage_scheme_,
        parameters_.block_pointers.get_const_data(),
        blocks_pattern.get_const_data(), blocks_.get_data()));

    // Just for printing
    gko::array<ValueType> blocks_data_ref(exec->get_master());
    blocks_data_ref = blocks_;

    for (int batch_id = 0; batch_id < num_batch; batch_id++) {
        std::cout << "batch idx: " << batch_id << std::endl << std::endl;
        for (int i = 0; i < num_blocks_; i++) {
            std::cout << "block idx: " << i << std::endl;
            const auto bsize = block_pointers_ref.get_const_data()[i + 1] -
                               block_pointers_ref.get_const_data()[i];
            for (int r = 0; r < bsize; r++) {
                for (int c = 0; c < bsize; c++) {
                    std::cout << "block_data[" << r << "," << c << "]:"
                              << blocks_data_ref.get_const_data()
                                     [storage_scheme_.get_global_block_offset(
                                          num_blocks_, batch_id, i) +
                                      r * storage_scheme_.get_stride() + c]
                              << std::endl;
                }
            }
        }
    }
}


#define GKO_DECLARE_BATCH_JACOBI(ValueType) class BatchJacobi<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_JACOBI);


}  // namespace preconditioner
}  // namespace gko
