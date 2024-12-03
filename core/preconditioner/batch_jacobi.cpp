// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/batch_jacobi.hpp"

#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"


namespace gko {
namespace batch {
namespace preconditioner {
namespace jacobi {


GKO_REGISTER_OPERATION(find_blocks, jacobi::find_blocks);
GKO_REGISTER_OPERATION(extract_common_blocks_pattern,
                       batch_jacobi::extract_common_blocks_pattern);
GKO_REGISTER_OPERATION(compute_block_jacobi,
                       batch_jacobi::compute_block_jacobi);
GKO_REGISTER_OPERATION(find_row_block_map, batch_jacobi::find_row_block_map);
GKO_REGISTER_OPERATION(compute_cumulative_block_storage,
                       batch_jacobi::compute_cumulative_block_storage);


}  // namespace jacobi


template <typename ValueType, typename IndexType>
size_type Jacobi<ValueType, IndexType>::compute_storage_space(
    const size_type num_batch) const noexcept
{
    return (num_blocks_ > 0)
               ? num_batch * (this->get_executor()->copy_val_to_host(
                                 blocks_cumulative_offsets_.get_const_data() +
                                 num_blocks_))
               : size_type{0};
}


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>::Jacobi(std::shared_ptr<const Executor> exec)
    : EnableBatchLinOp<Jacobi>(exec),
      block_pointers_(exec),
      num_blocks_{0},
      blocks_(exec),
      map_block_to_row_(exec),
      blocks_cumulative_offsets_(exec)
{}


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>::Jacobi(
    const Factory* factory, std::shared_ptr<const BatchLinOp> system_matrix)
    : EnableBatchLinOp<Jacobi>(factory->get_executor(),
                               gko::transpose(system_matrix->get_size())),
      parameters_{factory->get_parameters()},
      block_pointers_(std::move(parameters_.block_pointers)),
      num_blocks_{
          block_pointers_.get_size() > 0 ? block_pointers_.get_size() - 1 : 0},
      blocks_(factory->get_executor()),
      map_block_to_row_(factory->get_executor(),
                        system_matrix->get_common_size()[0]),
      blocks_cumulative_offsets_(factory->get_executor(), num_blocks_ + 1)
{
    GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(system_matrix);
    this->generate_precond(system_matrix.get());
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::detect_blocks(
    const gko::matrix::Csr<ValueType, IndexType>* first_system)
{
    this->block_pointers_.resize_and_reset(first_system->get_size()[0] + 1);
    this->get_executor()->run(
        jacobi::make_find_blocks(first_system, parameters_.max_block_size,
                                 num_blocks_, this->block_pointers_));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::generate_precond(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = gko::matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    auto exec = this->get_executor();

    if (parameters_.max_block_size == 1u) {
        // External generate does nothing in case of scalar block jacobi (as the
        // whole generation is done inside the solver kernel)
        num_blocks_ = system_matrix->get_common_size()[0];
        blocks_ = gko::array<ValueType>(exec);
        this->block_pointers_ = gko::array<IndexType>(exec);
        return;
    }

    auto* sys_csr = dynamic_cast<const matrix_type*>(system_matrix);
    std::shared_ptr<const matrix_type> sys_csr_shared_ptr{};

    if (!sys_csr) {
        sys_csr_shared_ptr = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<const matrix_type>>(system_matrix)
            ->convert_to(sys_csr_shared_ptr.get());
        sys_csr = sys_csr_shared_ptr.get();
    }

    const auto num_batch = sys_csr->get_num_batch_items();
    const auto num_rows = sys_csr->get_common_size()[0];
    const auto num_nz = sys_csr->get_num_elements_per_item();

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, sys_csr->get_common_size()[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, sys_csr->get_const_col_idxs());
    auto sys_vals_view =
        array<ValueType>::const_view(exec, num_nz, sys_csr->get_const_values());
    auto first_sys_csr = gko::share(unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view)));

    if (block_pointers_.get_data() == nullptr) {
        block_pointers_.set_executor(exec);
        this->detect_blocks(first_sys_csr.get());
        exec->synchronize();
        blocks_cumulative_offsets_.resize_and_reset(num_blocks_ + 1);
    }

    // cumulative block storage
    exec->run(jacobi::make_compute_cumulative_block_storage(
        num_blocks_, block_pointers_.get_const_data(),
        blocks_cumulative_offsets_.get_data()));

    blocks_.resize_and_reset(this->compute_storage_space(num_batch));

    exec->run(jacobi::make_find_row_block_map(num_blocks_,
                                              block_pointers_.get_const_data(),
                                              map_block_to_row_.get_data()));

    // Note: Row-major order offers advantage in terms of
    // performance in both preconditioner generation and application for both
    // reference and cuda backend.
    // Note: The pattern blocks in block_pattern are
    // also stored in a similar way.

    // array for storing the common pattern of the diagonal blocks
    gko::array<IndexType> block_nnz_idxs(exec, this->compute_storage_space(1));
    block_nnz_idxs.fill(static_cast<IndexType>(-1));

    // Since all the matrices in the batch have the same sparsity pattern, it is
    // advantageous to extract the blocks only once instead of repeating
    // computations for each matrix entry. Thus, first, a common pattern for the
    // blocks (corresponding to a batch entry) is extracted and then blocks
    // corresponding to different batch entries are obtained by just filling in
    // values based on the common pattern.
    exec->run(jacobi::make_extract_common_blocks_pattern(
        first_sys_csr.get(), num_blocks_,
        blocks_cumulative_offsets_.get_const_data(),
        block_pointers_.get_const_data(), map_block_to_row_.get_const_data(),
        block_nnz_idxs.get_data()));

    exec->run(jacobi::make_compute_block_jacobi(
        sys_csr, parameters_.max_block_size, num_blocks_,
        blocks_cumulative_offsets_.get_const_data(),
        block_pointers_.get_const_data(), block_nnz_idxs.get_const_data(),
        blocks_.get_data()));
}


#define GKO_DECLARE_BATCH_JACOBI(_type) class Jacobi<_type, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_WITH_HALF(GKO_DECLARE_BATCH_JACOBI);


}  // namespace preconditioner
}  // namespace batch
}  // namespace gko
