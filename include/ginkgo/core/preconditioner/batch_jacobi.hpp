// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace batch {
namespace preconditioner {


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks (stored in a dense row major fashion) of the
 * source operator.
 *
 * With the batched preconditioners, it is required that all items in the batch
 * have the same sparsity pattern. The detection of the blocks and the block
 * pointers require that the sparsity pattern of all the items be the same.
 * Other cases is undefined behaviour. The input batch matrix must be in
 * batch::Csr matrix format or must be convertible to batch::Csr matrix format.
 * The block detection algorithm and the conversion to dense blocks kernels
 * require this assumption.
 *
 * @note In a fashion similar to the non-batched Jacobi preconditioner, the
 * maximum possible size of the diagonal blocks is equal to the maximum warp
 * size on the device (32 for NVIDIA GPUs, 64 for AMD GPUs).
 *
 * @tparam ValueType  value precision of matrix elements
 * @tparam IndexType  index precision of matrix elements
 *
 * @ingroup jacobi
 * @ingroup precond
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Jacobi final : public EnableBatchLinOp<Jacobi<ValueType, IndexType>> {
    friend class EnableBatchLinOp<Jacobi>;
    friend class EnablePolymorphicObject<Jacobi, BatchLinOp>;

public:
    using EnableBatchLinOp<Jacobi>::convert_to;
    using EnableBatchLinOp<Jacobi>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = batch::matrix::Csr<ValueType, IndexType>;

    /**
     *  Returns the block pointers.
     *  @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1).
     *  @return the block pointers
     */
    const index_type* get_const_block_pointers() const noexcept
    {
        return parameters_.block_pointers.get_const_data();
    }

    /**
     * Returns information about which blocks are the rows of the matrix part
     * of.
     *
     * @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1).
     */
    const index_type* get_const_row_block_map_info() const noexcept
    {
        return row_block_map_info_.get_const_data();
    }

    /**
     *  Returns the cumulative blocks storage array
     *
     *  @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1).
     */
    const index_type* get_const_blocks_cumulative_storage() const noexcept
    {
        return blocks_cumulative_storage_.get_const_data();
    }

    /**
     * Returns the max block size.
     *
     * @return the max block size
     */
    uint32 get_max_block_size() const noexcept
    {
        return parameters_.max_block_size;
    }

    /**
     * Returns the number of blocks in an individual batch entry.
     *
     * @return the number of blocks in an individual batch entry.
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of the block, which belongs to the batch entry with
     * index = batch_id and has local id = "block_id" within its batch entry is
     * stored at the address = get_const_blocks() +
     * storage_scheme.get_global_block_offset(batch_id, num_blocks, block_id,
     * cumulative_blocks_storage) + i * storage_scheme.get_stride(block_id,
     * block_pointers) + j
     *
     * @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1). The blocks array is empty in case of scalar jacobi
     *  preconditioner as the preconditioner is generated inside the batched
     * solver kernel.
     *
     * @return the pointer to the memory used for storing the block data
     */
    const value_type* get_const_blocks() const noexcept
    {
        return blocks_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the dense blocks.
     *
     * @note Returns 0 in case of scalar jacobi preconditioner as the
     * preconditioner is generated inside the batched solver kernels, hence,
     * blocks array storage is not required.
     *
     * @return the number of elements explicitly stored in the dense blocks.
     */
    size_type get_num_stored_elements() const noexcept
    {
        if (parameters_.max_block_size == 1) {
            return 0;
        }
        return blocks_.get_size();
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Maximal size of diagonal blocks.
         *
         * @note This value has to be between 1 and 32 (NVIDIA)/64 (AMD). For
         * efficiency, when the max_block_size is set to 1, specialized kernels
         * are used and the additional objects (block_ptrs etc) are set to null
         * values.
         *
         * @note Unlike the regular block Jacobi preconditioner, for the batched
         * preconditioner, smaller blocks are more efficient, as the matrices
         * themselves are considerably smaller.
         */
        uint32 GKO_FACTORY_PARAMETER_SCALAR(max_block_size, 8u);

        /**
         * Starting (row / column) indexes of individual blocks.
         *
         * An index past the last block has to be supplied as the last value.
         * I.e. the size of the array has to be the number of blocks plus 1,
         * where the first value is 0, and the last value is the number of
         * rows / columns of the matrix.
         *
         * @note Even if not set explicitly, this parameter will be set to
         *       automatically detected values once the preconditioner is
         *       generated.
         * @note If the parameter is set automatically, the size of the array
         *       does not correlate to the number of blocks, and is
         *       implementation defined. To obtain the number of blocks `n` use
         *       Jacobi::get_num_blocks(). The starting indexes of the blocks
         *       are stored in the first `n+1` values of this array.
         * @note If the block-diagonal structure can be determined from the
         *       problem characteristics, it may be beneficial to pass this
         *       information specifically via this parameter, as the
         *       autodetection procedure is only a rough approximation of the
         *       true block structure.
         * @note The maximum block size set by the max_block_size parameter
         *       has to be respected when setting this parameter. Failure to do
         *       so will lead to undefined behavior.
         */
        gko::array<index_type> GKO_FACTORY_PARAMETER_VECTOR(block_pointers,
                                                            nullptr);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(Jacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

private:
    explicit Jacobi(std::shared_ptr<const Executor> exec);

    explicit Jacobi(const Factory* factory,
                    std::shared_ptr<const BatchLinOp> system_matrix);

    void generate_precond(const BatchLinOp* const system_matrix);

    size_type compute_storage_space(const size_type num_batch) const noexcept
    {
        return (num_blocks_ > 0)
                   ? num_batch *
                         (this->get_executor()->copy_val_to_host(
                             blocks_cumulative_storage_.get_const_data() +
                             num_blocks_))
                   : size_type{0};
    }

    void detect_blocks(
        const size_type num_batch,
        const gko::matrix::Csr<ValueType, IndexType>* system_matrix);

    size_type num_blocks_;
    array<value_type> blocks_;
    array<index_type> row_block_map_info_;
    array<index_type> blocks_cumulative_storage_;
};


}  // namespace preconditioner
}  // namespace batch
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_
