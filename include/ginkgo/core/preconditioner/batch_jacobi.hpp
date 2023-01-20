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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {

/**
 * The storage scheme used by batched block-Jacobi blocks.
 *
 * @note All blocks are stored in row-major order as square matrices of size =
 * max_block_size and stride = max_block_size.
 *
 * @note The actual size of each block could be found out using the block
 * pointers array.
 *
 * @note All the blocks corresponding to the first entry in the batch are
 * stored first, then all the blocks corresponding to the second entry and so
 * on...
 *
 * @ingroup batch_jacobi
 */
struct batched_blocks_storage_scheme {
    batched_blocks_storage_scheme() = default;

    batched_blocks_storage_scheme(const size_type max_block_size)
        : max_block_size_{max_block_size}
    {}

    size_type max_block_size_;

    /**
     * Computes the storage space required for the requested number of blocks.
     *
     * @param batch_size the number of entries in the batch
     * @param num_blocks  the number of blocks in a batched matrix entry
     *
     * @return the total memory (as the number of elements) that need to be
     *         allocated for the scheme
     *
     * @note  To simplify using the method in situations where the number of
     *        blocks is not known, for a special input `size_type{} - 1`
     *        the method returns `0` to avoid overallocation of memory.
     */
    GKO_ATTRIBUTES size_type compute_storage_space(
        const size_type batch_size, const size_type num_blocks) const noexcept
    {
        return (num_blocks + 1 == size_type{0})
                   ? size_type{0}
                   : batch_size * num_blocks * max_block_size_ *
                         max_block_size_;
    }

    /**
     * Returns the offset of the batch with id "batch_id"
     *
     * @param num_blocks the number of blocks in a batched matrix entry
     * @param batch_id the index of the batch entry in the batch
     *
     * @return the offset of the group belonging to block with ID `block_id`
     */
    GKO_ATTRIBUTES size_type get_batch_offset(
        const size_type num_blocks, const size_type batch_id) const noexcept
    {
        return batch_id * num_blocks * max_block_size_ * max_block_size_;
    }

    /**
     * Returns the (local) offset of the block with id: "block_id" within its
     * batch entry
     *
     * @param block_id the id of the block from the perspective of individual
     * batch entry
     *
     * @return the offset of the block with id: `block_id` within its batch
     * entry
     */
    GKO_ATTRIBUTES size_type
    get_block_offset(const size_type block_id) const noexcept
    {
        return block_id * max_block_size_ * max_block_size_;
    }

    /**
     * Returns the global offset of the block which belongs to the batch entry
     * with index = batch_id and has local id = "block_id" within its batch
     * entry
     *
     * @param num_blocks the number of blocks in a batched matrix entry
     * @param batch_id the index of the batch entry in the batch
     * @param block_id the id of the block from the perspective of individual
     * batch entry
     *
     * @return the global offset of the block which belongs to the batch entry
     * with index = batch_id and has local id = "block_id" within its batch
     * entry
     */
    GKO_ATTRIBUTES size_type get_global_block_offset(
        const size_type num_blocks, const size_type batch_id,
        const size_type block_id) const noexcept
    {
        return this->get_batch_offset(num_blocks, batch_id) +
               this->get_block_offset(block_id);
    }

    /**
     * Returns the stride between the rows of the block.
     *
     * @return stride between rows of the block
     */
    GKO_ATTRIBUTES size_type get_stride() const noexcept
    {
        return max_block_size_;
    }
};


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of the source operator.
 *
 * Note: Batched Preconditioners do not support user facing apply.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup jacobi
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchJacobi : public EnableBatchLinOp<BatchJacobi<ValueType, IndexType>>,
                    public BatchTransposable {
    friend class EnableBatchLinOp<BatchJacobi>;
    friend class EnablePolymorphicObject<BatchJacobi, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchJacobi>::convert_to;
    using EnableBatchLinOp<BatchJacobi>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::BatchCsr<ValueType, IndexType>;

    /**
     * Returns the storage scheme used for storing Batched Jacobi blocks.
     *
     * @return the storage scheme used for storing Batched Jacobi blocks
     *
     */
    const batched_blocks_storage_scheme& get_storage_scheme() const noexcept
    {
        return storage_scheme_;
    }

    /**
     *  Returns the block pointers.
     *  @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1).
     *  @return the block pointers
     */
    const index_type* get_const_block_pointers() const noexcept
    {
        if (parameters_.max_block_size == 1) {
            return nullptr;
        }
        return parameters_.block_pointers.get_const_data();
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
     *
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }


    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of the block, which belongs to the batch entry with
     * index = batch_id and has local id = "block_id" within its batch entry is
     * stored at the address = get_const_block_pointers() +
     * storage_scheme.get_global_block_offset(num_blocks, batch_id, block_id) +
     * i * storage_scheme.get_stride() + j
     *
     * @note Returns nullptr in case of a scalar jacobi preconditioner
     * (max_block_size = 1). The blocks array is empty in case of scalar jacobi
     *  preconditioner as the preconditioner is generated inside the batched
     * solver kernel.
     *
     * @return the pointer to the memory used for storing the block data
     *
     */
    const value_type* get_const_blocks() const noexcept
    {
        if (parameters_.max_block_size == 1) {
            return nullptr;
        }
        return blocks_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @note Returns 0 in case of scalar jacobi preconditioner as the
     * preconditioner is generated inside the batched solver kernels, hence,
     * blocks array storage is not required.
     *
     * @return the number of elements explicitly stored in the matrix.
     */
    size_type get_num_stored_elements() const noexcept
    {
        if (parameters_.max_block_size == 1) {
            return 0;
        }
        return blocks_.get_num_elems();
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
         */
        uint32 GKO_FACTORY_PARAMETER_SCALAR(max_block_size, 32u);

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

        /**
         * @brief `true` means it is known that the matrix given to this
         *        factory will be sorted first by row, then by column index,
         *        `false` means it is unknown or not sorted, so an additional
         *        sorting step will be performed during the preconditioner
         *        generation (it will not change the matrix given).
         *        The matrix must be sorted for this preconditioner to work.
         *
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this preconditioner might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchJacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

protected:
    /**
     * Creates an empty Jacobi preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchJacobi(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchJacobi>(exec), num_blocks_{}, blocks_(exec)
    {
        parameters_.block_pointers.set_executor(this->get_executor());
    }

    /**
     * Creates a Jacobi preconditioner from a matrix using a Jacobi::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchJacobi(const Factory* factory,
                         std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchJacobi>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          num_blocks_{parameters_.block_pointers.get_num_elems() - 1},
          storage_scheme_{
              batched_blocks_storage_scheme(parameters_.max_block_size)},
          blocks_(factory->get_executor(),
                  storage_scheme_.compute_storage_space(
                      system_matrix->get_num_batch_entries(),
                      parameters_.block_pointers.get_num_elems() - 1))
    {
        parameters_.block_pointers.set_executor(this->get_executor());
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix);
        this->generate_precond(lend(system_matrix));
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate_precond(const BatchLinOp* system_matrix);

    // Since there is no guarantee that the complete generation of the
    // preconditioner would occur outside the solver kernel, that is in the
    // external generate step, there is no logic of implementing "apply" for
    // batched preconditioners
    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override
        GKO_BATCHED_NOT_SUPPORTED(
            "batched preconditioners do not support apply");

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override
        GKO_BATCHED_NOT_SUPPORTED(
            "batched preconditioners do not support apply");

private:
    /**
     * Detects the diagonal blocks and allocates the memory needed to store the
     * preconditioner.
     */
    void detect_blocks(const size_type num_batch,
                       const matrix::Csr<ValueType, IndexType>* system_matrix);

    batched_blocks_storage_scheme storage_scheme_;
    size_type num_blocks_;
    array<value_type> blocks_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_JACOBI_HPP_
