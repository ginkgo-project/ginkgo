/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_PRECONDITIONER_JACOBI_HPP_
#define GKO_CORE_PRECONDITIONER_JACOBI_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace preconditioner {


// TODO: replace this with a custom accessor
/**
 * Defines the parameters of the interleaved block storage scheme used by
 * block-Jacobi blocks.
 *
 * @tparam IndexType  type used for storing indices of the matrix
 */
template <typename IndexType>
struct block_interleaved_storage_scheme {
    /**
     * The offset between consecutive blocks within the group.
     */
    IndexType block_offset;

    /**
     * The offset between two block groups.
     */
    IndexType group_offset;

    /**
     * Then base 2 power of the group.
     *
     * I.e. the group contains `1 << group_power` elements.
     */
    uint32 group_power;

    /**
     * Returns the number of elements in the group.
     *
     * @return the number of elements in the group
     */
    GKO_ATTRIBUTES IndexType get_group_size() const noexcept
    {
        return one<IndexType>() << group_power;
    }

    /**
     * Computes the storage space required for the requested number of blocks.
     *
     * @param num_blocks  the total number of blocks that needs to be stored
     *
     * @return the total memory (as the number of elements) that need to be
     *         allocated for the scheme
     */
    GKO_ATTRIBUTES IndexType compute_storage_space(IndexType num_blocks) const
        noexcept
    {
        return (num_blocks + 1 == size_type{0})
                   ? size_type{0}
                   : ceildiv(num_blocks, this->get_group_size()) * group_offset;
    }

    /**
     * Returns the offset of the group belonging to the block with the given ID.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the group belonging to block with ID `block_id`
     */
    GKO_ATTRIBUTES IndexType get_group_offset(IndexType block_id) const noexcept
    {
        return group_offset * (block_id >> group_power);
    }

    /**
     * Returns the offset of the block with the given ID within its group.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the block with ID `block_id` within its group
     */
    GKO_ATTRIBUTES IndexType get_block_offset(IndexType block_id) const noexcept
    {
        return block_offset * (block_id & (this->get_group_size() - 1));
    }

    /**
     * Returns the offset of the block with the given ID.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the block with ID `block_id`
     */
    GKO_ATTRIBUTES IndexType get_global_block_offset(IndexType block_id) const
        noexcept
    {
        return this->get_group_offset(block_id) +
               this->get_block_offset(block_id);
    }

    /**
     * Returns the stride between columns of the block.
     *
     * @return stride between columns of the block
     */
    GKO_ATTRIBUTES IndexType get_stride() const noexcept
    {
        return block_offset << group_power;
    }
};


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of the source operator.
 *
 * The Jacobi class implements the inversion of the diagonal blocks using
 * Gauss-Jordan elimination with column pivoting, and stores the inverse
 * explicitly in a customized format.
 *
 * If the diagonal blocks of the matrix are not explicitly set by the user, the
 * implementation will try to automatically detect the blocks by first finding
 * the natural blocks of the matrix, and then applying the supervariable
 * agglomeration procedure on them. However, if problem-specific knowledge
 * regarding the block diagonal structure is available, it is usually beneficial
 * to explicitly pass the starting rows of the diagonal blocks, as the block
 * detection is merely a heuristic and cannot perfectly detect the diagonal
 * block structure. The current implementation supports blocks of up to 32 rows
 * / columns.
 *
 * The implementation also includes an improved, adaptive version of the
 * block-Jacobi preconditioner, which can store some of the blocks in lower
 * precision and thus improve the performance of preconditioner application by
 * reducing the amount of memory transfers. This variant can be enabled by
 * setting the Jacobi::Factory's `storage_optimization` parameter.  Refer to the
 * documentation of the parameter for more details.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 *
 * @note The current implementation supports blocks of up to 32 rows / columns.
 * @note When using the adaptive variant, there may be a trade-off in terms of
 *       slightly longer preconditioner generation due to extra work required to
 *       detect the optimal precision of the blocks.
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Jacobi : public EnableLinOp<Jacobi<ValueType, IndexType>>,
               public ConvertibleTo<matrix::Dense<ValueType>>,
               public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableLinOp<Jacobi>;
    friend class EnablePolymorphicObject<Jacobi, LinOp>;

public:
    using EnableLinOp<Jacobi>::convert_to;
    using EnableLinOp<Jacobi>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    /**
     * Returns the number of blocks of the operator.
     *
     * @return the number of blocks of the operator
     *
     * TODO: replace with ranges
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }

    /**
     * Returns the storage scheme used for storing Jacobi blocks.
     *
     * @return the storage scheme used for storing Jacobi blocks
     *
     * TODO: replace with ranges
     */
    const block_interleaved_storage_scheme<index_type> &get_storage_scheme()
        const noexcept
    {
        return storage_scheme_;
    }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * stride + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     *
     * TODO: replace with ranges
     */
    const value_type *get_blocks() const noexcept
    {
        return blocks_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return blocks_.get_num_elems();
    }

    void convert_to(matrix::Dense<value_type> *result) const override;

    void move_to(matrix::Dense<value_type> *result) override;

    void write(mat_data &data) const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Maximal size of diagonal blocks.
         *
         * @note This value has to be between 1 and 32.
         */
        uint32 GKO_FACTORY_PARAMETER(max_block_size, 32u);

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
        gko::Array<index_type> GKO_FACTORY_PARAMETER(block_pointers);

    private:
        // See documentation of storage_optimization parameter for details about
        // this class
        struct storage_optimization_type {
            storage_optimization_type(precision_reduction p)
                : is_block_wise{false}, of_all_blocks{p}
            {}

            storage_optimization_type(
                const Array<precision_reduction> &block_wise_opt)
                : is_block_wise{true}, block_wise{block_wise_opt}
            {}

            storage_optimization_type(
                Array<precision_reduction> &&block_wise_opt)
                : is_block_wise{true}, block_wise{std::move(block_wise_opt)}
            {}

            operator precision_reduction() { return of_all_blocks; }

            bool is_block_wise;
            precision_reduction of_all_blocks;
            gko::Array<precision_reduction> block_wise;
        };

    public:
        /**
         * The precisions to use for the blocks of the matrix.
         *
         * This parameter can either be a single instance of precision_reduction
         * or an Array of precision_reduction values. If set to
         * `precision_reduction(0, 0)` (this is the default), a regular
         * full-precision block-Jacobi will be used. Any other value (or an
         * Array of values) will map to the adaptive variant.
         *
         * The best starting point when evaluating the potential of the adaptive
         * version is to set this parameter to
         * `precision_reduction::autodetect()`. This option will cause the
         * preconditioner to reduce the memory transfer volume as much as
         * possible, while trying to maintain the quality of the preconditioner
         * similar to that of the full precision block-Jacobi.
         *
         * For finer control, specific instances of precision_reduction can be
         * used. Supported values are `precision_reduction(0, 0)`,
         * `precision_reduction(0, 1)` and `precision_reduction(0, 2)`. Any
         * other value will have the same effect as `precision_reduction(0, 0)`.
         *
         * If the ValueType template parameter is set to `double` (or the
         * complex variant `std::complex<double>`), `precision_reduction(0, 0)`
         * will use IEEE double precision for preconditioner storage,
         * `precision_reduction(0, 1)` will use IEEE single precision, and
         * `precision_reduction(0, 2)` will use IEEE half precision.
         *
         * It ValueType is set to `float` (or `std::complex<float>`),
         * `precision_reduction(0, 0)` will use IEEE single precision for
         * preconditioner storage, and both `precision_reduction(0, 1)` and
         * `precision_reduction(0, 2)` will use IEEE half precision.
         *
         * Instead of specifying the same precision for all blocks, the
         * precision of the elements can be specified on per-block basis by
         * passing an array of precision_reduction objects. All values discussed
         * above are supported, with the same meaning. It is worth mentioning
         * that a value of `precision_reduction::autodetect()` will cause
         * autodetection on the per-block basis, so blocks whose precisions are
         * autodetected can end up having different precisions once the
         * preconditioner is generated. The detected precision generally depends
         * on the conditioning of the block.
         *
         * If the number of diagonal blocks is larger than the number of
         * elements in the passed Array, the entire Array will be replicated
         * until enough values are available. For example, if the original array
         * contained two precisions `(x, y)` and the preconditioner contains 5
         * blocks, the array will be transformed into `(x, y, x, y, x)` before
         * generating the preconditioner. As a consequence, specifying a single
         * value for this property is exactly equivalent to specifying an array
         * with a single element set to that value.
         *
         * Once an instance of the Jacobi linear operator is generated, the
         * precisions used for the blocks can be obtained by reading this
         * property. Whether the parameter was set to a single value or to an
         * array of values can be queried by reading the
         * `storage_optimization.is_block_wise` boolean sub-property. If it is
         * set to `false`, the precision used for all blocks can be obtained
         * using `storage_optimization.of_all_blocks` or by casting
         * `storage_optimization` to `precision_reduction`. Independently of the
         * value of `storage_optimization.is_block_wise`, the
         * `storage_optimization.block_wise` property will return an array of
         * precisions used for each block. All values set to
         * `precision_reduction::autodetect()` will be replaced with the value
         * representing the precision used for the corresponding block.
         * If the non-adaptive version of Jacobi is used, the
         * `storage_optimization.block_wise` Array will be empty.
         */
        storage_optimization_type GKO_FACTORY_PARAMETER(
            storage_optimization, precision_reduction(0, 0));
    };
    GKO_ENABLE_LIN_OP_FACTORY(Jacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Creates an empty Jacobi preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Jacobi(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Jacobi>(exec), blocks_(exec)
    {
        parameters_.block_pointers.set_executor(exec);
        parameters_.storage_optimization.block_wise.set_executor(exec);
    }

    /**
     * Creates a Jacobi preconditioner from a matrix using a Jacobi::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Jacobi(const Factory *factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Jacobi>(factory->get_executor(),
                              transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          storage_scheme_{compute_storage_scheme(parameters_.max_block_size)},
          num_blocks_{parameters_.block_pointers.get_num_elems() - 1},
          blocks_(factory->get_executor(),
                  storage_scheme_.compute_storage_space(
                      parameters_.block_pointers.get_num_elems() - 1))
    {
        if (parameters_.max_block_size >= 32 ||
            parameters_.max_block_size < 1) {
            NOT_SUPPORTED(this);
        }
        parameters_.block_pointers.set_executor(this->get_executor());
        parameters_.storage_optimization.block_wise.set_executor(
            this->get_executor());
        this->generate(lend(system_matrix));
    }

    /**
     * Stride between two columns of a block (as number of elements).
     *
     * Should be a multiple of cache line size for best performance.
     */
    static constexpr size_type max_block_stride_ = 32;

    /**
     * Returns the smallest power of 2 at least as larger as the input.
     *
     * @param n  a number
     *
     * @return a power of two at least as large as `n`
     */
    static size_type get_larger_power(size_type n) noexcept
    {
        size_type res = 1;
        while (res < n) res *= 2;
        return res;
    }

    /**
     * Returns the base-2 logarithm of `n` rounded down to the nearest integer.
     */
    static uint32 get_log2(size_type n) noexcept
    {
        for (auto r = uint32{0};; ++r) {
            if ((size_type{1} << (r + 1)) > n) {
                return r;
            }
        }
    }

    /**
     * Computes the storage scheme suitable for storing blocks of a given
     * maximum size.
     *
     * @param max_block_size  the maximum size of the blocks
     *
     * @return a suitable storage scheme
     */
    static block_interleaved_storage_scheme<index_type> compute_storage_scheme(
        uint32 max_block_size) noexcept
    {
        const auto group_size = static_cast<uint32>(
            max_block_stride_ / get_larger_power(max_block_size));
        const auto block_offset = max_block_size;
        const auto block_stride = group_size * block_offset;
        const auto group_offset = max_block_size * block_stride;
        return {static_cast<index_type>(block_offset),
                static_cast<index_type>(group_offset), get_log2(group_size)};
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate(const LinOp *system_matrix);

    /**
     * Detects the diagonal blocks and allocates the memory needed to store the
     * preconditioner.
     *
     * @param system_matrix  the source matrix whose diagonal block pattern is
     *                       to be detected
     */
    void detect_blocks(const matrix::Csr<ValueType, IndexType> *system_matrix);

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    block_interleaved_storage_scheme<index_type> storage_scheme_{};
    size_type num_blocks_;
    Array<value_type> blocks_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_JACOBI_HPP_
