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

#ifndef GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
#define GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace preconditioner {


template <typename, typename>
class BlockJacobiFactory;


template <typename, typename>
class AdaptiveBlockJacobiFactory;


namespace detail {


template <typename T>
struct value_type;

template <template <typename, typename> class Op, typename ValueType,
          typename IndexType>
struct value_type<Op<ValueType, IndexType>> {
    using type = ValueType;
};


template <typename T>
struct index_type;

template <template <typename, typename> class Op, typename ValueType,
          typename IndexType>
struct index_type<Op<ValueType, IndexType>> {
    using type = IndexType;
};


};  // namespace detail


/**
 * This is an intermediate class which implements common methods of block-Jacobi
 * preconditioners.
 *
 * See the documentation for children of this class for more details.
 *
 * @tparam ConcreteBlockJacobi  the concrete block-Jacobi preconditioner whose
 *                              common methods are implemented by this class
 */
template <typename ConcreteBlockJacobi>
class BasicBlockJacobi : public EnableLinOp<ConcreteBlockJacobi> {
public:
    using EnableLinOp<ConcreteBlockJacobi>::convert_to;
    using EnableLinOp<ConcreteBlockJacobi>::move_to;

    using value_type = typename detail::value_type<ConcreteBlockJacobi>::type;
    using index_type = typename detail::index_type<ConcreteBlockJacobi>::type;

    /**
     * Returns the number of blocks of the operator.
     *
     * @return the number of blocks of the operator
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }

    /**
     * Returns the maximum allowed block size of each block.
     *
     * @return the maximum allowed block size of each block
     */
    uint32 get_max_block_size() const noexcept { return max_block_size_; }

    /**
     * Returns the array of pointers to the start of diagonal blocks.
     *
     * @return the array of pointers to the start of diagonal blocks
     */
    index_type *get_block_pointers() noexcept
    {
        return block_pointers_.get_data();
    }

    /**
     * Returns the array of pointers to the start of diagonal blocks.
     *
     * @return the array of pointers to the start of diagonal blocks
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_block_pointers() const noexcept
    {
        return block_pointers_.get_const_data();
    }

    /**
     * Returns the stride used for storing the blocks.
     *
     * @return the stride used for storing the blocks
     */
    size_type get_stride() const noexcept { return max_block_size_; }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * stride + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     */
    value_type *get_blocks() noexcept { return blocks_.get_data(); }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * stride + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_blocks() const noexcept
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

protected:
    explicit BasicBlockJacobi(std::shared_ptr<const Executor> exec)
        : EnableLinOp<ConcreteBlockJacobi>(exec),
          block_pointers_(exec),
          blocks_(exec)
    {}

    BasicBlockJacobi(std::shared_ptr<const Executor> exec,
                     const LinOp *system_matrix, uint32 max_block_size,
                     const Array<index_type> &block_pointers)
        : EnableLinOp<ConcreteBlockJacobi>(
              exec, transpose(system_matrix->get_size())),
          num_blocks_(block_pointers.get_num_elems() - 1),
          max_block_size_(max_block_size),
          block_pointers_(block_pointers),
          blocks_(exec, this->get_size()[1] * max_block_size)
    {
        block_pointers_.set_executor(this->get_executor());
    }

protected:
    size_type num_blocks_{};
    uint32 max_block_size_{};
    Array<index_type> block_pointers_;
    Array<value_type> blocks_;
};


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of another operator.
 *
 * The BlockJacobi class implements this inversion of the diagonal blocks using
 * Gauss-Jordan elimination with column pivoting, and stores the inverse
 * explicitly in a customized format.
 *
 * If the diagonal blocks of the matrix are not explicitly set by the user, the
 * implementation will try to automatically detect the blocks by first finding
 * the natural blocks of the matrix, and then applying the supervariable
 * agglomeration procedure on these blocks.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobi : public BasicBlockJacobi<BlockJacobi<ValueType, IndexType>>,
                    public ConvertibleTo<matrix::Dense<ValueType>>,
                    public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableLinOp<BlockJacobi>;
    friend class EnablePolymorphicObject<BlockJacobi, LinOp>;
    friend class BlockJacobiFactory<ValueType, IndexType>;
    friend class AdaptiveBlockJacobiFactory<ValueType, IndexType>;

public:
    using BasicBlockJacobi<BlockJacobi>::convert_to;
    using BasicBlockJacobi<BlockJacobi>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    void convert_to(matrix::Dense<value_type> *result) const override;

    void move_to(matrix::Dense<value_type> *result) override;

    void write(mat_data &data) const override;

protected:
    using BasicBlockJacobi<BlockJacobi>::BasicBlockJacobi;

    BlockJacobi(std::shared_ptr<const Executor> exec,
                const LinOp *system_matrix, uint32 max_block_size,
                const Array<index_type> &block_pointers)
        : BasicBlockJacobi<BlockJacobi>(exec, system_matrix, max_block_size,
                                        block_pointers)
    {
        this->generate(system_matrix);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void generate(const LinOp *system_matrix);
};


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of another operator.
 *
 * This is a variant of the BlockJacobi preconditioner which can use lower
 * precision to store well-conditioned diagonal blocks, and thus improve the
 * performance of preconditioner application by reducing the amount of memory
 * that has to be read to apply the precondidionter.
 *
 * However, there is a trade-off in terms of longer preconditioner generation
 * due to extra work required to compute the condition numbers. This step is
 * necessary to preserve the regularity of the diagonal blocks.
 *
 * @tparam ValueType  highest precision of matrix elements, the precision of
 *                    vectors used for the apply() method and the precision in
 *                    which all computation is performed
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AdaptiveBlockJacobi
    : public BasicBlockJacobi<AdaptiveBlockJacobi<ValueType, IndexType>>,
      public ConvertibleTo<matrix::Dense<ValueType>>,
      public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableLinOp<AdaptiveBlockJacobi>;
    friend class EnablePolymorphicObject<AdaptiveBlockJacobi, LinOp>;
    friend class BlockJacobiFactory<ValueType, IndexType>;
    friend class AdaptiveBlockJacobiFactory<ValueType, IndexType>;

public:
    using BasicBlockJacobi<AdaptiveBlockJacobi>::convert_to;
    using BasicBlockJacobi<AdaptiveBlockJacobi>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    /**
     * This type is used to store data about precision of diagonal blocks.
     */
    enum precision {
        /**
         * Marks that double precision is used for the block.
         */
        double_precision,

        /**
         * Marks that single precision is used for the block.
         */
        single_precision,

        /**
         * Marks that half precision is used for the block.
         */
        half_precision,

        /**
         * The precision of the block will be determined automatically during
         * generation phase, based on the condition number of the block.
         */
        best_precision
    };

    void convert_to(matrix::Dense<value_type> *result) const override;

    void move_to(matrix::Dense<value_type> *result) override;

    void write(mat_data &data) const override;

    /**
     * Returns the precisions of diagonal blocks.
     *
     * @return precisions of diagonal blocks
     */
    precision *get_block_precisions() noexcept
    {
        return block_precisions_.get_data();
    }

    /**
     * Returns the precisions of diagonal blocks.
     *
     * @return precisions of diagonal blocks
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const precision *get_const_block_precisions() const noexcept
    {
        return block_precisions_.get_const_data();
    }

protected:
    AdaptiveBlockJacobi(std::shared_ptr<const Executor> exec)
        : BasicBlockJacobi<AdaptiveBlockJacobi>(exec), block_precisions_(exec)
    {}

    AdaptiveBlockJacobi(std::shared_ptr<const Executor> exec,
                        const LinOp *system_matrix, uint32 max_block_size,
                        const Array<index_type> &block_pointers,
                        const Array<precision> &block_precisions)
        : BasicBlockJacobi<AdaptiveBlockJacobi>(exec, system_matrix,
                                                max_block_size, block_pointers),
          block_precisions_(block_precisions)
    {
        this->generate(system_matrix);
    }

    void generate(const LinOp *system_matrix);

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    Array<precision> block_precisions_;
};


/**
 * This is an intermediate class which implements common methods of block-Jacobi
 * preconditioner factories.
 *
 * See the documentation for children of this class for more details.
 *
 * @tparam ConcreteBlockJacobiFactory  the concrete block-Jacobi factory whose
 *                                     common methods are implemented by this
 *                                     class
 */
template <typename ConcreteBlockJacobiFactory>
class BasicBlockJacobiFactory
    : public EnablePolymorphicObject<ConcreteBlockJacobiFactory, LinOpFactory> {
    friend class EnablePolymorphicObject<ConcreteBlockJacobiFactory,
                                         LinOpFactory>;

public:
    using value_type =
        typename detail::value_type<ConcreteBlockJacobiFactory>::type;
    using index_type =
        typename detail::index_type<ConcreteBlockJacobiFactory>::type;

    /**
     * Creates a new block-Jacobi factory.
     *
     * @param exec  the executor where the block-Jacobi preconditioner will be
     *              stored
     * @param max_block_size  maximum allowed size of diagonal blocks
     *                        (a tighter bound on the size can positively affect
     *                        the performance of the preconditioner generation
     *                        and application)
     *
     * @return a unique pointer to the newly created factory
     */
    static std::unique_ptr<ConcreteBlockJacobiFactory> create(
        std::shared_ptr<const Executor> exec, uint32 max_block_size)
    {
        return std::unique_ptr<ConcreteBlockJacobiFactory>(
            new ConcreteBlockJacobiFactory(std::move(exec), max_block_size));
    }

    /**
     * Returns the maximum allowed size of diagonal blocks.
     *
     * @return the maximum allowed size of diagonal blocks
     */
    uint32 get_max_block_size() const noexcept { return max_block_size_; }

    /**
     * Sets the array of block pointer which will be used to determine the
     * blocks for the matrix.
     *
     * @param block_pointers  The array of block pointer, the value at position
     *                        `i` of the array should be set to the row where
     *                        the `i`-th block starts. In addition, the last
     *                        value of the array should be set to the number of
     *                        rows of the matrix. Thus, the total number of
     *                        blocks is `block_pointers.get_num_elems() - 1`.
     */
    void set_block_pointers(const Array<index_type> &block_pointers)
    {
        block_pointers_ = block_pointers;
    }

    /**
     * @copydoc set_block_pointers(const Array<index_type> &)
     */
    void set_block_pointers(Array<index_type> &&block_pointers)
    {
        block_pointers_ = std::move(block_pointers);
    }

    /**
     * Returns a reference to the array of block pointers.
     *
     * @return the array of block pointers
     */
    const Array<index_type> &get_block_pointers() const noexcept
    {
        return block_pointers_;
    }

protected:
    BasicBlockJacobiFactory(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<ConcreteBlockJacobiFactory, LinOpFactory>(
              exec),
          max_block_size_{},
          block_pointers_(exec)
    {}

    BasicBlockJacobiFactory(std::shared_ptr<const Executor> exec,
                            uint32 max_block_size)
        : EnablePolymorphicObject<ConcreteBlockJacobiFactory, LinOpFactory>(
              exec),
          max_block_size_(max_block_size),
          block_pointers_(exec)
    {}

    uint32 max_block_size_;
    Array<index_type> block_pointers_;
};


/**
 * This factory is used to create a block-Jacobi preconditioner from the
 * operator A, by inverting the diagonal blocks of the operator.
 *
 * The factory generates a BlockJacobi object witch stores the preconditioner.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 *
 * @see BlockJacobi
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobiFactory
    : public BasicBlockJacobiFactory<BlockJacobiFactory<ValueType, IndexType>> {
    friend class BasicBlockJacobiFactory<BlockJacobiFactory>;
    friend class EnablePolymorphicObject<BlockJacobiFactory, LinOpFactory>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using generated_type = BlockJacobi<ValueType, IndexType>;

protected:
    using BasicBlockJacobiFactory<BlockJacobiFactory>::BasicBlockJacobiFactory;

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> base) const override;
};


/**
 * This factory is used to create a block-Jacobi preconditioner from the
 * operator A, by inverting the diagonal blocks of the operator.
 *
 * This is a variant which generates AdaptiveBlockJacobi preconditioners,
 * which can use lower precision to store well-conditioned diagonal blocks,
 * and possibly improve the performance of preconditioner application.
 *
 * @tparam ValueType  highest precision of matrix elements, the precision of
 *                    vectors used for the LinOp::apply() method and the
 *                    precision in which all computation is performed
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 *
 * @see AdaptiveBlockJacobi
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AdaptiveBlockJacobiFactory
    : public BasicBlockJacobiFactory<
          AdaptiveBlockJacobiFactory<ValueType, IndexType>> {
    friend class BasicBlockJacobiFactory<AdaptiveBlockJacobiFactory>;
    friend class EnablePolymorphicObject<AdaptiveBlockJacobiFactory,
                                         LinOpFactory>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using generated_type = AdaptiveBlockJacobi<ValueType, IndexType>;

    /**
     * Sets the precision to use for storing each of the blocks.
     *
     * @param block_recisions  an array of precisions for each block
     */
    void set_block_precisions(
        const Array<typename generated_type::precision> &block_precisions)
    {
        block_precisions_ = block_precisions;
    }

    /**
     * @copydoc set_block_precisions(const Array<precision> &)
     */
    void set_block_precisions(
        Array<typename generated_type::precision> &&block_precisions)
    {
        block_precisions_ = std::move(block_precisions);
    }

    /**
     * Returns the precisions of diagonal blocks.
     *
     * @return precisions of diagonal blocks
     */
    const Array<typename generated_type::precision> &get_block_precisions()
        const noexcept
    {
        return block_precisions_;
    }

protected:
    AdaptiveBlockJacobiFactory(std::shared_ptr<const Executor> exec,
                               uint32 max_block_size = {})
        : BasicBlockJacobiFactory<AdaptiveBlockJacobiFactory>(exec,
                                                              max_block_size),
          block_precisions_(exec)
    {}

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> base) const override;

    Array<typename generated_type::precision> block_precisions_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
