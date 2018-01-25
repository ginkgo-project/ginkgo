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


namespace gko {
namespace preconditioner {


template <typename, typename>
class BlockJacobiFactory;


template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobi : public LinOp,
                    public ConvertibleTo<BlockJacobi<ValueType, IndexType>> {
    friend class BlockJacobiFactory<ValueType, IndexType>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(BlockJacobi<ValueType, IndexType> *result) const override;

    void move_to(BlockJacobi<ValueType, IndexType> *result) override;

    size_type get_num_blocks() const noexcept { return num_blocks_; }

    void set_num_blocks(size_type num_blocks) noexcept
    {
        num_blocks_ = num_blocks;
    }

    int32 get_max_block_size() const noexcept { return max_block_size_; }

    const Array<IndexType> &get_block_pointers() const noexcept
    {
        return block_pointers_;
    }

    Array<IndexType> &get_block_pointers() noexcept { return block_pointers_; }

    size_type get_padding() const noexcept { return max_block_size_; }

    ValueType *get_blocks() noexcept { return blocks_.get_data(); }

    const ValueType *get_const_blocks() const noexcept
    {
        return blocks_.get_const_data();
    }

    std::shared_ptr<const LinOp> get_system_matrix() const noexcept
    {
        return system_matrix_;
    }

protected:
    BlockJacobi(std::shared_ptr<const Executor> exec,
                const LinOp *system_matrix, int32 max_block_size,
                const Array<IndexType> &block_pointers)
        : LinOp(exec, system_matrix->get_num_rows(),
                system_matrix->get_num_cols(),
                system_matrix->get_num_rows() * max_block_size),
          num_blocks_(block_pointers.get_num_elems() - 1),
          max_block_size_(max_block_size),
          block_pointers_(block_pointers),
          blocks_(exec, system_matrix->get_num_rows() * max_block_size)
    {
        block_pointers_.set_executor(this->get_executor());
        this->generate(system_matrix);
    }

    static std::unique_ptr<BlockJacobi> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<const LinOp> system_matrix, int32 max_block_size,
        const Array<IndexType> &block_pointers)
    {
        return std::unique_ptr<BlockJacobi>(
            new BlockJacobi(std::move(exec), std::move(system_matrix),
                            max_block_size, block_pointers));
    }

    void generate(const LinOp *system_matrix);

private:
    size_type num_blocks_;
    int32 max_block_size_;
    Array<IndexType> block_pointers_;
    Array<ValueType> blocks_;
};


template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobiFactory : public LinOpFactory {
public:
    using value_type = ValueType;

    static std::unique_ptr<BlockJacobiFactory> create(
        std::shared_ptr<const Executor> exec, int32 max_block_size)
    {
        return std::unique_ptr<BlockJacobiFactory>(
            new BlockJacobiFactory(std::move(exec), max_block_size));
    }

    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

    int32 get_max_block_size() const noexcept { return max_block_size_; }

    void set_block_pointers(const Array<IndexType> &block_pointers)
    {
        block_pointers_ = block_pointers;
    }

    void set_block_pointers(Array<IndexType> &&block_pointers)
    {
        block_pointers_ = std::move(block_pointers);
    }

    const Array<IndexType> &get_block_pointers() const noexcept
    {
        return block_pointers_;
    }

protected:
    BlockJacobiFactory(std::shared_ptr<const Executor> exec,
                       int32 max_block_size)
        : LinOpFactory(exec),
          max_block_size_(max_block_size),
          block_pointers_(exec)
    {}

private:
    int32 max_block_size_;
    Array<IndexType> block_pointers_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
