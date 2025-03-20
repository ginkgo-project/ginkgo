// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_COMPRESSED_COO_HPP_
#define GKO_PUBLIC_CORE_MATRIX_COMPRESSED_COO_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType = default_precision, typename IndexType = int32>
class CompactRowCoo : public EnableLinOp<CompactRowCoo<ValueType, IndexType>>,
                      public ReadableFromMatrixData<ValueType, IndexType>,
                      public WritableToMatrixData<ValueType, IndexType> {
    friend class EnablePolymorphicObject<CompactRowCoo, LinOp>;

public:
    using EnableLinOp<CompactRowCoo>::convert_to;
    using EnableLinOp<CompactRowCoo>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    value_type* get_values() noexcept { return values_.get_data(); }
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    const index_type* get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    uint32* get_row_bits() noexcept { return row_bits_.get_data(); }

    const uint32* get_const_row_bits() const noexcept
    {
        return row_bits_.get_const_data();
    }

    index_type* get_row_bit_ranks() noexcept { return row_ranks_.get_data(); }

    const index_type* get_const_row_bit_ranks() const noexcept
    {
        return row_ranks_.get_const_data();
    }

    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_size();
    }

    static std::unique_ptr<CompactRowCoo> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
        size_type num_nonzeros = {});

protected:
    CompactRowCoo(std::shared_ptr<const Executor> exec,
                  const dim<2>& size = dim<2>{}, size_type num_nonzeros = {});

    void row_bits_from_idxs(const array<IndexType> row_idxs);

    void resize(dim<2> new_size, size_type nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<uint32> row_bits_;
    array<index_type> row_ranks_;
};


template <typename ValueType = default_precision, typename IndexType = int32>
class CompactRowCompressedColumnCoo
    : public EnableLinOp<CompactRowCompressedColumnCoo<ValueType, IndexType>>,
      public ReadableFromMatrixData<ValueType, IndexType> {
    friend class EnablePolymorphicObject<CompactRowCompressedColumnCoo, LinOp>;

public:
    using EnableLinOp<CompactRowCompressedColumnCoo>::convert_to;
    using EnableLinOp<CompactRowCompressedColumnCoo>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    value_type* get_values() noexcept { return values_.get_data(); }
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    const index_type* get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    uint32* get_row_bits() noexcept { return row_bits_.get_data(); }

    const uint32* get_const_row_bits() const noexcept
    {
        return row_bits_.get_const_data();
    }

    index_type* get_row_bit_ranks() noexcept { return row_ranks_.get_data(); }

    const index_type* get_const_row_bit_ranks() const noexcept
    {
        return row_ranks_.get_const_data();
    }

    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_size();
    }

    static std::unique_ptr<CompactRowCompressedColumnCoo> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
        size_type num_nonzeros = {});

protected:
    CompactRowCompressedColumnCoo(std::shared_ptr<const Executor> exec,
                                  const dim<2>& size = dim<2>{},
                                  size_type num_nonzeros = {});

    void resize(dim<2> new_size, size_type nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<uint8> col_deltas_;
    array<uint32> col_base_bits_;
    array<uint32> col_base_ranks_;
    array<uint32> col_bases_;
    array<uint32> row_bits_;
    array<index_type> row_ranks_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_COMPRESSED_COO_HPP_
