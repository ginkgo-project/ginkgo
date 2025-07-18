// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_HYBRID_HPP_
#define GKO_PUBLIC_CORE_MATRIX_HYBRID_HPP_


#include <algorithm>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Csr;


/**
 * HYBRID is a matrix format which splits the matrix into ELLPACK and COO
 * format. Achieve the excellent performance with a proper partition of ELLPACK
 * and COO.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup hybrid
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Hybrid
    : public EnableLinOp<Hybrid<ValueType, IndexType>>,
      public ConvertibleTo<Hybrid<next_precision<ValueType>, IndexType>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<Hybrid<next_precision<ValueType, 2>, IndexType>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<Hybrid<next_precision<ValueType, 3>, IndexType>>,
#endif
      public ConvertibleTo<Dense<ValueType>>,
      public ConvertibleTo<Csr<ValueType, IndexType>>,
      public DiagonalExtractable<ValueType>,
      public ReadableFromMatrixData<ValueType, IndexType>,
      public WritableToMatrixData<ValueType, IndexType>,
      public EnableAbsoluteComputation<
          remove_complex<Hybrid<ValueType, IndexType>>> {
    friend class EnablePolymorphicObject<Hybrid, LinOp>;
    friend class Dense<ValueType>;
    friend class Csr<ValueType, IndexType>;
    friend class Hybrid<to_complex<ValueType>, IndexType>;


public:
    using EnableLinOp<Hybrid>::convert_to;
    using EnableLinOp<Hybrid>::move_to;
    using ConvertibleTo<
        Hybrid<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Hybrid<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Csr<ValueType, IndexType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using coo_type = Coo<ValueType, IndexType>;
    using ell_type = Ell<ValueType, IndexType>;
    using absolute_type = remove_complex<Hybrid>;


    /**
     * strategy_type is to decide how to set the hybrid config. It
     * computes the number of stored elements per row of the ell part and
     * then set the number of residual nonzeros as the number of nonzeros of the
     * coo part.
     *
     * The practical strategy method should inherit strategy_type and implement
     * its `compute_ell_num_stored_elements_per_row` function.
     */
    class strategy_type {
    public:
        /**
         * Creates a strategy_type.
         */
        strategy_type()
            : ell_num_stored_elements_per_row_(zero<size_type>()),
              coo_nnz_(zero<size_type>())
        {}

        /**
         * Computes the config of the Hybrid matrix
         * (ell_num_stored_elements_per_row and coo_nnz). For now, it copies
         * row_nnz to the reference executor and performs all operations on the
         * reference executor.
         *
         * @param row_nnz  the number of nonzeros of each row
         * @param ell_num_stored_elements_per_row  the output number of stored
         *                                         elements per row of the ell
         *                                         part
         * @param coo_nnz  the output number of nonzeros of the coo part
         */
        void compute_hybrid_config(const array<size_type>& row_nnz,
                                   size_type* ell_num_stored_elements_per_row,
                                   size_type* coo_nnz)
        {
            array<size_type> ref_row_nnz(row_nnz.get_executor()->get_master(),
                                         row_nnz.get_size());
            ref_row_nnz = row_nnz;
            ell_num_stored_elements_per_row_ =
                this->compute_ell_num_stored_elements_per_row(&ref_row_nnz);
            coo_nnz_ = this->compute_coo_nnz(ref_row_nnz);
            *ell_num_stored_elements_per_row = ell_num_stored_elements_per_row_;
            *coo_nnz = coo_nnz_;
        }

        /**
         * Returns the number of stored elements per row of the ell part.
         *
         * @return the number of stored elements per row of the ell part
         */
        size_type get_ell_num_stored_elements_per_row() const noexcept
        {
            return ell_num_stored_elements_per_row_;
        }

        /**
         * Returns the number of nonzeros of the coo part.
         *
         * @return the number of nonzeros of the coo part
         */
        size_type get_coo_nnz() const noexcept { return coo_nnz_; }

        /**
         * Computes the number of stored elements per row of the ell part.
         *
         * @param row_nnz  the number of nonzeros of each row
         *
         * @return the number of stored elements per row of the ell part
         */
        virtual size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const = 0;

    protected:
        /**
         * Computes the number of residual nonzeros as the number of nonzeros of
         * the coo part.
         *
         * @param row_nnz  the number of nonzeros of each row
         *
         * @return the number of nonzeros of the coo part
         */
        size_type compute_coo_nnz(const array<size_type>& row_nnz) const
        {
            size_type coo_nnz = 0;
            auto row_nnz_val = row_nnz.get_const_data();
            for (size_type i = 0; i < row_nnz.get_size(); i++) {
                if (row_nnz_val[i] > ell_num_stored_elements_per_row_) {
                    coo_nnz +=
                        row_nnz_val[i] - ell_num_stored_elements_per_row_;
                }
            }
            return coo_nnz;
        }

    private:
        size_type ell_num_stored_elements_per_row_;
        size_type coo_nnz_;
    };

    /**
     * column_limit is a strategy_type which decides the number of stored
     * elements per row of the ell part by specifying the number of columns.
     */
    class column_limit : public strategy_type {
    public:
        /**
         * Creates a column_limit strategy.
         *
         * @param num_column  the specified number of columns of the ell part
         */
        explicit column_limit(size_type num_column = 0)
            : num_columns_(num_column)
        {}

        size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const override
        {
            return num_columns_;
        }

        /**
         * Get the number of columns limit
         *
         * @return the number of columns limit
         */
        auto get_num_columns() const { return num_columns_; }

    private:
        size_type num_columns_;
    };

    /**
     * imbalance_limit is a strategy_type which decides the number of stored
     * elements per row of the ell part according to the percent. It sorts the
     * number of nonzeros of each row and takes the value at the position
     * `floor(percent * num_row)` as the number of stored elements per row of
     * the ell part. Thus, at least `percent` rows of all are in the ell part.
     */
    class imbalance_limit : public strategy_type {
    public:
        /**
         * Creates a imbalance_limit strategy.
         *
         * @param percent  the row_nnz[floor(num_rows*percent)] is the number of
         *                 stored elements per row of the ell part
         */
        explicit imbalance_limit(double percent = 0.8) : percent_(percent)
        {
            percent_ = std::min(percent_, 1.0);
            percent_ = std::max(percent_, 0.0);
        }

        size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const override
        {
            auto row_nnz_val = row_nnz->get_data();
            auto num_rows = row_nnz->get_size();
            if (num_rows == 0) {
                return 0;
            }
            std::sort(row_nnz_val, row_nnz_val + num_rows);
            if (percent_ < 1) {
                auto percent_pos = static_cast<size_type>(num_rows * percent_);
                return row_nnz_val[percent_pos];
            } else {
                return row_nnz_val[num_rows - 1];
            }
        }

        /**
         * Get the percent setting
         *
         * @return percent
         */
        auto get_percentage() const { return percent_; }

    private:
        double percent_;
    };

    /**
     * imbalance_bounded_limit is a strategy_type which decides the number of
     * stored elements per row of the ell part. It uses the imbalance_limit and
     * adds the upper bound of the number of ell's cols by the number of rows.
     */
    class imbalance_bounded_limit : public strategy_type {
    public:
        /**
         * Creates a imbalance_bounded_limit strategy.
         */
        imbalance_bounded_limit(double percent = 0.8, double ratio = 0.0001)
            : strategy_(imbalance_limit(percent)), ratio_(ratio)
        {}

        size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const override
        {
            auto num_rows = row_nnz->get_size();
            auto ell_cols =
                strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
            return std::min(ell_cols,
                            static_cast<size_type>(num_rows * ratio_));
        }

        /**
         * Get the percent setting
         *
         * @return percent
         */
        auto get_percentage() const { return strategy_.get_percentage(); }

        /**
         * Get the ratio setting
         *
         * @return ratio
         */
        auto get_ratio() const { return ratio_; }

    private:
        imbalance_limit strategy_;
        double ratio_;
    };


    /**
     * minimal_storage_limit is a strategy_type which decides the number of
     * stored elements per row of the ell part. It is determined by the size of
     * ValueType and IndexType, the storage is the minimum among all partition.
     */
    class minimal_storage_limit : public strategy_type {
    public:
        /**
         * Creates a minimal_storage_limit strategy.
         */
        minimal_storage_limit()
            : strategy_(
                  imbalance_limit(static_cast<double>(sizeof(IndexType)) /
                                  (sizeof(ValueType) + 2 * sizeof(IndexType))))
        {}

        size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const override
        {
            return strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
        }

        /**
         * Get the percent setting
         *
         * @return percent
         */
        auto get_percentage() const { return strategy_.get_percentage(); }

    private:
        imbalance_limit strategy_;
    };


    /**
     * automatic is a strategy_type which decides the number of stored elements
     * per row of the ell part automatically.
     */
    class automatic : public strategy_type {
    public:
        /**
         * Creates an automatic strategy.
         */
        automatic() : strategy_(imbalance_bounded_limit(1.0 / 3.0, 0.001)) {}

        size_type compute_ell_num_stored_elements_per_row(
            array<size_type>* row_nnz) const override
        {
            return strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
        }

    private:
        imbalance_bounded_limit strategy_;
    };

    friend class Hybrid<previous_precision<ValueType>, IndexType>;

    void convert_to(
        Hybrid<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Hybrid<next_precision<ValueType>, IndexType>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class Hybrid<previous_precision<ValueType, 2>, IndexType>;
    using ConvertibleTo<
        Hybrid<next_precision<ValueType, 2>, IndexType>>::convert_to;
    using ConvertibleTo<
        Hybrid<next_precision<ValueType, 2>, IndexType>>::move_to;

    void convert_to(
        Hybrid<next_precision<ValueType, 2>, IndexType>* result) const override;

    void move_to(
        Hybrid<next_precision<ValueType, 2>, IndexType>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class Hybrid<previous_precision<ValueType, 3>, IndexType>;
    using ConvertibleTo<
        Hybrid<next_precision<ValueType, 3>, IndexType>>::convert_to;
    using ConvertibleTo<
        Hybrid<next_precision<ValueType, 3>, IndexType>>::move_to;

    void convert_to(
        Hybrid<next_precision<ValueType, 3>, IndexType>* result) const override;

    void move_to(
        Hybrid<next_precision<ValueType, 3>, IndexType>* result) override;
#endif

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void convert_to(Csr<ValueType, IndexType>* other) const override;

    void move_to(Csr<ValueType, IndexType>* other) override;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns the values of the ell part.
     *
     * @return the values of the ell part
     */
    value_type* get_ell_values() noexcept { return ell_->get_values(); }

    /**
     * @copydoc Hybrid::get_ell_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_ell_values() const noexcept
    {
        return ell_->get_const_values();
    }

    /**
     * Returns the column indexes of the ell part.
     *
     * @return the column indexes of the ell part
     */
    index_type* get_ell_col_idxs() noexcept { return ell_->get_col_idxs(); }

    /**
     * @copydoc Hybrid::get_ell_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_ell_col_idxs() const noexcept
    {
        return ell_->get_const_col_idxs();
    }

    /**
     * Returns the number of stored elements per row of ell part.
     *
     * @return the number of stored elements per row of ell part
     */
    size_type get_ell_num_stored_elements_per_row() const noexcept
    {
        return ell_->get_num_stored_elements_per_row();
    }

    /**
     * Returns the stride of the ell part.
     *
     * @return the stride of the ell part
     */
    size_type get_ell_stride() const noexcept { return ell_->get_stride(); }

    /**
     * Returns the number of elements explicitly stored in the ell part.
     *
     * @return the number of elements explicitly stored in the ell part
     */
    size_type get_ell_num_stored_elements() const noexcept
    {
        return ell_->get_num_stored_elements();
    }

    /**
     * Returns the `idx`-th non-zero element of the `row`-th row in the ell
     * part.
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& ell_val_at(size_type row, size_type idx) noexcept
    {
        return ell_->val_at(row, idx);
    }

    /**
     * @copydoc Hybrid::ell_val_at(size_type, size_type)
     */
    value_type ell_val_at(size_type row, size_type idx) const noexcept
    {
        return ell_->val_at(row, idx);
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row in the ell part.
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    index_type& ell_col_at(size_type row, size_type idx) noexcept
    {
        return ell_->col_at(row, idx);
    }

    /**
     * @copydoc Hybrid::ell_col_at(size_type, size_type)
     */
    index_type ell_col_at(size_type row, size_type idx) const noexcept
    {
        return ell_->col_at(row, idx);
    }

    /**
     * Returns the matrix of the ell part
     *
     * @return the matrix of the ell part
     */
    const ell_type* get_ell() const noexcept { return ell_.get(); }

    /**
     * Returns the values of the coo part.
     *
     * @return the values of the coo part.
     */
    value_type* get_coo_values() noexcept { return coo_->get_values(); }

    /**
     * @copydoc Hybrid::get_coo_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_coo_values() const noexcept
    {
        return coo_->get_const_values();
    }

    /**
     * Returns the column indexes of the coo part.
     *
     * @return the column indexes of the coo part.
     */
    index_type* get_coo_col_idxs() noexcept { return coo_->get_col_idxs(); }

    /**
     * @copydoc Hybrid::get_coo_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_coo_col_idxs() const noexcept
    {
        return coo_->get_const_col_idxs();
    }

    /**
     * Returns the row indexes of the coo part.
     *
     * @return the row indexes of the coo part.
     */
    index_type* get_coo_row_idxs() noexcept { return coo_->get_row_idxs(); }

    /**
     * @copydoc Hybrid::get_coo_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_coo_row_idxs() const noexcept
    {
        return coo_->get_const_row_idxs();
    }

    /**
     * Returns the number of elements explicitly stored in the coo part.
     *
     * @return the number of elements explicitly stored in the coo part
     */
    size_type get_coo_num_stored_elements() const noexcept
    {
        return coo_->get_num_stored_elements();
    }

    /**
     * Returns the matrix of the coo part
     *
     * @return the matrix of the coo part
     */
    const coo_type* get_coo() const noexcept { return coo_.get(); }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return coo_->get_num_stored_elements() +
               ell_->get_num_stored_elements();
    }

    /**
     * Returns the strategy
     *
     * @return the strategy
     */
    std::shared_ptr<strategy_type> get_strategy() const noexcept
    {
        return strategy_;
    }

    /**
     * Returns the current strategy allowed in given hybrid format
     *
     * @tparam HybType  hybrid type
     *
     * @return the strategy
     */
    template <typename HybType>
    std::shared_ptr<typename HybType::strategy_type> get_strategy() const;

    /**
     * Creates an uninitialized Hybrid matrix of specified method.
     *    (ell_num_stored_elements_per_row is set to the number of cols of the
     * matrix. ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  strategy of deciding the Hybrid config
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Hybrid> create(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<strategy_type> strategy =
            std::make_shared<automatic>());

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *    (ell_num_stored_elements_per_row is set to the number of cols of the
     * matrix. ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param strategy  strategy of deciding the Hybrid config
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Hybrid> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        std::shared_ptr<strategy_type> strategy =
            std::make_shared<automatic>());

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *    (ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stroed elements per
     *                                      row
     * @param strategy  strategy of deciding the Hybrid config
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Hybrid> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_stored_elements_per_row,
        std::shared_ptr<strategy_type> strategy =
            std::make_shared<automatic>());

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     * @param stride  stride of the rows
     * @param strategy  strategy of deciding the Hybrid config
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Hybrid> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_stored_elements_per_row, size_type stride,
        std::shared_ptr<strategy_type> strategy);

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     * @param stride  stride of the rows
     * @param num_nonzeros  number of nonzeros
     * @param strategy  strategy of deciding the Hybrid config
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Hybrid> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_stored_elements_per_row, size_type stride,
        size_type num_nonzeros = {},
        std::shared_ptr<strategy_type> strategy =
            std::make_shared<automatic>());

    /**
     * Copy-assigns a Hybrid matrix. Preserves the executor, copy-assigns the
     * Ell and Coo matrices.
     */
    Hybrid& operator=(const Hybrid&);

    /**
     * Move-assigns a Hybrid matrix. Preserves the executor, move-assigns the
     * Ell and Coo matrices. The moved-from matrix is empty (0x0 with empty
     * Ell/Coo matrices).
     */
    Hybrid& operator=(Hybrid&&);

    /**
     * Copy-assigns a Hybrid matrix. Inherits the executor, copies the Ell and
     * Coo matrices.
     */
    Hybrid(const Hybrid&);

    /**
     * Move-assigns a Hybrid matrix. Inherits the executor, moves the Ell and
     * Coo matrices. The moved-from matrix is empty (0x0 with empty Ell/Coo
     * matrices).
     */
    Hybrid(Hybrid&&);

protected:
    Hybrid(std::shared_ptr<const Executor> exec, const dim<2>& size = {},
           size_type num_stored_elements_per_row = 0, size_type stride = 0,
           size_type num_nonzeros = 0,
           std::shared_ptr<strategy_type> strategy =
               std::make_shared<automatic>());

    /**
     * Resizes the matrix to the given dimensions and storage sizes.
     *
     * @param new_size  the new matrix dimensions
     * @param ell_row_nnz  the number of non-zeros per row stored in ELL
     * @param coo_nnz  the number of non-zeros stored in COO
     *
     * @see Ell::resize(dim<2>, size_type)
     * @see Coo::resize(dim<2>, size_type)
     */
    void resize(dim<2> new_size, size_type ell_row_nnz, size_type coo_nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::unique_ptr<ell_type> ell_;
    std::unique_ptr<coo_type> coo_;
    std::shared_ptr<strategy_type> strategy_;
};


template <typename ValueType, typename IndexType>
template <typename HybType>
std::shared_ptr<typename HybType::strategy_type>
Hybrid<ValueType, IndexType>::get_strategy() const
{
    static_assert(
        std::is_same<HybType, Hybrid<typename HybType::value_type,
                                     typename HybType::index_type>>::value,
        "The given `HybType` type must be of type `matrix::Hybrid`!");

    std::shared_ptr<typename HybType::strategy_type> strategy;
    if (std::dynamic_pointer_cast<automatic>(strategy_)) {
        strategy = std::make_shared<typename HybType::automatic>();
    } else if (auto temp = std::dynamic_pointer_cast<minimal_storage_limit>(
                   strategy_)) {
        // minimal_storage_limit is related to ValueType and IndexType size.
        if (sizeof(value_type) == sizeof(typename HybType::value_type) &&
            sizeof(index_type) == sizeof(typename HybType::index_type)) {
            strategy =
                std::make_shared<typename HybType::minimal_storage_limit>();
        } else {
            strategy = std::make_shared<typename HybType::imbalance_limit>(
                temp->get_percentage());
        }
    } else if (auto temp = std::dynamic_pointer_cast<imbalance_bounded_limit>(
                   strategy_)) {
        strategy = std::make_shared<typename HybType::imbalance_bounded_limit>(
            temp->get_percentage(), temp->get_ratio());
    } else if (auto temp =
                   std::dynamic_pointer_cast<imbalance_limit>(strategy_)) {
        strategy = std::make_shared<typename HybType::imbalance_limit>(
            temp->get_percentage());
    } else if (auto temp = std::dynamic_pointer_cast<column_limit>(strategy_)) {
        strategy = std::make_shared<typename HybType::column_limit>(
            temp->get_num_columns());
    } else {
        GKO_NOT_SUPPORTED(strategy_);
    }
    return strategy;
}


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_HYBRID_HPP_
