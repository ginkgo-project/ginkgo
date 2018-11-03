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

#ifndef GKO_CORE_MATRIX_HYBRID_HPP_
#define GKO_CORE_MATRIX_HYBRID_HPP_


#include <algorithm>


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/matrix/coo.hpp"
#include "core/matrix/ell.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;


/**
 * HYBRID is a matrix format which splits the matrix into ELLPACK and COO
 * format. Achieve the excellent performance with a proper partition of ELLPACK
 * and COO.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Hybrid : public EnableLinOp<Hybrid<ValueType, IndexType>>,
               public EnableCreateMethod<Hybrid<ValueType, IndexType>>,
               public ConvertibleTo<Dense<ValueType>>,
               public ReadableFromMatrixData<ValueType, IndexType>,
               public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableCreateMethod<Hybrid>;
    friend class EnablePolymorphicObject<Hybrid, LinOp>;
    friend class Dense<ValueType>;

public:
    using EnableLinOp<Hybrid>::convert_to;
    using EnableLinOp<Hybrid>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using coo_type = Coo<ValueType, IndexType>;
    using ell_type = Ell<ValueType, IndexType>;

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
        void compute_hybrid_config(const Array<size_type> &row_nnz,
                                   size_type *ell_num_stored_elements_per_row,
                                   size_type *coo_nnz)
        {
            Array<size_type> ref_row_nnz(row_nnz.get_executor()->get_master(),
                                         row_nnz.get_num_elems());
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
        const size_type get_ell_num_stored_elements_per_row() const noexcept
        {
            return ell_num_stored_elements_per_row_;
        }

        /**
         * Returns the number of nonzeros of the coo part.
         *
         * @return the number of nonzeros of the coo part
         */
        const size_type get_coo_nnz() const noexcept { return coo_nnz_; }

        /**
         * Computes the number of stored elements per row of the ell part.
         *
         * @param row_nnz  the number of nonzeros of each row
         *
         * @return the number of stored elements per row of the ell part
         */
        virtual size_type compute_ell_num_stored_elements_per_row(
            Array<size_type> *row_nnz) const = 0;

    protected:
        /**
         * Computes the number of residual nonzeros as the number of nonzeros of
         * the coo part.
         *
         * @param row_nnz  the number of nonzeros of each row
         *
         * @return the number of nonzeros of the coo part
         */
        size_type compute_coo_nnz(const Array<size_type> &row_nnz) const
        {
            size_type coo_nnz = 0;
            auto row_nnz_val = row_nnz.get_const_data();
            for (size_type i = 0; i < row_nnz.get_num_elems(); i++) {
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
            Array<size_type> *row_nnz) const override
        {
            return num_columns_;
        }

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
        explicit imbalance_limit(float percent = 0.8) : percent_(percent)
        {
            percent_ = std::min(percent_, 1.0f);
            percent_ = std::max(percent_, 0.0f);
        }

        size_type compute_ell_num_stored_elements_per_row(
            Array<size_type> *row_nnz) const override
        {
            auto row_nnz_val = row_nnz->get_data();
            auto num_rows = row_nnz->get_num_elems();
            std::sort(row_nnz_val, row_nnz_val + num_rows);
            if (percent_ < 1) {
                auto percent_pos = static_cast<size_type>(num_rows * percent_);
                return row_nnz_val[percent_pos];
            } else {
                return row_nnz_val[num_rows - 1];
            }
        }

    private:
        float percent_;
    };

    /**
     * automatic is a stratgy_type which decides the number of stored elements
     * per row of the ell part automatically.
     */
    class automatic : public strategy_type {
    public:
        /**
         * Creates an automatic strategy.
         */
        automatic() {}

        size_type compute_ell_num_stored_elements_per_row(
            Array<size_type> *row_nnz) const override
        {
            const auto num_rows = row_nnz->get_num_elems();
            const auto row_nnz_val = row_nnz->get_const_data();
            size_type minimal(num_rows / 1000);
            for (size_type i = 0; i < num_rows; i++) {
                minimal = std::min(minimal, row_nnz_val[i]);
            }
            return minimal;
        }
    };

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    /**
     * Returns the values of the ell part.
     *
     * @return the values of the ell part
     */
    value_type *get_ell_values() noexcept { return ell_->get_values(); }

    /**
     * @copydoc Hybrid::get_ell_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_ell_values() const noexcept
    {
        return ell_->get_const_values();
    }

    /**
     * Returns the column indexes of the ell part.
     *
     * @return the column indexes of the ell part
     */
    index_type *get_ell_col_idxs() noexcept { return ell_->get_col_idxs(); }

    /**
     * @copydoc Hybrid::get_ell_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_ell_col_idxs() const noexcept
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
    value_type &ell_val_at(size_type row, size_type idx) noexcept
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
    index_type &ell_col_at(size_type row, size_type idx) noexcept
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
    const ell_type *get_ell() const noexcept { return ell_.get(); }

    /**
     * Returns the values of the coo part.
     *
     * @return the values of the coo part.
     */
    value_type *get_coo_values() noexcept { return coo_->get_values(); }

    /**
     * @copydoc Hybrid::get_coo_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_coo_values() const noexcept
    {
        return coo_->get_const_values();
    }

    /**
     * Returns the column indexes of the coo part.
     *
     * @return the column indexes of the coo part.
     */
    index_type *get_coo_col_idxs() noexcept { return coo_->get_col_idxs(); }

    /**
     * @copydoc Hybrid::get_coo_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_coo_col_idxs() const noexcept
    {
        return coo_->get_const_col_idxs();
    }

    /**
     * Returns the row indexes of the coo part.
     *
     * @return the row indexes of the coo part.
     */
    index_type *get_coo_row_idxs() noexcept { return coo_->get_row_idxs(); }

    /**
     * @copydoc Hybrid::get_coo_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_coo_row_idxs() const noexcept
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
    const coo_type *get_coo() const noexcept { return coo_.get(); }

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
     * Copies data from another Hybrid.
     *
     * @param other  the Hybrid to copy from
     *
     * @return this
     */
    Hybrid &operator=(const Hybrid &other)
    {
        if (&other == this) {
            return *this;
        }
        EnableLinOp<Hybrid<ValueType, IndexType>>::operator=(other);
        this->coo_->copy_from(other.get_coo());
        this->ell_->copy_from(other.get_ell());
        return *this;
    }

protected:
    /**
     * Creates an uninitialized Hybrid matrix of specified method.
     *    (ell_num_stored_elements_per_row is set to the number of cols of the
     * matrix. ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  strategy of deciding the Hybrid config
     */
    Hybrid(
        std::shared_ptr<const Executor> exec,
        std::shared_ptr<strategy_type> strategy = std::make_shared<automatic>())
        : Hybrid(std::move(exec), dim<2>{}, std::move(strategy))
    {}

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *    (ell_num_stored_elements_per_row is set to the number of cols of the
     * matrix. ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param strategy  strategy of deciding the Hybrid config
     */
    Hybrid(
        std::shared_ptr<const Executor> exec, const dim<2> &size,
        std::shared_ptr<strategy_type> strategy = std::make_shared<automatic>())
        : Hybrid(std::move(exec), size, size[1], std::move(strategy))
    {}

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *    (ell_stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stroed elements per
     *                                      row
     * @param strategy  strategy of deciding the Hybrid config
     */
    Hybrid(
        std::shared_ptr<const Executor> exec, const dim<2> &size,
        size_type num_stored_elements_per_row,
        std::shared_ptr<strategy_type> strategy = std::make_shared<automatic>())
        : Hybrid(std::move(exec), size, num_stored_elements_per_row, size[0],
                 {}, std::move(strategy))
    {}

    /**
     * Creates an uninitialized Hybrid matrix of the specified size and method.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_stored_elements_per_row   the number of stored elements per
     *                                      row
     * @param stride  stride of the rows
     * @param strategy  strategy of deciding the Hybrid config
     */
    Hybrid(std::shared_ptr<const Executor> exec, const dim<2> &size,
           size_type num_stored_elements_per_row, size_type stride,
           std::shared_ptr<strategy_type> strategy)
        : Hybrid(std::move(exec), size, num_stored_elements_per_row, stride, {},
                 std::move(strategy))
    {}

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
     */
    Hybrid(
        std::shared_ptr<const Executor> exec, const dim<2> &size,
        size_type num_stored_elements_per_row, size_type stride,
        size_type num_nonzeros = {},
        std::shared_ptr<strategy_type> strategy = std::make_shared<automatic>())
        : EnableLinOp<Hybrid>(exec, size),
          ell_(std::move(ell_type::create(
              exec, size, num_stored_elements_per_row, stride))),
          coo_(std::move(coo_type::create(exec, size, num_nonzeros))),
          strategy_(std::move(strategy))
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::shared_ptr<ell_type> ell_;
    std::shared_ptr<coo_type> coo_;
    std::shared_ptr<strategy_type> strategy_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_HYBRID_HPP_
