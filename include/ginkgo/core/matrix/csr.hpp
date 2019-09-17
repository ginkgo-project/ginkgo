/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_CSR_HPP_
#define GKO_CORE_MATRIX_CSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;


/**
 * CSR is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format).
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column of each
 * nonzero element.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup csr
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Csr : public EnableLinOp<Csr<ValueType, IndexType>>,
            public EnableCreateMethod<Csr<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ConvertibleTo<Coo<ValueType, IndexType>>,
            public ConvertibleTo<Ell<ValueType, IndexType>>,
            public ConvertibleTo<Hybrid<ValueType, IndexType>>,
            public ConvertibleTo<Sellp<ValueType, IndexType>>,
            public ConvertibleTo<SparsityCsr<ValueType, IndexType>>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public Transposable,
            public Permutable<Csr<ValueType, IndexType>, IndexType> {
    friend class EnableCreateMethod<Csr>;
    friend class EnablePolymorphicObject<Csr, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Ell<ValueType, IndexType>;
    friend class Hybrid<ValueType, IndexType>;
    friend class Sellp<ValueType, IndexType>;
    friend class SparsityCsr<ValueType, IndexType>;

public:
    using EnableLinOp<Csr>::convert_to;
    using EnableLinOp<Csr>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    class automatical;

    class strategy_type {
        friend class automatical;

    public:
        strategy_type(std::string name) : name_(name) {}

        std::string get_name() { return name_; }

        virtual void process(const Array<index_type> &mtx_row_ptrs,
                             Array<index_type> *mtx_srow) = 0;

        virtual int64_t clac_size(const int64_t nnz) = 0;

    protected:
        void set_name(std::string name) { name_ = name; }

    private:
        std::string name_;
    };

    class classical : public strategy_type {
    public:
        classical() : strategy_type("classical") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow)
        {}

        int64_t clac_size(const int64_t nnz) { return 0; }
    };

    class merge_path : public strategy_type {
    public:
        merge_path() : strategy_type("merge_path") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow)
        {}

        int64_t clac_size(const int64_t nnz) { return 0; }
    };

    class cusparse : public strategy_type {
    public:
        cusparse() : strategy_type("cusparse") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow)
        {}

        int64_t clac_size(const int64_t nnz) { return 0; }
    };

    class load_balance : public strategy_type {
    public:
        load_balance()
            : load_balance(std::move(
                  gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
        {}

        load_balance(std::shared_ptr<const CudaExecutor> exec)
            : load_balance(exec->get_num_warps())
        {}

        load_balance(int64_t nwarps)
            : strategy_type("load_balance"), nwarps_(nwarps)
        {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow)
        {
            constexpr uint32 warp_size = 32;
            auto nwarps = mtx_srow->get_num_elems();

            if (nwarps > 0) {
                auto host_srow_exec = mtx_srow->get_executor()->get_master();
                auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
                const bool is_srow_on_host{host_srow_exec ==
                                           mtx_srow->get_executor()};
                const bool is_mtx_on_host{host_mtx_exec ==
                                          mtx_row_ptrs.get_executor()};
                Array<index_type> row_ptrs_host(host_mtx_exec);
                Array<index_type> srow_host(host_srow_exec);
                const index_type *row_ptrs{};
                index_type *srow{};
                if (is_srow_on_host) {
                    srow = mtx_srow->get_data();
                } else {
                    srow_host = *mtx_srow;
                    srow = srow_host.get_data();
                }
                if (is_mtx_on_host) {
                    row_ptrs = mtx_row_ptrs.get_const_data();
                } else {
                    row_ptrs_host = mtx_row_ptrs;
                    row_ptrs = row_ptrs_host.get_const_data();
                }
                for (size_type i = 0; i < nwarps; i++) {
                    srow[i] = 0;
                }
                const auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
                const auto num_elems = row_ptrs[num_rows];
                for (size_type i = 0; i < num_rows; i++) {
                    auto bucket =
                        ceildiv((ceildiv(row_ptrs[i + 1], warp_size) * nwarps),
                                ceildiv(num_elems, warp_size));
                    if (bucket < nwarps) {
                        srow[bucket]++;
                    }
                }
                // find starting row for thread i
                for (size_type i = 1; i < nwarps; i++) {
                    srow[i] += srow[i - 1];
                }
                if (!is_srow_on_host) {
                    *mtx_srow = srow_host;
                }
            }
        }

        int64_t clac_size(const int64_t nnz)
        {
            constexpr uint32 warp_size = 32;
            int multiple = 8;
            if (nnz >= 2000000) {
                multiple = 128;
            } else if (nnz >= 200000) {
                multiple = 32;
            }
            auto nwarps = nwarps_ * multiple;
            return min(ceildiv(nnz, warp_size), static_cast<int64_t>(nwarps));
        }

    private:
        int64_t nwarps_;
    };

    class automatical : public strategy_type {
    public:
        automatical()
            : automatical(std::move(
                  gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
        {}

        automatical(std::shared_ptr<const CudaExecutor> exec)
            : automatical(exec->get_num_warps())
        {}

        automatical(int64_t nwarps)
            : strategy_type("automatical"), nwarps_(nwarps)
        {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow)
        {
            // if the number of stored elements is larger than 1e6 or
            // the maximum number of stored elements per row is larger than
            // 64, use load_balance otherwise use classical
            auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
            const bool is_mtx_on_host{host_mtx_exec ==
                                      mtx_row_ptrs.get_executor()};
            Array<index_type> row_ptrs_host(host_mtx_exec);
            const index_type *row_ptrs{};
            if (is_mtx_on_host) {
                row_ptrs = mtx_row_ptrs.get_const_data();
            } else {
                row_ptrs_host = mtx_row_ptrs;
                row_ptrs = row_ptrs_host.get_const_data();
            }
            const auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
            if (row_ptrs[num_rows] > index_type(1e6)) {
                load_balance actual_strategy(nwarps_);
                if (is_mtx_on_host) {
                    actual_strategy.process(mtx_row_ptrs, mtx_srow);
                } else {
                    actual_strategy.process(row_ptrs_host, mtx_srow);
                }
                this->set_name(actual_strategy.get_name());
            } else {
                index_type maxnum = 0;
                for (index_type i = 1; i < num_rows + 1; i++) {
                    maxnum = max(maxnum, row_ptrs[i] - row_ptrs[i - 1]);
                }
                if (maxnum > 64) {
                    load_balance actual_strategy(nwarps_);
                    if (is_mtx_on_host) {
                        actual_strategy.process(mtx_row_ptrs, mtx_srow);
                    } else {
                        actual_strategy.process(row_ptrs_host, mtx_srow);
                    }
                    this->set_name(actual_strategy.get_name());
                } else {
                    classical actual_strategy;
                    if (is_mtx_on_host) {
                        actual_strategy.process(mtx_row_ptrs, mtx_srow);
                    } else {
                        actual_strategy.process(row_ptrs_host, mtx_srow);
                    }
                    this->set_name(actual_strategy.get_name());
                }
            }
        }

        int64_t clac_size(const int64_t nnz)
        {
            return std::make_shared<load_balance>(nwarps_)->clac_size(nnz);
        }

    private:
        int64_t nwarps_;
    };

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void convert_to(Coo<ValueType, IndexType> *result) const override;

    void move_to(Coo<ValueType, IndexType> *result) override;

    void convert_to(Ell<ValueType, IndexType> *result) const override;

    void move_to(Ell<ValueType, IndexType> *result) override;

    void convert_to(Hybrid<ValueType, IndexType> *result) const override;

    void move_to(Hybrid<ValueType, IndexType> *result) override;

    void convert_to(Sellp<ValueType, IndexType> *result) const override;

    void move_to(Sellp<ValueType, IndexType> *result) override;

    void convert_to(SparsityCsr<ValueType, IndexType> *result) const override;

    void move_to(SparsityCsr<ValueType, IndexType> *result) override;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    std::unique_ptr<Csr<ValueType, IndexType>> row_permute(
        const Array<IndexType> *permutation_indices) const override;

    std::unique_ptr<Csr<ValueType, IndexType>> column_permute(
        const Array<IndexType> *permutation_indices) const override;

    /**
     * Sorts all (value, col_idx) pairs in each row by column index
     */
    void sort_by_column_index();

    /*
     * Tests if all row entry pairs (value, col_idx) are sorted by column index
     *
     * @returns True if all row entry pairs (value, col_idx) are sorted by
     *          column index
     */
    bool is_sorted_by_column_index() const;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Csr::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type *get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc Csr::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * Returns the starting rows.
     *
     * @return the starting rows.
     */
    index_type *get_srow() noexcept { return srow_.get_data(); }

    /**
     * @copydoc Csr::get_srow()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_srow() const noexcept
    {
        return srow_.get_const_data();
    }

    /**
     * Returns the number of the srow stored elements (involved warps)
     *
     * @return the number of the srow stored elements (involved warps)
     */
    size_type get_num_srow_elements() const noexcept
    {
        return srow_.get_num_elems();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /** Returns the strategy
     *
     * @return the strategy
     */
    std::shared_ptr<strategy_type> get_strategy() const noexcept
    {
        return strategy_;
    }

protected:
    /**
     * Creates an uninitialized CSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  the strategy of CSR
     */
    Csr(std::shared_ptr<const Executor> exec,
        std::shared_ptr<strategy_type> strategy)
        : Csr(std::move(exec), dim<2>{}, {}, std::move(strategy))
    {}

    /**
     * Creates an uninitialized CSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param strategy  the strategy of CSR
     */
    Csr(std::shared_ptr<const Executor> exec, const dim<2> &size = dim<2>{},
        size_type num_nonzeros = {},
        std::shared_ptr<strategy_type> strategy = std::make_shared<cusparse>())
        : EnableLinOp<Csr>(exec, size),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          // avoid allocation for empty matrix
          row_ptrs_(exec, size[0] + (size[0] > 0)),
          srow_(exec, strategy->clac_size(num_nonzeros)),
          strategy_(std::move(strategy))
    {}

    /**
     * Creates a CSR matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    Csr(std::shared_ptr<const Executor> exec, const dim<2> &size,
        ValuesArray &&values, ColIdxsArray &&col_idxs, RowPtrsArray &&row_ptrs,
        std::shared_ptr<strategy_type> strategy = std::make_shared<cusparse>())
        : EnableLinOp<Csr>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)},
          srow_(exec),
          strategy_(std::move(strategy))
    {
        GKO_ASSERT_EQ(values_.get_num_elems(), col_idxs_.get_num_elems());
        GKO_ASSERT_EQ(this->get_size()[0] + 1, row_ptrs_.get_num_elems());
        this->make_srow();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Computes srow. It should be run after changing any row_ptrs_ value.
     */
    void make_srow()
    {
        srow_.resize_and_reset(strategy_->clac_size(values_.get_num_elems()));
        strategy_->process(row_ptrs_, &srow_);
    }

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<index_type> row_ptrs_;
    Array<index_type> srow_;
    std::shared_ptr<strategy_type> strategy_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_HPP_
