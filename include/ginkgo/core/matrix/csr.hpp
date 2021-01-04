/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_MATRIX_CSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_CSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>


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

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class CsrBuilder;


namespace detail {


template <typename ValueType = default_precision, typename IndexType = int32>
void strategy_rebuild_helper(Csr<ValueType, IndexType> *result);


}  // namespace detail


/**
 * CSR is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format).
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column of each
 * nonzero element.
 *
 * The Csr LinOp supports different operations:
 *
 * ```cpp
 * matrix::Csr *A, *B, *C;      // matrices
 * matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
 * matrix::Dense *alpha, *beta; // scalars of dimension 1x1
 * matrix::Identity *I;         // identity matrix
 *
 * // Applying to Dense matrices computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 *
 * // Applying to Csr matrices computes a SpGEMM product of two sparse matrices
 * A->apply(B, C)              // C = A*B
 * A->apply(alpha, B, beta, C) // C = alpha*A*B + beta*C
 *
 * // Applying to an Identity matrix computes a SpGEAM sparse matrix addition
 * A->apply(alpha, I, beta, B) // B = alpha*A + beta*B
 * ```
 * Both the SpGEMM and SpGEAM operation require the input matrices to be sorted
 * by column index, otherwise the algorithms will produce incorrect results.
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
            public ConvertibleTo<Csr<next_precision<ValueType>, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ConvertibleTo<Coo<ValueType, IndexType>>,
            public ConvertibleTo<Ell<ValueType, IndexType>>,
            public ConvertibleTo<Hybrid<ValueType, IndexType>>,
            public ConvertibleTo<Sellp<ValueType, IndexType>>,
            public ConvertibleTo<SparsityCsr<ValueType, IndexType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public Transposable,
            public Permutable<IndexType>,
            public EnableAbsoluteComputation<
                remove_complex<Csr<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Csr>;
    friend class EnablePolymorphicObject<Csr, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Ell<ValueType, IndexType>;
    friend class Hybrid<ValueType, IndexType>;
    friend class Sellp<ValueType, IndexType>;
    friend class SparsityCsr<ValueType, IndexType>;
    friend class CsrBuilder<ValueType, IndexType>;
    friend class Csr<to_complex<ValueType>, IndexType>;

public:
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Csr<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Csr>;

    class automatical;

    /**
     * strategy_type is to decide how to set the csr algorithm.
     *
     * The practical strategy method should inherit strategy_type and implement
     * its `process`, `clac_size` function and the corresponding device kernel.
     */
    class strategy_type {
        friend class automatical;

    public:
        /**
         * Creates a strategy_type.
         *
         * @param name  the name of strategy
         */
        strategy_type(std::string name) : name_(name) {}

        /**
         * Returns the name of strategy
         *
         * @return the name of strategy
         */
        std::string get_name() { return name_; }

        /**
         * Computes srow according to row pointers.
         *
         * @param mtx_row_ptrs  the row pointers of the matrix
         * @param mtx_srow  the srow of the matrix
         */
        virtual void process(const Array<index_type> &mtx_row_ptrs,
                             Array<index_type> *mtx_srow) = 0;

        /**
         * Computes the srow size according to the number of nonzeros.
         *
         * @param nnz  the number of nonzeros
         *
         * @return the size of srow
         */
        virtual int64_t clac_size(const int64_t nnz) = 0;

        /**
         * Copy a strategy. This is a workaround until strategies are revamped,
         * since strategies like `automatical` do not work when actually shared.
         */
        virtual std::shared_ptr<strategy_type> copy() = 0;

    protected:
        void set_name(std::string name) { name_ = name; }

    private:
        std::string name_;
    };

    /**
     * classical is a strategy_type which uses the same number of threads on
     * each row. Classical strategy uses multithreads to calculate on parts of
     * rows and then do a reduction of these threads results. The number of
     * threads per row depends on the max number of stored elements per row.
     */
    class classical : public strategy_type {
    public:
        /**
         * Creates a classical strategy.
         */
        classical() : strategy_type("classical"), max_length_per_row_(0) {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {
            auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
            Array<index_type> row_ptrs_host(host_mtx_exec);
            const bool is_mtx_on_host{host_mtx_exec ==
                                      mtx_row_ptrs.get_executor()};
            const index_type *row_ptrs{};
            if (is_mtx_on_host) {
                row_ptrs = mtx_row_ptrs.get_const_data();
            } else {
                row_ptrs_host = mtx_row_ptrs;
                row_ptrs = row_ptrs_host.get_const_data();
            }
            auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
            max_length_per_row_ = 0;
            for (index_type i = 1; i < num_rows + 1; i++) {
                max_length_per_row_ = std::max(max_length_per_row_,
                                               row_ptrs[i] - row_ptrs[i - 1]);
            }
        }

        int64_t clac_size(const int64_t nnz) override { return 0; }

        index_type get_max_length_per_row() const noexcept
        {
            return max_length_per_row_;
        }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<classical>();
        }

    private:
        index_type max_length_per_row_;
    };

    /**
     * merge_path is a strategy_type which uses the merge_path algorithm.
     * merge_path is according to Merrill and Garland: Merge-Based Parallel
     * Sparse Matrix-Vector Multiplication
     */
    class merge_path : public strategy_type {
    public:
        /**
         * Creates a merge_path strategy.
         */
        merge_path() : strategy_type("merge_path") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {}

        int64_t clac_size(const int64_t nnz) override { return 0; }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<merge_path>();
        }
    };

    /**
     * cusparse is a strategy_type which uses the sparselib csr.
     *
     * @note cusparse is also known to the hip executor which converts between
     *       cuda and hip.
     */
    class cusparse : public strategy_type {
    public:
        /**
         * Creates a cusparse strategy.
         */
        cusparse() : strategy_type("cusparse") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {}

        int64_t clac_size(const int64_t nnz) override { return 0; }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<cusparse>();
        }
    };

    /**
     * sparselib is a strategy_type which uses the sparselib csr.
     *
     * @note Uses cusparse in cuda and hipsparse in hip.
     */
    class sparselib : public strategy_type {
    public:
        /**
         * Creates a sparselib strategy.
         */
        sparselib() : strategy_type("sparselib") {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {}

        int64_t clac_size(const int64_t nnz) override { return 0; }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<sparselib>();
        }
    };

    /**
     * load_balance is a strategy_type which uses the load balance algorithm.
     */
    class load_balance : public strategy_type {
    public:
        /**
         * Creates a load_balance strategy.
         */
        load_balance()
            : load_balance(std::move(
                  gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
        {}

        /**
         * Creates a load_balance strategy with CUDA executor.
         *
         * @param exec the CUDA executor
         */
        load_balance(std::shared_ptr<const CudaExecutor> exec)
            : load_balance(exec->get_num_warps(), exec->get_warp_size())
        {}

        /**
         * Creates a load_balance strategy with HIP executor.
         *
         * @param exec the HIP executor
         */
        load_balance(std::shared_ptr<const HipExecutor> exec)
            : load_balance(exec->get_num_warps(), exec->get_warp_size(), false)
        {}

        /**
         * Creates a load_balance strategy with specified parameters
         *
         * @param nwarps the number of warps in the executor
         * @param warp_size the warp size of the executor
         * @param cuda_strategy  whether the `cuda_strategy` needs to be used.
         *
         * @note The warp_size must be the size of full warp. When using this
         *       constructor, set_strategy needs to be called with correct
         *       parameters which is replaced during the conversion.
         */
        load_balance(int64_t nwarps, int warp_size = 32,
                     bool cuda_strategy = true)
            : strategy_type("load_balance"),
              nwarps_(nwarps),
              warp_size_(warp_size),
              cuda_strategy_(cuda_strategy)
        {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {
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
                        ceildiv((ceildiv(row_ptrs[i + 1], warp_size_) * nwarps),
                                ceildiv(num_elems, warp_size_));
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

        int64_t clac_size(const int64_t nnz) override
        {
            if (warp_size_ > 0) {
                int multiple = 8;
                if (nnz >= static_cast<int64_t>(2e8)) {
                    multiple = 2048;
                } else if (nnz >= static_cast<int64_t>(2e7)) {
                    multiple = 512;
                } else if (nnz >= static_cast<int64_t>(2e6)) {
                    multiple = 128;
                } else if (nnz >= static_cast<int64_t>(2e5)) {
                    multiple = 32;
                }

#if GINKGO_HIP_PLATFORM_HCC
                if (!cuda_strategy_) {
                    multiple = 8;
                    if (nnz >= static_cast<int64_t>(1e7)) {
                        multiple = 64;
                    } else if (nnz >= static_cast<int64_t>(1e6)) {
                        multiple = 16;
                    }
                }
#endif  // GINKGO_HIP_PLATFORM_HCC

                auto nwarps = nwarps_ * multiple;
                return min(ceildiv(nnz, warp_size_), int64_t(nwarps));
            } else {
                return 0;
            }
        }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<load_balance>(nwarps_, warp_size_,
                                                  cuda_strategy_);
        }

    private:
        int64_t nwarps_;
        int warp_size_;
        bool cuda_strategy_;
    };

    class automatical : public strategy_type {
    public:
        /* Use imbalance strategy when the maximum number of nonzero per row is
         * more than 1024 on NVIDIA hardware */
        const index_type nvidia_row_len_limit = 1024;
        /* Use imbalance strategy when the matrix has more more than 1e6 on
         * NVIDIA hardware */
        const index_type nvidia_nnz_limit{static_cast<index_type>(1e6)};
        /* Use imbalance strategy when the maximum number of nonzero per row is
         * more than 768 on AMD hardware */
        const index_type amd_row_len_limit = 768;
        /* Use imbalance strategy when the matrix has more more than 1e8 on AMD
         * hardware */
        const index_type amd_nnz_limit{static_cast<index_type>(1e8)};

        /**
         * Creates an automatical strategy.
         */
        automatical()
            : automatical(std::move(
                  gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
        {}

        /**
         * Creates an automatical strategy with CUDA executor.
         *
         * @param exec the CUDA executor
         */
        automatical(std::shared_ptr<const CudaExecutor> exec)
            : automatical(exec->get_num_warps(), exec->get_warp_size())
        {}

        /**
         * Creates an automatical strategy with HIP executor.
         *
         * @param exec the HIP executor
         */
        automatical(std::shared_ptr<const HipExecutor> exec)
            : automatical(exec->get_num_warps(), exec->get_warp_size(), false)
        {}

        /**
         * Creates an automatical strategy with specified parameters
         *
         * @param nwarps the number of warps in the executor
         * @param warp_size the warp size of the executor
         * @param cuda_strategy  whether the `cuda_strategy` needs to be used.
         *
         * @note The warp_size must be the size of full warp. When using this
         *       constructor, set_strategy needs to be called with correct
         *       parameters which is replaced during the conversion.
         */
        automatical(int64_t nwarps, int warp_size = 32,
                    bool cuda_strategy = true)
            : strategy_type("automatical"),
              nwarps_(nwarps),
              warp_size_(warp_size),
              cuda_strategy_(cuda_strategy),
              max_length_per_row_(0)
        {}

        void process(const Array<index_type> &mtx_row_ptrs,
                     Array<index_type> *mtx_srow) override
        {
            // if the number of stored elements is larger than <nnz_limit> or
            // the maximum number of stored elements per row is larger than
            // <row_len_limit>, use load_balance otherwise use classical
            index_type nnz_limit = nvidia_nnz_limit;
            index_type row_len_limit = nvidia_row_len_limit;
#if GINKGO_HIP_PLATFORM_HCC
            if (!cuda_strategy_) {
                nnz_limit = amd_nnz_limit;
                row_len_limit = amd_row_len_limit;
            }
#endif  // GINKGO_HIP_PLATFORM_HCC
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
            if (row_ptrs[num_rows] > nnz_limit) {
                load_balance actual_strategy(nwarps_, warp_size_,
                                             cuda_strategy_);
                if (is_mtx_on_host) {
                    actual_strategy.process(mtx_row_ptrs, mtx_srow);
                } else {
                    actual_strategy.process(row_ptrs_host, mtx_srow);
                }
                this->set_name(actual_strategy.get_name());
            } else {
                index_type maxnum = 0;
                for (index_type i = 1; i < num_rows + 1; i++) {
                    maxnum = std::max(maxnum, row_ptrs[i] - row_ptrs[i - 1]);
                }
                if (maxnum > row_len_limit) {
                    load_balance actual_strategy(nwarps_, warp_size_,
                                                 cuda_strategy_);
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
                        max_length_per_row_ =
                            actual_strategy.get_max_length_per_row();
                    } else {
                        actual_strategy.process(row_ptrs_host, mtx_srow);
                        max_length_per_row_ =
                            actual_strategy.get_max_length_per_row();
                    }
                    this->set_name(actual_strategy.get_name());
                }
            }
        }

        int64_t clac_size(const int64_t nnz) override
        {
            return std::make_shared<load_balance>(nwarps_, warp_size_,
                                                  cuda_strategy_)
                ->clac_size(nnz);
        }

        index_type get_max_length_per_row() const noexcept
        {
            return max_length_per_row_;
        }

        std::shared_ptr<strategy_type> copy() override
        {
            return std::make_shared<automatical>(nwarps_, warp_size_,
                                                 cuda_strategy_);
        }

    private:
        int64_t nwarps_;
        int warp_size_;
        bool cuda_strategy_;
        index_type max_length_per_row_;
    };

    void convert_to(Csr<ValueType, IndexType> *result) const override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        // NOTE: as soon as strategies are improved, this can be reverted
        result->values_ = this->values_;
        result->col_idxs_ = this->col_idxs_;
        result->row_ptrs_ = this->row_ptrs_;
        result->srow_ = this->srow_;
        result->set_size(this->get_size());
        if (!same_executor) {
            convert_strategy_helper(result);
        } else {
            result->set_strategy(std::move(this->get_strategy()->copy()));
        }
        // END NOTE
    }

    void move_to(Csr<ValueType, IndexType> *result) override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        EnableLinOp<Csr>::move_to(result);
        if (!same_executor) {
            detail::strategy_rebuild_helper(result);
        }
    }
    friend class Csr<next_precision<ValueType>, IndexType>;

    void convert_to(
        Csr<next_precision<ValueType>, IndexType> *result) const override;

    void move_to(Csr<next_precision<ValueType>, IndexType> *result) override;

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

    std::unique_ptr<LinOp> row_permute(
        const Array<IndexType> *permutation_indices) const override;

    std::unique_ptr<LinOp> column_permute(
        const Array<IndexType> *permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_row_permute(
        const Array<IndexType> *inverse_permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_column_permute(
        const Array<IndexType> *inverse_permutation_indices) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

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

    /**
     * Set the strategy
     *
     * @param strategy the csr strategy
     */
    void set_strategy(std::shared_ptr<strategy_type> strategy)
    {
        strategy_ = std::move(strategy->copy());
        this->make_srow();
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
        std::shared_ptr<strategy_type> strategy = std::make_shared<sparselib>())
        : EnableLinOp<Csr>(exec, size),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          row_ptrs_(exec, size[0] + 1),
          srow_(exec, strategy->clac_size(num_nonzeros)),
          strategy_(strategy->copy())
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
        std::shared_ptr<strategy_type> strategy = std::make_shared<sparselib>())
        : EnableLinOp<Csr>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)},
          srow_(exec),
          strategy_(strategy->copy())
    {
        GKO_ASSERT_EQ(values_.get_num_elems(), col_idxs_.get_num_elems());
        GKO_ASSERT_EQ(this->get_size()[0] + 1, row_ptrs_.get_num_elems());
        this->make_srow();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    // TODO clean this up as soon as we improve strategy_type
    template <typename CsrType>
    void convert_strategy_helper(CsrType *result) const
    {
        auto strat = this->get_strategy().get();
        std::shared_ptr<typename CsrType::strategy_type> new_strat;
        if (dynamic_cast<classical *>(strat)) {
            new_strat = std::make_shared<typename CsrType::classical>();
        } else if (dynamic_cast<merge_path *>(strat)) {
            new_strat = std::make_shared<typename CsrType::merge_path>();
        } else if (dynamic_cast<cusparse *>(strat)) {
            new_strat = std::make_shared<typename CsrType::cusparse>();
        } else if (dynamic_cast<sparselib *>(strat)) {
            new_strat = std::make_shared<typename CsrType::sparselib>();
        } else {
            auto rexec = result->get_executor();
            auto cuda_exec =
                std::dynamic_pointer_cast<const CudaExecutor>(rexec);
            auto hip_exec = std::dynamic_pointer_cast<const HipExecutor>(rexec);
            auto lb = dynamic_cast<load_balance *>(strat);
            if (cuda_exec) {
                if (lb) {
                    new_strat =
                        std::make_shared<typename CsrType::load_balance>(
                            cuda_exec);
                } else {
                    new_strat = std::make_shared<typename CsrType::automatical>(
                        cuda_exec);
                }
            } else if (hip_exec) {
                if (lb) {
                    new_strat =
                        std::make_shared<typename CsrType::load_balance>(
                            hip_exec);
                } else {
                    new_strat = std::make_shared<typename CsrType::automatical>(
                        hip_exec);
                }
            } else {
                // Try to preserve this executor's configuration
                auto this_cuda_exec =
                    std::dynamic_pointer_cast<const CudaExecutor>(
                        this->get_executor());
                auto this_hip_exec =
                    std::dynamic_pointer_cast<const HipExecutor>(
                        this->get_executor());
                if (this_cuda_exec) {
                    if (lb) {
                        new_strat =
                            std::make_shared<typename CsrType::load_balance>(
                                this_cuda_exec);
                    } else {
                        new_strat =
                            std::make_shared<typename CsrType::automatical>(
                                this_cuda_exec);
                    }
                } else if (this_hip_exec) {
                    if (lb) {
                        new_strat =
                            std::make_shared<typename CsrType::load_balance>(
                                this_hip_exec);
                    } else {
                        new_strat =
                            std::make_shared<typename CsrType::automatical>(
                                this_hip_exec);
                    }
                } else {
                    // We had a load balance or automatical strategy from a non
                    // HIP or Cuda executor and are moving to a non HIP or Cuda
                    // executor.
                    // FIXME this creates a long delay
                    if (lb) {
                        new_strat =
                            std::make_shared<typename CsrType::load_balance>();
                    } else {
                        new_strat =
                            std::make_shared<typename CsrType::automatical>();
                    }
                }
            }
        }
        result->set_strategy(new_strat);
    }

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


namespace detail {


/**
 * When strategy is load_balance or automatical, rebuild the strategy
 * according to executor's property.
 *
 * @param result  the csr matrix.
 */
template <typename ValueType, typename IndexType>
void strategy_rebuild_helper(Csr<ValueType, IndexType> *result)
{
    using load_balance = typename Csr<ValueType, IndexType>::load_balance;
    using automatical = typename Csr<ValueType, IndexType>::automatical;
    auto strategy = result->get_strategy();
    auto executor = result->get_executor();
    if (std::dynamic_pointer_cast<load_balance>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<load_balance>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<load_balance>(exec));
        }
    } else if (std::dynamic_pointer_cast<automatical>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<automatical>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<automatical>(exec));
        }
    }
}


}  // namespace detail
}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_CSR_HPP_
