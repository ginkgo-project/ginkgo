/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType>
class Diagonal;

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
class Fbcsr;

template <typename ValueType, typename IndexType>
class CsrBuilder;


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
            public ConvertibleTo<Fbcsr<ValueType, IndexType>>,
            public ConvertibleTo<Hybrid<ValueType, IndexType>>,
            public ConvertibleTo<Sellp<ValueType, IndexType>>,
            public ConvertibleTo<SparsityCsr<ValueType, IndexType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public Transposable,
            public Permutable<IndexType>,
            public EnableAbsoluteComputation<
                remove_complex<Csr<ValueType, IndexType>>>,
            public ScaledIdentityAddable {
    friend class EnableCreateMethod<Csr>;
    friend class EnablePolymorphicObject<Csr, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class Diagonal<ValueType>;
    friend class Ell<ValueType, IndexType>;
    friend class Hybrid<ValueType, IndexType>;
    friend class Sellp<ValueType, IndexType>;
    friend class SparsityCsr<ValueType, IndexType>;
    friend class Fbcsr<ValueType, IndexType>;
    friend class CsrBuilder<ValueType, IndexType>;
    friend class Csr<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Csr>::convert_to;
    using EnableLinOp<Csr>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Csr<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
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
        strategy_type(std::string name);

        virtual ~strategy_type() = default;

        /**
         * Returns the name of strategy
         *
         * @return the name of strategy
         */
        std::string get_name();

        /**
         * Computes srow according to row pointers.
         *
         * @param mtx_row_ptrs  the row pointers of the matrix
         * @param mtx_srow  the srow of the matrix
         */
        virtual void process(const array<index_type>& mtx_row_ptrs,
                             array<index_type>* mtx_srow) = 0;

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
        void set_name(std::string name);

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
        classical();

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        index_type get_max_length_per_row() const noexcept;

        std::shared_ptr<strategy_type> copy() override;

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
        merge_path();

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        std::shared_ptr<strategy_type> copy() override;
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
        cusparse();

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        std::shared_ptr<strategy_type> copy() override;
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
        sparselib();

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        std::shared_ptr<strategy_type> copy() override;
    };

    /**
     * load_balance is a strategy_type which uses the load balance algorithm.
     */
    class load_balance : public strategy_type {
    public:
        /**
         * Creates a load_balance strategy.
         *
         * @warning this is deprecated! Please rely on the new automatic
         *          strategy instantiation or use one of the other constructors.
         */
        [[deprecated]] load_balance();

        /**
         * Creates a load_balance strategy with CUDA executor.
         *
         * @param exec the CUDA executor
         */
        load_balance(std::shared_ptr<const CudaExecutor> exec);

        /**
         * Creates a load_balance strategy with HIP executor.
         *
         * @param exec the HIP executor
         */
        load_balance(std::shared_ptr<const HipExecutor> exec);

        /**
         * Creates a load_balance strategy with DPCPP executor.
         *
         * @param exec the DPCPP executor
         *
         * @note TODO: porting - we hardcode the subgroup size is 32 and the
         *             number of threads in a SIMD unit is 7
         */
        load_balance(std::shared_ptr<const DpcppExecutor> exec);

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
                     bool cuda_strategy = true,
                     std::string strategy_name = "none");

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        std::shared_ptr<strategy_type> copy() override;

    private:
        int64_t nwarps_;
        int warp_size_;
        bool cuda_strategy_;
        std::string strategy_name_;
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
        /* Use imbalance strategy when the maximum number of nonzero per row is
         * more than 25600 on Intel hardware */
        const index_type intel_row_len_limit = 25600;
        /* Use imbalance strategy when the matrix has more more than 3e8 on
         * Intel hardware */
        const index_type intel_nnz_limit{static_cast<index_type>(3e8)};

    public:
        /**
         * Creates an automatical strategy.
         *
         * @warning this is deprecated! Please rely on the new automatic
         *          strategy instantiation or use one of the other constructors.
         */
        [[deprecated]] automatical();

        /**
         * Creates an automatical strategy with CUDA executor.
         *
         * @param exec the CUDA executor
         */
        automatical(std::shared_ptr<const CudaExecutor> exec);

        /**
         * Creates an automatical strategy with HIP executor.
         *
         * @param exec the HIP executor
         */
        automatical(std::shared_ptr<const HipExecutor> exec);

        /**
         * Creates an automatical strategy with Dpcpp executor.
         *
         * @param exec the Dpcpp executor
         *
         * @note TODO: porting - we hardcode the subgroup size is 32 and the
         *             number of threads in a SIMD unit is 7
         */
        automatical(std::shared_ptr<const DpcppExecutor> exec);

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
                    bool cuda_strategy = true,
                    std::string strategy_name = "none");

        void process(const array<index_type>& mtx_row_ptrs,
                     array<index_type>* mtx_srow) override;

        int64_t clac_size(const int64_t nnz) override;

        index_type get_max_length_per_row() const noexcept;

        std::shared_ptr<strategy_type> copy() override;

    private:
        int64_t nwarps_;
        int warp_size_;
        bool cuda_strategy_;
        std::string strategy_name_;
        index_type max_length_per_row_;
    };

    friend class Csr<next_precision<ValueType>, IndexType>;

    void convert_to(
        Csr<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Csr<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    void convert_to(Coo<ValueType, IndexType>* result) const override;

    void move_to(Coo<ValueType, IndexType>* result) override;

    void convert_to(Ell<ValueType, IndexType>* result) const override;

    void move_to(Ell<ValueType, IndexType>* result) override;

    void convert_to(Fbcsr<ValueType, IndexType>* result) const override;

    void move_to(Fbcsr<ValueType, IndexType>* result) override;

    void convert_to(Hybrid<ValueType, IndexType>* result) const override;

    void move_to(Hybrid<ValueType, IndexType>* result) override;

    void convert_to(Sellp<ValueType, IndexType>* result) const override;

    void move_to(Sellp<ValueType, IndexType>* result) override;

    void convert_to(SparsityCsr<ValueType, IndexType>* result) const override;

    void move_to(SparsityCsr<ValueType, IndexType>* result) override;

    void read(const mat_data& data) override;

    void read(const device_mat_data& data) override;

    void read(device_mat_data&& data) override;

    void write(mat_data& data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    std::unique_ptr<LinOp> permute(
        const array<IndexType>* permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_permute(
        const array<IndexType>* inverse_permutation_indices) const override;

    std::unique_ptr<LinOp> row_permute(
        const array<IndexType>* permutation_indices) const override;

    std::unique_ptr<LinOp> column_permute(
        const array<IndexType>* permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_row_permute(
        const array<IndexType>* inverse_permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_column_permute(
        const array<IndexType>* inverse_permutation_indices) const override;

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
    value_type* get_values() noexcept;

    /**
     * @copydoc Csr::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values() const noexcept;

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type* get_col_idxs() noexcept;

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_col_idxs() const noexcept;

    /**
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type* get_row_ptrs() noexcept;

    /**
     * @copydoc Csr::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_ptrs() const noexcept;

    /**
     * Returns the starting rows.
     *
     * @return the starting rows.
     */
    index_type* get_srow() noexcept;

    /**
     * @copydoc Csr::get_srow()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_srow() const noexcept;

    /**
     * Returns the number of the srow stored elements (involved warps)
     *
     * @return the number of the srow stored elements (involved warps)
     */
    size_type get_num_srow_elements() const noexcept;

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept;

    /** Returns the strategy
     *
     * @return the strategy
     */
    std::shared_ptr<strategy_type> get_strategy() const noexcept;

    /**
     * Set the strategy
     *
     * @param strategy the csr strategy
     */
    void set_strategy(std::shared_ptr<strategy_type> strategy);

    /**
     * Scales the matrix with a scalar.
     *
     * @param alpha  The entire matrix is scaled by alpha. alpha has to be a 1x1
     * Dense matrix.
     */
    void scale(const LinOp* alpha);

    /**
     * Scales the matrix with the inverse of a scalar.
     *
     * @param alpha  The entire matrix is scaled by 1 / alpha. alpha has to be a
     * 1x1 Dense matrix.
     */
    void inv_scale(const LinOp* alpha);

    /**
     * Creates a constant (immutable) Csr matrix from a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param row_ptrs  the row pointer array of the matrix
     * @param strategy  the strategy the matrix uses for SpMV operations
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          these arrays on the correct executor.
     */
    static std::unique_ptr<const Csr> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_ptrs,
        std::shared_ptr<strategy_type> strategy);

    /**
     * This is version of create_const with a default strategy.
     */
    static std::unique_ptr<const Csr> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_ptrs);

    /**
     * Creates a submatrix from this Csr matrix given row and column index_set
     * objects.
     *
     * @param row_index_set  the row index set containing the set of rows to be
     *                       in the submatrix.
     * @param column_index_set  the col index set containing the set of columns
     *                          to be in the submatrix.
     * @return A new CSR matrix with the elements that belong to the row and
     *          columns of this matrix as specified by the index sets.
     * @note This is not a view but creates a new, separate CSR matrix.
     */
    std::unique_ptr<Csr<ValueType, IndexType>> create_submatrix(
        const index_set<IndexType>& row_index_set,
        const index_set<IndexType>& column_index_set) const;

    /**
     * Creates a submatrix from this Csr matrix given row and column spans
     *
     * @param row_span  the row span containing the contiguous set of rows to be
     *                  in the submatrix.
     * @param column_span  the column span containing the contiguous set of
     *                     columns to be in the submatrix.
     * @return A new CSR matrix with the elements that belong to the row and
     *          columns of this matrix as specified by the index sets.
     * @note This is not a view but creates a new, separate CSR matrix.
     */
    std::unique_ptr<Csr<ValueType, IndexType>> create_submatrix(
        const span& row_span, const span& column_span) const;

    /**
     * Copy-assigns a Csr matrix. Preserves executor, copies everything else.
     */
    Csr& operator=(const Csr&);

    /**
     * Move-assigns a Csr matrix. Preserves executor, moves the data and leaves
     * the moved-from object in an empty state (0x0 LinOp with unchanged
     * executor and strategy, no nonzeros and valid row pointers).
     */
    Csr& operator=(Csr&&);

    /**
     * Copy-constructs a Csr matrix. Inherits executor, strategy and data.
     */
    Csr(const Csr&);

    /**
     * Move-constructs a Csr matrix. Inherits executor and strategy, moves the
     * data and leaves the moved-from object in an empty state (0x0 LinOp with
     * unchanged executor and strategy, no nonzeros and valid row pointers).
     */
    Csr(Csr&&);

protected:
    /**
     * Creates an uninitialized CSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  the strategy of CSR
     */
    Csr(std::shared_ptr<const Executor> exec,
        std::shared_ptr<strategy_type> strategy);

    /**
     * Creates an uninitialized CSR matrix of the specified size with a user
     * chosen strategy.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param strategy  the strategy of CSR
     */
    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type num_nonzeros, std::shared_ptr<strategy_type> strategy);

    /**
     * Creates an uninitialized CSR matrix of the specified size with a
     * default strategy.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{},
        size_type num_nonzeros = {});

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
     * @param strategy  the strategy the matrix uses for SpMV operations
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size,
        ValuesArray&& values, ColIdxsArray&& col_idxs, RowPtrsArray&& row_ptrs,
        std::shared_ptr<strategy_type> strategy)
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

    /**
     * Creates a CSR matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @note This is the same as the previous constructor but with a default
     *       strategy.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size,
        ValuesArray&& values, ColIdxsArray&& col_idxs, RowPtrsArray&& row_ptrs)
        : Csr{exec,
              size,
              std::forward<ValuesArray>(values),
              std::forward<ColIdxsArray>(col_idxs),
              std::forward<RowPtrsArray>(row_ptrs),
              Csr::make_default_strategy(exec)}
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    // TODO: This provides some more sane settings. Please fix this!
    static std::shared_ptr<strategy_type> make_default_strategy(
        std::shared_ptr<const Executor> exec);

    // TODO clean this up as soon as we improve strategy_type
    template <typename CsrType>
    void convert_strategy_helper(CsrType* result) const;

    /**
     * Computes srow. It should be run after changing any row_ptrs_ value.
     */
    void make_srow();

    /**
     * @copydoc scale(const LinOp *)
     *
     * @note  Other implementations of Csr should override this function
     *        instead of scale(const LinOp *alpha).
     */
    virtual void scale_impl(const LinOp* alpha);

    /**
     * @copydoc inv_scale(const LinOp *)
     *
     * @note  Other implementations of Csr should override this function
     *        instead of inv_scale(const LinOp *alpha).
     */
    virtual void inv_scale_impl(const LinOp* alpha);

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;
    array<index_type> srow_;
    std::shared_ptr<strategy_type> strategy_;

    void add_scaled_identity_impl(const LinOp* a, const LinOp* b) override;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_CSR_HPP_
