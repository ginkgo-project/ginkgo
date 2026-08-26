// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_CSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_CSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>


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


namespace csr {


/**
 * Type describes the Csr SpMV strategy.
 */
enum class spmv_strategy {
    /**
     * automatic is the strategy choosing between load_balance and classical
     * based on the maximum number of entries per row and the number of entries
     * of the matrix.
     */
    automatic,
    /**
     * load_balance is the strategy trying to distribute the work equally in
     * terms of the number of matrix entries. More detail can be checked in
     * Goran and Enrique: Balanced CSR sparse matrix-vector product on graphics
     * processors (doi: 10.1007/978-3-319-64203-1_50).
     */
    load_balance,
    /**
     * merge_path is the strategy trying to distribute the work equally in terms
     * of the number of matrix entries and row pointers. More detail can be
     * checked in Merrill and Garland: Merge-Based Parallel Sparse Matrix-Vector
     * Multiplication (doi: 10.1109/SC.2016.57).
     */
    merge_path,
    /**
     * classical is the strategy assigning the same amount of the working
     * resource to each row.
     */
    classical,
    /**
     * sparselib is the strategy calling the backend sparse library
     * implementation when it is supported.
     *
     * - reference/omp: ginkgo's classical spmv
     * - cuda: cuSPARSE
     * - hip: hipSPARSE
     * - dpcpp: oneMKL
     */
    sparselib
};


}  // namespace csr


/**
 * CSR is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format).
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column of each
 * nonzero element.
 *
 * The Csr LinOp supports three families of `apply` operations,
 * dispatched on the type of the right operand:
 *
 * - Against a `Dense` operand `b`, `apply` computes a sparse matrix-vector
 *   (or matrix-multivector) product:
 *   \f[ x = A b, \qquad x = \alpha A b + \beta x. \f]
 *
 * - Against another `Csr` operand `B`, `apply` computes a sparse-sparse
 *   matrix product (SpGEMM):
 *   \f[ C = A B, \qquad C = \alpha A B + \beta C. \f]
 *
 * - Against an `Identity` operand, `apply` reduces to a sparse-sparse
 *   matrix addition (SpGEAM):
 *   \f[ B = \alpha A + \beta B. \f]
 *
 * In code:
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
class Csr : public LinOp,
            public EnableCloneable<Csr<ValueType, IndexType>>,
            public ConvertibleTo<Csr<next_precision<ValueType>, IndexType>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
            public ConvertibleTo<Csr<next_precision<ValueType, 2>, IndexType>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
            public ConvertibleTo<Csr<next_precision<ValueType, 3>, IndexType>>,
#endif
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
    friend class EnableCloneable<Csr>;
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
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using EnableCloneable<Csr>::convert_to;
    using EnableCloneable<Csr>::move_to;
    using ConvertibleTo<Csr<next_precision<ValueType>, IndexType>>::convert_to;
    using ConvertibleTo<Csr<next_precision<ValueType>, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ConvertibleTo<Coo<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Coo<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Ell<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Ell<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Sellp<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, IndexType>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, IndexType>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, IndexType>>::move_to;
    using ReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Csr<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Csr>;
    using device_view = view::csr<value_type, index_type>;
    using const_device_view = view::csr<const value_type, const index_type>;

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::<strategy>")
        strategy_type {
    public:
        virtual ~strategy_type() = default;

        // return the corresponding enum in incoming release
        virtual csr::spmv_strategy get_enum() const = 0;
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::classical") classical
        : public strategy_type {
    public:
        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::classical;
        }
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::merge_path")
        merge_path : public strategy_type {
    public:
        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::merge_path;
        }
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::sparselib") cusparse
        : public strategy_type {
    public:
        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::sparselib;
        }
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::sparselib") sparselib
        : public strategy_type {
    public:
        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::sparselib;
        }
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::load_balance")
        load_balance : public strategy_type {
    public:
        load_balance(std::shared_ptr<const Executor>) {}

        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::load_balance;
        }
    };

    class GKO_DEPRECATED(
        "please use enum gko::matrix::csr::spmv_strategy::automatic")
        automatical : public strategy_type {
    public:
        automatical(std::shared_ptr<const Executor>) {}

        csr::spmv_strategy get_enum() const override
        {
            return csr::spmv_strategy::automatic;
        }
    };


    friend class Csr<previous_precision<ValueType>, IndexType>;

    void convert_to(
        Csr<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(Csr<next_precision<ValueType>, IndexType>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class Csr<previous_precision<ValueType, 2>, IndexType>;
    using ConvertibleTo<
        Csr<next_precision<ValueType, 2>, IndexType>>::convert_to;
    using ConvertibleTo<Csr<next_precision<ValueType, 2>, IndexType>>::move_to;

    void convert_to(
        Csr<next_precision<ValueType, 2>, IndexType>* result) const override;

    void move_to(Csr<next_precision<ValueType, 2>, IndexType>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class Csr<previous_precision<ValueType, 3>, IndexType>;
    using ConvertibleTo<
        Csr<next_precision<ValueType, 3>, IndexType>>::convert_to;
    using ConvertibleTo<Csr<next_precision<ValueType, 3>, IndexType>>::move_to;

    void convert_to(
        Csr<next_precision<ValueType, 3>, IndexType>* result) const override;

    void move_to(Csr<next_precision<ValueType, 3>, IndexType>* result) override;
#endif

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

    /**
     * Returns a non-owning device view of this matrix.
     *
     * @return a device view of this matrix.
     */
    device_view get_device_view();

    /**
     * Returns a non-owning const device view of this matrix.
     *
     * @return a const device view of this matrix.
     */
    const_device_view get_const_device_view() const;

    /**
     * Class describing the internal lookup structures created by
     * multiply_reuse(const Csr*) to recompute a sparse matrix-matrix product
     * with updated values.
     */
    class multiply_reuse_info {
        friend class Csr;

    public:
        explicit multiply_reuse_info();

        ~multiply_reuse_info();

        multiply_reuse_info(const multiply_reuse_info&) = delete;

        multiply_reuse_info(multiply_reuse_info&&) noexcept;

        multiply_reuse_info& operator=(const multiply_reuse_info&) = delete;

        multiply_reuse_info& operator=(multiply_reuse_info&&) noexcept;

        /**
         * Recomputes the sparse matrix-matrix product `out = mtx1 * mtx2` when
         * only the values of mtx1 and mtx2 changed, but the sparsity patterns
         * of mtx1, mtx2 and out are unchanged.
         */
        void update_values(ptr_param<const Csr> mtx1, ptr_param<const Csr> mtx2,
                           ptr_param<Csr> out) const;

    private:
        struct lookup_data;

        explicit multiply_reuse_info(std::unique_ptr<lookup_data> data);

        std::unique_ptr<lookup_data> internal;
    };

    /**
     * Computes the sparse matrix product `this * other` on the executor of this
     * matrix.
     *
     * @param other  the matrix with which the product will be computed.
     *               It needs to be sorted by column indices when using
     *               OmpExecutor or DpcppExecutor for `this`.
     * @return  the product of the two matrices, stored on the same executor as
     *          this matrix.
     */
    std::unique_ptr<Csr> multiply(ptr_param<const Csr> other) const;

    /**
     * Computes the sparse matrix product `this * other` on the executor of this
     * matrix, and necessary data for value updates:
     * ```
     * auto [C, reuse] = A->multiply_reuse(B);
     * change_values(A, B);
     * reuse->update_values(A, B, C);
     * ```
     *
     * @param other  the matrix with which the product will be computed.
     *               It needs to be sorted by column indices when using
     *               OmpExecutor or DpcppExecutor for `this`.
     * @return  std::pair containing the product of the two matrices, stored on
     *          the same executor as this matrix, and a multiply_reuse_info
     *          object allowing value updates to the output matrix.
     */
    std::pair<std::unique_ptr<Csr>, multiply_reuse_info> multiply_reuse(
        ptr_param<const Csr> other) const;

    /**
     * Class describing the internal lookup structures created by
     * multiply_add_reuse to recompute a sparse matrix-matrix product
     * with updated values.
     */
    class multiply_add_reuse_info {
        friend class Csr;

    public:
        explicit multiply_add_reuse_info();

        ~multiply_add_reuse_info();

        multiply_add_reuse_info(const multiply_add_reuse_info&) = delete;

        multiply_add_reuse_info(multiply_add_reuse_info&&) noexcept;

        multiply_add_reuse_info& operator=(const multiply_add_reuse_info&) =
            delete;

        multiply_add_reuse_info& operator=(multiply_add_reuse_info&&) noexcept;

        /**
         * Recomputes the sparse matrix-matrix product
         * `out = scale_mult * mtx * mtx_mult + scale_add * mtx_add` when only
         * the values of mtx, scale_mult, mtx_mult, scale_add, mtx_add changed,
         * but the sparsity patterns of mtx, mtx_mult, mtx_add and out are
         * unchanged.
         */
        void update_values(ptr_param<const Csr> mtx,
                           ptr_param<const Dense<value_type>> scale_mult,
                           ptr_param<const Csr> mtx_mult,
                           ptr_param<const Dense<value_type>> scale_add,
                           ptr_param<const Csr> mtx_add,
                           ptr_param<Csr> out) const;

    private:
        struct lookup_data;

        explicit multiply_add_reuse_info(std::unique_ptr<lookup_data> data);

        std::unique_ptr<lookup_data> internal;
    };

    /**
     * Computes the sparse matrix product
     * `scale_mult * this * mtx_mult + scale_add * mtx_add` on the executor of
     * this matrix.
     *
     * @param scale_mult  the scalar by which the matrix product will be scaled.
     * @param mtx_mult    the matrix with which the product will be computed. It
     *                    needs to be sorted by column indices when using
     *                    OmpExecutor or DpcppExecutor for `this`.
     * @param scale_add   the scalar by which the matrix mtx_add will be scaled.
     * @param mtx_add     the matrix which will be added to the product, scaled
     *                    by scale_add.
     * @return  the result of the computation, stored on the same executor as
     *          this matrix.
     */
    std::unique_ptr<Csr> multiply_add(
        ptr_param<const Dense<value_type>> scale_mult,
        ptr_param<const Csr> mtx_mult,
        ptr_param<const Dense<value_type>> scale_add,
        ptr_param<const Csr> mtx_add) const;

    /**
     * Computes the sparse matrix product
     * `scale_mult * this * mtx_mult + scale_add * mtx_add` on the executor of
     * this matrix, and necessary data for value updates:
     * ```
     * auto [result, reuse] = mtx->multiply_add_reuse(sm, mm, sa, ma);
     * change_values(mtx, sm, mm, sa, ma);
     * reuse->update_values(mtx, sm, mm, sa, ma, result);
     * ```
     *
     * @param scale_mult  the scalar by which the matrix product will be scaled.
     * @param mtx_mult    the matrix with which the product will be computed. It
     *                    needs to be sorted by column indices when using
     *                    OmpExecutor or DpcppExecutor for `this`.
     * @param scale_add   the scalar by which the matrix mtx_add will be scaled.
     * @param mtx_add     the matrix which will be added to the product, scaled
     *                    by scale_add.
     * @return  std::pair containing the result of the computation, stored on
     *          the same executor as this matrix, and a multiply_add_reuse_info
     *          object allowing value updates to the output matrix.
     */
    std::pair<std::unique_ptr<Csr>, multiply_add_reuse_info> multiply_add_reuse(
        ptr_param<const Dense<value_type>> scale_mult,
        ptr_param<const Csr> mtx_mult,
        ptr_param<const Dense<value_type>> scale_add,
        ptr_param<const Csr> mtx_add) const;

    /**
     * Class describing the internal lookup structures created by
     * scale_add_reuse to recompute a sparse matrix-matrix sum
     * with updated values.
     */
    class scale_add_reuse_info {
        friend class Csr;

    public:
        explicit scale_add_reuse_info();

        ~scale_add_reuse_info();

        scale_add_reuse_info(const scale_add_reuse_info&) = delete;

        scale_add_reuse_info(scale_add_reuse_info&&) noexcept;

        scale_add_reuse_info& operator=(const scale_add_reuse_info&) = delete;

        scale_add_reuse_info& operator=(scale_add_reuse_info&&) noexcept;

        /**
         * Recomputes the sparse matrix-matrix sum
         * `out = scale1 * mtx1 + scale2 * mtx2` when only the values of
         * mtx1, scale1, mtx2, scale2 changed, but the sparsity patterns of
         * mtx1, mtx2 and out are unchanged.
         */
        void update_values(ptr_param<const Dense<value_type>> scale1,
                           ptr_param<const Csr> mtx1,
                           ptr_param<const Dense<value_type>> scale2,
                           ptr_param<const Csr> mtx2, ptr_param<Csr> out) const;

    private:
        struct lookup_data;

        explicit scale_add_reuse_info(std::unique_ptr<lookup_data> data);

        std::unique_ptr<lookup_data> internal;
    };

    /**
     * Computes the sparse matrix sum
     * `scale_this * this + scale_other * mtx_add` on the executor of this
     * matrix. This matrix needs to be sorted by column index, otherwise the
     * result will be incorrect.
     *
     * @param scale_this   the scalar by which this matrix will be scaled.
     * @param scale_other  the scalar by which this matrix will be scaled.
     * @param mtx_other    the matrix which will be added to this, scaled by
     *                     scale_other. It needs to be sorted by column index,
     *                     otherwise the result will be incorrect.
     * @return  the result of the computation, stored on the same executor as
     *          this matrix.
     */
    std::unique_ptr<Csr> scale_add(
        ptr_param<const Dense<value_type>> scale_this,
        ptr_param<const Dense<value_type>> scale_other,
        ptr_param<const Csr> mtx_other) const;

    /**
     * Computes the sparse matrix sum
     * `scale_this * this + scale_other * mtx_add` on the executor of this
     * matrix, and necessary data for value updates:
     * ```
     * auto [result, reuse] = mtx->add_scale_reuse(alpha, beta, mtx2);
     * change_values(alpha, mtx, beta, mtx2);
     * reuse->update_values(alpha, mtx, beta, mtx2, result);
     * ```
     * This matrix needs to be sorted by column index, otherwise the
     * result will be incorrect.
     *
     * @param scale_this   the scalar by which this matrix will be scaled.
     * @param scale_other  the scalar by which this matrix will be scaled.
     * @param mtx_other    the matrix which will be added to this, scaled by
     *                     scale_other. It needs to be sorted by column index,
     *                     otherwise the result will be incorrect.
     * @return  std::pair containing the result of the computation, stored on
     *          the same executor as this matrix, and a scale_add_reuse_info
     *          object allowing value updates to the output matrix.
     */
    std::pair<std::unique_ptr<Csr>, scale_add_reuse_info> add_scale_reuse(
        ptr_param<const Dense<value_type>> scale_this,
        ptr_param<const Dense<value_type>> scale_other,
        ptr_param<const Csr> mtx_other) const;

    /**
     * A struct describing a transformation of the matrix that reorders the
     * values of the matrix into the transformed matrix.
     */
    struct permuting_reuse_info {
        /** Creates an empty reuse info. */
        explicit permuting_reuse_info();

        /** Creates a reuse info structure from its value permutation. */
        explicit permuting_reuse_info(
            std::unique_ptr<Permutation<index_type>> value_permutation);

        /**
         * Propagates the values from an input matrix to the transformed matrix.
         * The output matrix needs to have been computed using the
         * transformation that was also used to generate this reuse data.
         * Internally, this permutes the input value vector into the output
         * value vector.
         */
        void update_values(ptr_param<const Csr> input,
                           ptr_param<Csr> output) const;

        std::unique_ptr<Permutation<IndexType>> value_permutation;
    };

    /**
     * Computes the necessary data to update a transposed matrix from its
     * original matrix.
     * ```
     * auto [transposed, reuse] = matrix->transpose_reuse();
     * change_values(matrix);
     * reuse->update_values(matrix, transposed);
     * ```
     * @return an std::pair consisting of the transposed matrix and a reuse info
     *         struct that can be used to update values in the transposed
     *         matrix.
     */
    std::pair<std::unique_ptr<Csr>, permuting_reuse_info> transpose_reuse()
        const;

    /**
     * Creates a permuted copy \f$A'\f$ of this matrix \f$A\f$ with the given
     * permutation \f$P\f$. By default, this computes a symmetric permutation
     * (permute_mode::symmetric). For the effect of the different permutation
     * modes, see @ref permute_mode
     *
     * @param permutation  The input permutation.
     * @param mode  The permutation mode. If permute_mode::inverse is set, we
     *              use the inverse permutation \f$P^{-1}\f$ instead of \f$P\f$.
     *              If permute_mode::rows is set, the rows will be permuted.
     *              If permute_mode::columns is set, the columns will be
     *              permuted.
     * @return  The permuted matrix.
     */
    std::unique_ptr<Csr> permute(
        ptr_param<const Permutation<index_type>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Creates a non-symmetrically permuted copy \f$A'\f$ of this matrix \f$A\f$
     * with the given row and column permutations \f$P\f$ and \f$Q\f$. The
     * operation will compute \f$A'(i, j) = A(p[i], q[j])\f$, or \f$A' = P A
     * Q^T\f$ if `invert` is `false`, and \f$A'(p[i], q[j]) = A(i,j)\f$, or
     * \f$A' = P^{-1} A Q^{-T}\f$ if `invert` is `true`.
     *
     * @param row_permutation  The permutation \f$P\f$ to apply to the rows
     * @param column_permutation  The permutation \f$Q\f$ to apply to the
     * columns
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses \f$P^{-1}, Q^{-1}\f$
     * @return  The permuted matrix.
     */
    std::unique_ptr<Csr> permute(
        ptr_param<const Permutation<index_type>> row_permutation,
        ptr_param<const Permutation<index_type>> column_permutation,
        bool invert = false) const;

    /**
     * Computes the operations necessary to propagate changed values from a
     * matrix A to a permuted matrix.
     * The semantics of this function match those of
     * permute(ptr_param<const Permutation<index_type>>, permute_mode).
     * Updating values works as follows:
     * ```
     * auto [permuted, reuse] = matrix->permute_reuse(permutation, mode);
     * change_values(matrix);
     * reuse->update_values(matrix, permuted);
     * ```
     * @param permutation  The input permutation.
     * @param mode  The permutation mode. If permute_mode::inverse is set, we
     *              use the inverse permutation \f$P^{-1}\f$ instead of \f$P\f$.
     *              If permute_mode::rows is set, the rows will be permuted.
     *              If permute_mode::columns is set, the columns will be
     *              permuted.
     * @return an std::pair consisting of the permuted matrix and the reuse info
     *         that can be used to update values in the permuted matrix.
     */
    std::pair<std::unique_ptr<Csr>, permuting_reuse_info> permute_reuse(
        ptr_param<const Permutation<index_type>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Computes the operations necessary to propagate changed values from a
     * matrix A to a permuted matrix.
     * The semantics of this function match those of
     * permute(ptr_param<const Permutation<index_type>>, ptr_param<const
     * Permutation<index_type>>, bool). Updating values works as follows:
     * ```
     * auto [permuted, reuse] = matrix->permute_reuse(row_perm, col_perm, inv);
     * change_values(matrix);
     * reuse->update_values(matrix, permuted);
     * ```
     * @param row_permutation  The permutation \f$P\f$ to apply to the rows
     * @param column_permutation  The permutation \f$Q\f$ to apply to the
     * columns
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses \f$P^{-1}, Q^{-1}\f$
     * @return an std::pair consisting of the permuted matrix and the reuse info
     *         that can be used to update values in the permuted matrix.
     */
    std::pair<std::unique_ptr<Csr>, permuting_reuse_info> permute_reuse(
        ptr_param<const Permutation<index_type>> row_permutation,
        ptr_param<const Permutation<index_type>> column_permutation,
        bool invert = false) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the permutation modes, see
     * @ref permute(ptr_param<const Permutation<index_type>>, permute_mode)
     *
     * @param permutation  The scaled permutation.
     * @param mode  The permutation mode.
     * @return The permuted matrix.
     */
    std::unique_ptr<Csr> scale_permute(
        ptr_param<const ScaledPermutation<value_type, index_type>> permutation,
        permute_mode = permute_mode::symmetric) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the parameters, see
     * @ref permute(ptr_param<const Permutation<index_type>>, ptr_param<const
     * Permutation<index_type>>, permute_mode)
     *
     * @param row_permutation  The scaled row permutation.
     * @param column_permutation  The scaled column permutation.
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses \f$P^{-1}, Q^{-1}\f$
     * @return The permuted matrix.
     */
    std::unique_ptr<Csr> scale_permute(
        ptr_param<const ScaledPermutation<value_type, index_type>>
            row_permutation,
        ptr_param<const ScaledPermutation<value_type, index_type>>
            column_permutation,
        bool invert = false) const;

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
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Csr::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Creates a Dense view of the value array of this matrix as a column
     * vector of dimensions nnz x 1.
     */
    std::unique_ptr<Dense<ValueType>> create_value_view();

    /**
     * Creates a const Dense view of the value array of this matrix as a column
     * vector of dimensions nnz x 1.
     */
    std::unique_ptr<const Dense<ValueType>> create_const_value_view() const;

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type* get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc Csr::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * Returns the starting rows.
     *
     * @return the starting rows.
     */
    index_type* get_srow() noexcept { return srow_.get_data(); }

    /**
     * @copydoc Csr::get_srow()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_srow() const noexcept
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
        return srow_.get_size();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_size();
    }

    /**
     * Returns the strategy
     *
     * @return the strategy
     */
    csr::spmv_strategy get_strategy() const noexcept;

    /**
     * Set the strategy
     *
     * @param strategy the csr strategy
     */
    void set_strategy(csr::spmv_strategy strategy)
    {
        strategy_ = strategy;
        this->make_srow();
    }

    /**
     * Scales the matrix with a scalar.
     *
     * @param alpha  The entire matrix is scaled by alpha. alpha has to be a 1x1
     * Dense matrix.
     */
    void scale(ptr_param<const LinOp> alpha)
    {
        auto exec = this->get_executor();
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        this->scale_impl(make_temporary_clone(exec, alpha).get());
    }

    /**
     * Scales the matrix with the inverse of a scalar.
     *
     * @param alpha  The entire matrix is scaled by 1 / alpha. alpha has to be a
     * 1x1 Dense matrix.
     */
    void inv_scale(ptr_param<const LinOp> alpha)
    {
        auto exec = this->get_executor();
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        this->inv_scale_impl(make_temporary_clone(exec, alpha).get());
    }

    void validate_data() const override;

    /**
     * Creates an uninitialized CSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  the strategy of CSR
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Csr> create(std::shared_ptr<const Executor> exec,
                                       csr::spmv_strategy strategy);

    /**
     * Creates an uninitialized CSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param strategy  the strategy the matrix uses for SpMV operations,
     *                  default is automatic.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Csr> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = {},
        size_type num_nonzeros = {},
        csr::spmv_strategy strategy = csr::spmv_strategy::automatic);

    /**
     * Creates a CSR matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     * @param strategy  the strategy the matrix uses for SpMV operations,
     *                  default is automatic.
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Csr> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<value_type> values, array<index_type> col_idxs,
        array<index_type> row_ptrs,
        csr::spmv_strategy strategy = csr::spmv_strategy::automatic);

    /**
     * @copydoc std::unique_ptr<Csr> create(std::shared_ptr<const Executor>,
     * const dim<2>&, array<value_type>, array<index_type>, array<index_type>)
     */
    template <typename InputValueType, typename InputColumnIndexType,
              typename InputRowPtrType>
    GKO_DEPRECATED(
        "explicitly construct the gko::array argument instead of passing "
        "initializer lists")
    static std::unique_ptr<Csr> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        std::initializer_list<InputValueType> values,
        std::initializer_list<InputColumnIndexType> col_idxs,
        std::initializer_list<InputRowPtrType> row_ptrs)
    {
        return create(exec, size, array<value_type>{exec, std::move(values)},
                      array<index_type>{exec, std::move(col_idxs)},
                      array<index_type>{exec, std::move(row_ptrs)});
    }

    /**
     * Creates a constant (immutable) Csr matrix from a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param row_ptrs  the row pointer array of the matrix
     * @param strategy  the strategy the matrix uses for SpMV operations,
     *                  default is automatic.
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          these arrays on the correct executor.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<const Csr> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_ptrs,
        csr::spmv_strategy strategy = csr::spmv_strategy::automatic);

    GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS

    /**
     * @copydoc std::unique_ptr<Csr> create(std::shared_ptr<const Executor>,
     * csr::spmv_strategy)
     */
    [[deprecated("please use enum version")]] static std::unique_ptr<Csr>
    create(std::shared_ptr<const Executor> exec,
           std::shared_ptr<strategy_type> strategy);

    /**
     * @copydoc std::unique_ptr<Csr> create(std::shared_ptr<const Executor>,
     * const dim<2>&, array<value_type>, array<index_type>, array<index_type>,
     * csr::spmv_strategy)
     */
    [[deprecated("please use enum version")]] static std::unique_ptr<Csr>
    create(std::shared_ptr<const Executor> exec, const dim<2>& size,
           array<value_type> values, array<index_type> col_idxs,
           array<index_type> row_ptrs, std::shared_ptr<strategy_type> strategy);

    /**
     * @copydoc std::unique_ptr<const Csr> create_const(std::shared_ptr<const
     * Executor>, const dim<2>&, gko::detail::const_array_view<ValueType>&&,
     * gko::detail::const_array_view<IndexType>&&,
     * gko::detail::const_array_view<IndexType>&&, csr::spmv_strategy)
     */
    [[deprecated("please use enum version")]] static std::unique_ptr<const Csr>
    create_const(std::shared_ptr<const Executor> exec, const dim<2>& size,
                 gko::detail::const_array_view<ValueType>&& values,
                 gko::detail::const_array_view<IndexType>&& col_idxs,
                 gko::detail::const_array_view<IndexType>&& row_ptrs,
                 std::shared_ptr<strategy_type> strategy);

    GKO_END_DISABLE_DEPRECATION_WARNINGS

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
    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size = {},
        size_type num_nonzeros = {},
        csr::spmv_strategy strategy = csr::spmv_strategy::automatic);

    Csr(std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<value_type> values, array<index_type> col_idxs,
        array<index_type> row_ptrs,
        csr::spmv_strategy strategy = csr::spmv_strategy::automatic);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

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

    /**
     * Returns the actual strategy. When the strategy is automatic, this
     * returns the actual underlying strategy. This returns the same strategy as
     * `get_strategy` when the strategy is not automatic.
     *
     * @return the actual strategy
     */
    csr::spmv_strategy get_actual_strategy() const noexcept;

private:
    csr::spmv_strategy strategy_;
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;
    array<index_type> srow_;
    index_type max_nnz_per_row_;

    void add_scaled_identity_impl(const LinOp* a, const LinOp* b) override;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_CSR_HPP_
