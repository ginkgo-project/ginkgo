// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_DD_MATRIX_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_DD_MATRIX_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;


}


namespace detail {


/**
 * Helper struct to test if the Builder type has a function create<ValueType,
 * IndexType>(std::shared_ptr<const Executor>).
 */
template <typename Builder, typename ValueType, typename IndexType,
          typename FourthType>
struct is_matrix_type_builder;


template <template <typename, typename> class MatrixType,
          typename... CreateArgs>
struct MatrixTypeBuilderFromValueAndIndex;


}  // namespace detail


namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
class Partition;
template <typename ValueType>
class Vector;


/**
 * The DdMatrix class defines a (MPI-)distributed matrix.
 *
 * The matrix is stored in an unassembled distributed format as it occurs in
 * Domain Decomposition applications.
 * Each process owns a locally assembled matrix, and the local contributions
 * are coupled on subdomain interfaces via a restriction and a prolongation
 * operator, which are distributed matrices
 * (gko::experimental::distributed::Matrix) defined through the global indices
 * appearing in the local contributions as well as a partition.
 * The partition considers the partitioning of vectors x to which the
 * matrix is applied and the resulting vector y = A*x. The following example
 * attempts to give an overview over how this would look for a 3x3 matrix
 * distributed to two ranks.
 * ```
 * Local Contribution on       Globally Assembled Matrix A
 * Rank 0        Rank 1
 * |  4 -2  0 |  |  0  0  0 |  |  4 -2  0 |
 * | -2  2  0 |  |  0  2 -2 |  | -2  4 -2 |
 * |  0  0  0 |  |  0 -2  4 |  |  0 -2  4 |
 * ```
 * With a partition where rank 0 owns the first two rows and rank 1 the
 * third, this would lead to a restriction operator R
 * ```
 * Part-Id  Global              Local    Non-Local
 * 0        | 1 0 ! 0 |         | 1 0 |  | |
 * 0        | 0 1 ! 0 |         | 0 1 |  | |
 *          |---------|  ---->
 * 1        | 0 1 ! 0 |         | 0 |    | 1 |
 * 1        | 0 0 ! 1 |         | 1 |    | 0 |
 * ```
 * and a prolongation operator R^T
 * ```
 * Part-Id  Global                Local    Non-Local
 * 0        | 1 0 ! 0 0 |         | 1 0 |  | 0 |
 * 0        | 0 1 ! 1 0 |         | 0 1 |  | 1 |
 *          |-----------|  ---->
 * 1        | 0 0 ! 0 1 |         | 0 1 |  | |
 * ```
 * With these operators and a block diagonal 4x4 matrix A_BD
 * ```
 * |  4 -2  0  0 |
 * | -2  2  0  0 |
 * |  0  0  2 -2 |
 * |  0  0 -2  4 |
 * ```
 * we can now write A = R^T A_BD R.
 *
 * The Matrix should be filled using the read_distributed method, e.g.
 * ```
 * auto part = Partition<...>::build_from_mapping(...);
 * auto mat = Matrix<...>::create(exec, comm);
 * mat->read_distributed(matrix_data, part);
 * ```
 * This will set the dimensions of the global and local matrices and generate
 * the restriction and prolongation matrices automatically by deducing the sizes
 * from the partition.
 *
 * By default the Matrix type uses Csr for the local matrix and the storage of
 * the local and non-local parts of the restriction and prolongation matrices.
 * It is possible to explicitly change the datatype for the local matrices, with
 * the constraint that the new type should implement the LinOp and
 * ReadableFromMatrixData interface. The type can be set by:
 * ```
 * auto mat = Matrix<ValueType, LocalIndexType[, ...]>::create(
 *   exec, comm,
 *   Coo<ValueType, LocalIndexType>::create(exec).get());
 * ```
 * Alternatively, the helper function with_matrix_type can be used:
 * ```
 * auto mat = Matrix<ValueType, LocalIndexType>::create(
 *   exec, comm,
 *   with_matrix_type<Coo>());
 * ```
 * @see with_matrix_type
 *
 * The DdMatrix LinOp supports the following operations:
 * ```cpp
 * experimental::distributed::Matrix *A;       // distributed matrix
 * experimental::distributed::Vector *b, *x;   // distributed multi-vectors
 * matrix::Dense *alpha, *beta;  // scalars of dimension 1x1
 *
 * // Applying to distributed multi-vectors computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 * A->row_scale(b)             // A = A * diag(b)
 * A->col_scale(b)             // A = diag(b) * A
 * ```
 *
 * @tparam ValueType  The underlying value type.
 * @tparam LocalIndexType  The index type used by the local matrices.
 * @tparam GlobalIndexType  The type for global indices.
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class DdMatrix
    : public EnableLinOp<DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public ConvertibleTo<DdMatrix<next_precision_base<ValueType>,
                                    LocalIndexType, GlobalIndexType>>,
      public DistributedBase {
    friend class EnablePolymorphicObject<DdMatrix, LinOp>;
    friend class DdMatrix<next_precision_base<ValueType>, LocalIndexType,
                          GlobalIndexType>;

public:
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using global_matrix_type =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using global_vector_type =
        gko::experimental::distributed::Vector<ValueType>;
    using local_vector_type = typename global_vector_type::local_vector_type;

    using EnableLinOp<DdMatrix>::convert_to;
    using EnableLinOp<DdMatrix>::move_to;
    using ConvertibleTo<DdMatrix<next_precision_base<ValueType>, LocalIndexType,
                                 GlobalIndexType>>::convert_to;
    using ConvertibleTo<DdMatrix<next_precision_base<ValueType>, LocalIndexType,
                                 GlobalIndexType>>::move_to;

    void convert_to(DdMatrix<next_precision_base<value_type>, local_index_type,
                             global_index_type>* result) const override;

    void move_to(DdMatrix<next_precision_base<value_type>, local_index_type,
                          global_index_type>* result) override;

    /**
     * Reads a square matrix from the device_matrix_data structure and a global
     * partition.
     *
     * The global size of the final matrix is inferred from the size of the
     * partition. Both the number of rows and columns of the device_matrix_data
     * are ignored.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *        by the process. The local matrix still considers these and the
     *        restriction and prolongation operators take care of fetching /
     *        re-distributing the corresponding vector entries.
     *
     * @param data  The device_matrix_data structure.
     * @param partition  The global partition.
     */
    void read_distributed(
        const device_matrix_data<value_type, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            partition);

    /**
     * Reads a square matrix from the matrix_data structure and a global
     * partition.
     *
     * @see read_distributed
     *
     * @note For efficiency it is advised to use the device_matrix_data
     * overload.
     */
    void read_distributed(
        const matrix_data<value_type, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            partition);

    /**
     * Get read access to the stored local matrix.
     *
     * @return  Shared pointer to the stored local matrix
     */
    std::shared_ptr<const LinOp> get_local_matrix() const { return local_mtx_; }

    /**
     * Get read access to the stored restriction operator.
     *
     * @return  Shared pointer to the stored restriction operator.
     */
    std::shared_ptr<const global_matrix_type> get_restriction() const
    {
        return restriction_;
    }

    /**
     * Get read access to the stored prolongation operator.
     *
     * @return  Shared pointer to the stored prolongation operator.
     */
    std::shared_ptr<const global_matrix_type> get_prolongation() const
    {
        return prolongation_;
    }

    /**
     * Copy constructs a Matrix.
     *
     * @param other  Matrix to copy from.
     */
    DdMatrix(const DdMatrix& other);

    /**
     * Move constructs a Matrix.
     *
     * @param other  Matrix to move from.
     */
    DdMatrix(DdMatrix&& other) noexcept;

    /**
     * Copy assigns a Matrix.
     *
     * @param other  Matrix to copy from, has to have a communicator of the same
     *               size as this.
     *
     * @return  this.
     */
    DdMatrix& operator=(const DdMatrix& other);

    /**
     * Move assigns a Matrix.
     *
     * @param other  Matrix to move from, has to have a communicator of the same
     *               size as this.
     *
     * @return  this.
     */
    DdMatrix& operator=(DdMatrix&& other);

    /**
     * Creates an empty distributed domain decomposition matrix.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     *              The default is the MPI_COMM_WORLD.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<DdMatrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm);

    /**
     * Creates an empty distributed domain decomposition matrix with specified
     * type for local matrices.
     *
     * @note This is mainly a convenience wrapper for
     *       Matrix(std::shared_ptr<const Executor>, mpi::communicator, const
     *       LinOp*)
     *
     * @tparam MatrixType  A type that has a `create<ValueType,
     *                     IndexType>(exec)` function to create a smart pointer
     *                     of a type derived from LinOp and
     *                     ReadableFromMatrixData. @see with_matrix_type
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param matrix_template  the local matrices will be constructed with the
     *                         same type as `create` returns. It should be the
     *                         return value of make_matrix_template.
     *
     * @return A smart pointer to the newly created matrix.
     */
    template <typename MatrixType,
              typename = std::enable_if_t<gko::detail::is_matrix_type_builder<
                  MatrixType, ValueType, LocalIndexType, void>::value>>
    static std::unique_ptr<DdMatrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        MatrixType matrix_template)
    {
        return create(
            exec, comm,
            matrix_template.template create<ValueType, LocalIndexType>(exec));
    }

    /**
     * Creates an empty distributed domain decomposition matrix with specified
     * type for local matrices.
     *
     * @note It internally clones the passed in matrix_template. Therefore, the
     *       LinOp should be empty.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param matrix_template  the local matrices will be constructed with the
     *                         same runtime type.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<DdMatrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        ptr_param<const LinOp> matrix_template);

    /**
     * Scales the columns of the matrix by the respective entries of the vector.
     * The vector's row partition has to be the same as the matrix's left
     * partition. The scaling is done in-place.
     *
     * @param scaling_factors  The vector containing the scaling factors.
     */
    void col_scale(ptr_param<const global_vector_type> scaling_factors);

    /**
     * Scales the rows of the matrix by the respective entries of the vector.
     * The vector's row partition has to be the same as the matrix's right
     * partition. The scaling is done in-place.
     *
     * @param scaling_factors  The vector containing the scaling factors.
     */
    void row_scale(ptr_param<const global_vector_type> scaling_factors);

    const gko::experimental::distributed::index_map<LocalIndexType,
                                                    GlobalIndexType>
    get_map() const
    {
        return map_;
    }

protected:
    explicit DdMatrix(std::shared_ptr<const Executor> exec,
                      mpi::communicator comm);

    explicit DdMatrix(std::shared_ptr<const Executor> exec,
                      mpi::communicator comm,
                      ptr_param<const LinOp> matrix_template);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    gko::experimental::distributed::detail::VectorCache<value_type> lhs_buffer_;
    gko::experimental::distributed::detail::VectorCache<value_type> rhs_buffer_;
    std::shared_ptr<global_matrix_type> restriction_;
    std::shared_ptr<LinOp> local_mtx_;
    std::shared_ptr<global_matrix_type> prolongation_;
    gko::experimental::distributed::index_map<LocalIndexType, GlobalIndexType>
        map_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_DD_MATRIX_HPP_
