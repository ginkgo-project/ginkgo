// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;


}


namespace multigrid {


template <typename ValueType, typename IndexType>
class Pgm;


}


namespace detail {


/**
 * Helper struct to test if the Builder type has a function create<ValueType,
 * IndexType>(std::shared_ptr<const Executor>).
 */
template <typename Builder, typename ValueType, typename IndexType,
          typename = void>
struct is_matrix_type_builder : std::false_type {};


template <typename Builder, typename ValueType, typename IndexType>
struct is_matrix_type_builder<
    Builder, ValueType, IndexType,
    xstd::void_t<
        decltype(std::declval<Builder>().template create<ValueType, IndexType>(
            std::declval<std::shared_ptr<const Executor>>()))>>
    : std::true_type {};


template <template <typename, typename> class MatrixType,
          typename... CreateArgs>
struct MatrixTypeBuilderFromValueAndIndex {
    template <typename ValueType, typename IndexType, std::size_t... I>
    auto create_impl(std::shared_ptr<const Executor> exec,
                     std::index_sequence<I...>)
    {
        return MatrixType<ValueType, IndexType>::create(
            exec, std::get<I>(create_args)...);
    }


    template <typename ValueType, typename IndexType>
    auto create(std::shared_ptr<const Executor> exec)
    {
        // with c++17 we could use std::apply
        static constexpr auto size = sizeof...(CreateArgs);
        return create_impl<ValueType, IndexType>(
            std::move(exec), std::make_index_sequence<size>{});
    }

    std::tuple<CreateArgs...> create_args;
};


}  // namespace detail


/**
 * This function returns a type that delays a call to MatrixType::create.
 *
 * It can be used to set the used value and index type, as well as the executor
 * at a later stage.
 *
 * For example, the following code creates first a temporary object, which is
 * then used later to construct an operator of the previously defined base type:
 * ```
 * auto type = gko::with_matrix_type<gko::matrix::Csr>();
 * ...
 * std::unique_ptr<LinOp> concrete_op
 * if(flag1){
 *   concrete_op = type.template create<double, int>(exec);
 * } else {
 *   concrete_op = type.template create<float, int>(exec);
 * }
 * ```
 *
 * @note This is mainly a helper function to specify the local matrix type for a
 *       gko::experimental::distributed::Matrix more easily.
 *
 * @tparam MatrixType  A template type that accepts two types, the first one
 *                     will be set to the value type, the second one to the
 *                     index type.
 * @tparam Args  Types of the arguments passed to MatrixType::create.
 *
 * @param create_args  arguments that will be forwarded to MatrixType::create
 *
 * @return  A type with a function `create<value_type, index_type>(executor)`.
 */
template <template <typename, typename> class MatrixType, typename... Args>
auto with_matrix_type(Args&&... create_args)
{
    return detail::MatrixTypeBuilderFromValueAndIndex<MatrixType, Args...>{
        std::forward_as_tuple(create_args...)};
}


namespace experimental {
namespace distributed {


/**
 * assembly_mode defines how the read_distributed function of the distributed
 * matrix treats non-local indices in the (device_)matrix_data:
 * - `communicate` communicates the overlap between ranks and adds up all local
 *   contributions. Indices smaller than 0 or larger than the global size
 *   of the matrix are ignored.
 * - `local_only` does not communicate any overlap but ignores all non-local
 *   indices.
 */
enum class assembly_mode { communicate, local_only };


template <typename LocalIndexType, typename GlobalIndexType>
class Partition;
template <typename ValueType>
class Vector;


/**
 * The Matrix class defines a (MPI-)distributed matrix.
 *
 * The matrix is stored in a row-wise distributed format.
 * Each process owns a specific set of rows, where the assignment of rows is
 * defined by a row Partition. The following depicts the distribution of
 * global rows according to their assigned part-id (which will usually be the
 * owning process id):
 * ```
 * Part-Id  Global Rows                   Part-Id  Local Rows
 * 0        | .. 1  2  .. .. .. |         0        | .. 1  2  .. .. .. |
 * 1        | 3  4  .. .. .. .. |                  | 13 .. .. .. 14 .. |
 * 2        | .. 5  6  ..  7 .. |  ---->  1        | 3  4  .. .. .. .. |
 * 2        | .. .. .. 8  ..  9 |  ---->           | .. .. .. 10 11 12 |
 * 1        | .. .. .. 10 11 12 |         2        | .. 5  6  ..  7 .. |
 * 0        | 13 .. .. .. 14 .. |                  | .. .. .. 8  ..  9 |
 * ```
 * The local rows are further split into two matrices on each process.
 * One matrix, called `local`, contains only entries from columns that are
 * also owned by the process, while the other one, called `non_local`,
 * contains entries from columns that are not owned by the process. The
 * non-local matrix is stored in a compressed format, where empty columns are
 * discarded and the remaining columns are renumbered. This splitting is
 * depicted in the following:
 * ```
 * Part-Id  Global                            Local      Non-Local
 * 0        | .. 1  ! 2  .. ! .. .. |         | .. 1  |  | 2  |
 * 0        | 3  4  ! .. .. ! .. .. |         | 3  4  |  | .. |
 *          |-----------------------|
 * 1        | .. 5  ! 6  .. ! 7  .. |  ---->  | 6  .. |  | 5  7  .. |
 * 1        | .. .. ! .. 8  ! ..  9 |  ---->  | 8  .. |  | .. .. 9  |
 *          |-----------------------|
 * 2        | .. .. ! .. 10 ! 11 12 |         | 11 12 |  | .. 10 |
 * 2        | 13 .. ! .. .. ! 14 .. |         | 14 .. |  | 13 .. |
 * ```
 * This uses the same ownership of the columns as for the rows.
 * Additionally, the ownership of the columns may be explicitly defined with an
 * second column partition. If that is not provided, the same row partition will
 * be used for the columns. Using a column partition also allows to create
 * non-square matrices, like the one below:
 * ```
 * Part-Id  Global                  Local      Non-Local
 * P_R/P_C    2  2  0  1
 * 0        | .. 1  2  .. |         | 2  |     | 1  .. |
 * 0        | 3  4  .. .. |         | .. |     | 3  4  |
 *          |-------------|
 * 1        | .. 5  6  .. |  ---->  | .. |     | 6  5  |
 * 1        | .. .. .. 8  |  ---->  | 8  |     | .. .. |
 *          |-------------|
 * 2        | .. .. .. 10 |         | .. .. |  | 10 |
 * 2        | 13 .. .. .. |         | 13 .. |  | .. |
 * ```
 * Here `P_R` denotes the row partition and `P_C` denotes the column partition.
 *
 * The Matrix should be filled using the read_distributed method, e.g.
 * ```
 * auto part = Partition<...>::build_from_mapping(...);
 * auto mat = Matrix<...>::create(exec, comm);
 * mat->read_distributed(matrix_data, part);
 * ```
 * or if different partitions for the rows and columns are used:
 * ```
 * auto row_part = Partition<...>::build_from_mapping(...);
 * auto col_part = Partition<...>::build_from_mapping(...);
 * auto mat = Matrix<...>::create(exec, comm);
 * mat->read_distributed(matrix_data, row_part, col_part);
 * ```
 * This will set the dimensions of the global and local matrices automatically
 * by deducing the sizes from the partitions.
 *
 * By default the Matrix type uses Csr for both stored matrices. It is possible
 * to explicitly change the datatype for the stored matrices, with the
 * constraint that the new type should implement the LinOp and
 * ReadableFromMatrixData interface. The type can be set by:
 * ```
 * auto mat = Matrix<ValueType, LocalIndexType[, ...]>::create(
 *   exec, comm,
 *   Ell<ValueType, LocalIndexType>::create(exec).get(),
 *   Coo<ValueType, LocalIndexType>::create(exec).get());
 * ```
 * Alternatively, the helper function with_matrix_type can be used:
 * ```
 * auto mat = Matrix<ValueType, LocalIndexType>::create(
 *   exec, comm,
 *   with_matrix_type<Ell>(),
 *   with_matrix_type<Coo>());
 * ```
 * @see with_matrix_type
 *
 * The Matrix LinOp supports the following operations:
 * ```cpp
 * experimental::distributed::Matrix *A;       // distributed matrix
 * experimental::distributed::Vector *b, *x;   // distributed multi-vectors
 * matrix::Dense *alpha, *beta;  // scalars of dimension 1x1
 *
 * // Applying to distributed multi-vectors computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 * ```
 *
 * @tparam ValueType  The underlying value type.
 * @tparam LocalIndexType  The index type used by the local matrices.
 * @tparam GlobalIndexType  The type for global indices.
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class Matrix
    : public EnableLinOp<Matrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public ConvertibleTo<
          Matrix<next_precision<ValueType>, LocalIndexType, GlobalIndexType>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<Matrix<next_precision<ValueType, 2>, LocalIndexType,
                                  GlobalIndexType>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<Matrix<next_precision<ValueType, 3>, LocalIndexType,
                                  GlobalIndexType>>,
#endif
      public DistributedBase {
    friend class EnablePolymorphicObject<Matrix, LinOp>;
    friend class Matrix<previous_precision<ValueType>, LocalIndexType,
                        GlobalIndexType>;
    friend class multigrid::Pgm<ValueType, LocalIndexType>;


public:
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using global_vector_type =
        gko::experimental::distributed::Vector<ValueType>;
    using local_vector_type = typename global_vector_type::local_vector_type;

    using EnableLinOp<Matrix>::convert_to;
    using EnableLinOp<Matrix>::move_to;
    using ConvertibleTo<Matrix<next_precision<ValueType>, LocalIndexType,
                               GlobalIndexType>>::convert_to;
    using ConvertibleTo<Matrix<next_precision<ValueType>, LocalIndexType,
                               GlobalIndexType>>::move_to;

    void convert_to(Matrix<next_precision<value_type>, local_index_type,
                           global_index_type>* result) const override;

    void move_to(Matrix<next_precision<value_type>, local_index_type,
                        global_index_type>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class Matrix<previous_precision<ValueType, 2>, LocalIndexType,
                        GlobalIndexType>;
    using ConvertibleTo<Matrix<next_precision<value_type, 2>, local_index_type,
                               global_index_type>>::convert_to;
    using ConvertibleTo<Matrix<next_precision<value_type, 2>, local_index_type,
                               global_index_type>>::move_to;

    void convert_to(Matrix<next_precision<value_type, 2>, local_index_type,
                           global_index_type>* result) const override;

    void move_to(Matrix<next_precision<value_type, 2>, local_index_type,
                        global_index_type>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class Matrix<previous_precision<ValueType, 3>, LocalIndexType,
                        GlobalIndexType>;
    using ConvertibleTo<Matrix<next_precision<value_type, 3>, local_index_type,
                               global_index_type>>::convert_to;
    using ConvertibleTo<Matrix<next_precision<value_type, 3>, local_index_type,
                               global_index_type>>::move_to;

    void convert_to(Matrix<next_precision<value_type, 3>, local_index_type,
                           global_index_type>* result) const override;

    void move_to(Matrix<next_precision<value_type, 3>, local_index_type,
                        global_index_type>* result) override;
#endif
    /**
     * Reads a square matrix from the device_matrix_data structure and a global
     * partition.
     *
     * The global size of the final matrix is inferred from the size of the
     * partition. Both the number of rows and columns of the device_matrix_data
     * are ignored.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *       by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure.
     * @param partition  The global row and column partition.
     * @param x  The mode of assembly.
     *
     * @return the index_map induced by the partitions and the matrix structure
     */
    void read_distributed(
        const device_matrix_data<value_type, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            partition,
        assembly_mode assembly_type = assembly_mode::local_only);

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
            partition,
        assembly_mode assembly_type = assembly_mode::local_only);

    /**
     * Reads a matrix from the device_matrix_data structure, a global row
     * partition, and a global column partition.
     *
     * The global size of the final matrix is inferred from the size of the row
     * partition and the size of the column partition. Both the number of rows
     * and columns of the device_matrix_data are ignored.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *       by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure.
     * @param row_partition  The global row partition.
     * @param col_partition  The global col partition.
     * @param assembly_type  The mode of assembly.
     *
     * @return the index_map induced by the partitions and the matrix structure
     */
    void read_distributed(
        const device_matrix_data<value_type, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            row_partition,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            col_partition,
        assembly_mode assembly_type = assembly_mode::local_only);

    /**
     * Reads a matrix from the matrix_data structure, a global row partition,
     * and a global column partition.
     *
     * @see read_distributed
     *
     * @note For efficiency it is advised to use the device_matrix_data
     * overload.
     */
    void read_distributed(
        const matrix_data<value_type, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            row_partition,
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            col_partition,
        assembly_mode assembly_type = assembly_mode::local_only);

    /**
     * Get read access to the stored local matrix.
     *
     * @return  Shared pointer to the stored local matrix
     */
    std::shared_ptr<const LinOp> get_local_matrix() const { return local_mtx_; }

    /**
     * Get read access to the stored non-local matrix.
     *
     * @return  Shared pointer to the stored non-local matrix
     */
    std::shared_ptr<const LinOp> get_non_local_matrix() const
    {
        return non_local_mtx_;
    }

    /**
     * Copy constructs a Matrix.
     *
     * @param other  Matrix to copy from.
     */
    Matrix(const Matrix& other);

    /**
     * Move constructs a Matrix.
     *
     * @param other  Matrix to move from.
     */
    Matrix(Matrix&& other) noexcept;

    /**
     * Copy assigns a Matrix.
     *
     * @param other  Matrix to copy from, has to have a communicator of the same
     *               size as this.
     *
     * @return  this.
     */
    Matrix& operator=(const Matrix& other);

    /**
     * Move assigns a Matrix.
     *
     * @param other  Matrix to move from, has to have a communicator of the same
     *               size as this.
     *
     * @return  this.
     */
    Matrix& operator=(Matrix&& other);

    /**
     * Creates an empty distributed matrix.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     *              The default is the MPI_COMM_WORLD.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Matrix> create(std::shared_ptr<const Executor> exec,
                                          mpi::communicator comm);

    /**
     * Creates an empty distributed matrix with specified type
     * for local matrices.
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
                  MatrixType, ValueType, LocalIndexType>::value>>
    static std::unique_ptr<Matrix> create(std::shared_ptr<const Executor> exec,
                                          mpi::communicator comm,
                                          MatrixType matrix_template)
    {
        return create(
            exec, comm,
            matrix_template.template create<ValueType, LocalIndexType>(exec));
    }

    /**
     * Creates an empty distributed matrix with specified types for the local
     * matrix and the non-local matrix.
     *
     * @note This is mainly a convenience wrapper for
     *       Matrix(std::shared_ptr<const Executor>, mpi::communicator,
     *       const LinOp*, const LinOp*)
     *
     * @tparam LocalMatrixType  A type that has a `create<ValueType,
     *                          IndexType>(exec)` function to create a smart
     *                          pointer of a type derived from LinOp and
     *                          ReadableFromMatrixData. @see with_matrix_type
     * @tparam NonLocalMatrixType  A (possible different) type with the same
     *                             constraints as LocalMatrixType.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param local_matrix_template  the local matrix will be constructed
     *                               with the same type as `create` returns. It
     *                               should be the return value of
     *                               make_matrix_template.
     * @param non_local_matrix_template  the non-local matrix will be
     *                                   constructed with the same type as
     *                                   `create` returns. It should be the
     *                                   return value of make_matrix_template.
     *
     * @return A smart pointer to the newly created matrix.
     */
    template <typename LocalMatrixType, typename NonLocalMatrixType,
              typename = std::enable_if_t<
                  gko::detail::is_matrix_type_builder<
                      LocalMatrixType, ValueType, LocalIndexType>::value &&
                  gko::detail::is_matrix_type_builder<
                      NonLocalMatrixType, ValueType, LocalIndexType>::value>>
    static std::unique_ptr<Matrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        LocalMatrixType local_matrix_template,
        NonLocalMatrixType non_local_matrix_template)
    {
        return create(
            exec, comm,
            local_matrix_template.template create<ValueType, LocalIndexType>(
                exec),
            non_local_matrix_template
                .template create<ValueType, LocalIndexType>(exec));
    }

    /**
     * Creates an empty distributed matrix with specified type
     * for local matrices.
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
    static std::unique_ptr<Matrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        ptr_param<const LinOp> matrix_template);

    /**
     * Creates an empty distributed matrix with specified types for the local
     * matrix and the non-local matrix.
     *
     * @note It internally clones the passed in local_matrix_template and
     *       non_local_matrix_template. Therefore, those LinOps should be empty.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param local_matrix_template  the local matrix will be constructed
     *                               with the same runtime type.
     * @param non_local_matrix_template  the non-local matrix will be
     *                                   constructed with the same runtime type.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Matrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        ptr_param<const LinOp> local_matrix_template,
        ptr_param<const LinOp> non_local_matrix_template);

    /**
     * Creates a local-only distributed matrix with existent LinOp
     *
     * @note It use the input to build up the distributed matrix
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param size  the global size
     * @param local_linop  the local linop
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Matrix> create(std::shared_ptr<const Executor> exec,
                                          mpi::communicator comm, dim<2> size,
                                          std::shared_ptr<LinOp> local_linop);

    /**
     * Creates distributed matrix with existent local and non-local LinOp and
     * the corresponding mapping to collect the non-local data from the other
     * ranks.
     *
     * @note It use the input to build up the distributed matrix
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param size  the global size
     * @param local_linop  the local linop
     * @param non_local_linop  the non-local linop
     * @param recv_sizes  the size of non-local receiver
     * @param recv_offsets  the offset of non-local receiver
     * @param recv_gather_idxs  the gathering index of non-local receiver
     *
     * @return A smart pointer to the newly created matrix.
     */
    [[deprecated(
        "Please use the overload with an index_map instead.")]] static std::
        unique_ptr<Matrix>
        create(std::shared_ptr<const Executor> exec, mpi::communicator comm,
               dim<2> size, std::shared_ptr<LinOp> local_linop,
               std::shared_ptr<LinOp> non_local_linop,
               std::vector<comm_index_type> recv_sizes,
               std::vector<comm_index_type> recv_offsets,
               array<local_index_type> recv_gather_idxs);

    /**
     * Creates distributed matrix with existent local and non-local LinOp and
     * the corresponding mapping to collect the non-local data from the other
     * ranks.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param imap  The index map to define the communication pattern
     * @param local_linop  the local linop
     * @param non_local_linop  the non-local linop
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<Matrix> create(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        index_map<local_index_type, global_index_type> imap,
        std::shared_ptr<LinOp> local_linop,
        std::shared_ptr<LinOp> non_local_linop);

    /**
     * Scales the columns of the matrix by the respective entries of the vector.
     * The vector's row partition has to be the same as the matrix's column
     * partition. The scaling is done in-place.
     *
     * @param scaling_factors  The vector containing the scaling factors.
     */
    void col_scale(ptr_param<const global_vector_type> scaling_factors);

    /**
     * Scales the rows of the matrix by the respective entries of the vector.
     * The vector and the matrix have to have the same row partition.
     * The scaling is done in-place.
     *
     * @param scaling_factors  The vector containing the scaling factors.
     */
    void row_scale(ptr_param<const global_vector_type> scaling_factors);

protected:
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm);

    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm,
                    ptr_param<const LinOp> local_matrix_template,
                    ptr_param<const LinOp> non_local_matrix_template);

    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, dim<2> size,
                    std::shared_ptr<LinOp> local_linop);

    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm,
                    index_map<local_index_type, global_index_type> imap,
                    std::shared_ptr<LinOp> local_linop,
                    std::shared_ptr<LinOp> non_local_linop);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::shared_ptr<RowGatherer<LocalIndexType>> row_gatherer_;
    index_map<local_index_type, global_index_type> imap_;
    gko::detail::ScalarCache one_scalar_;
    detail::GenericVectorCache recv_buffer_;
    detail::GenericVectorCache host_recv_buffer_;
    std::shared_ptr<LinOp> local_mtx_;
    std::shared_ptr<LinOp> non_local_mtx_;
};


/* @brief create a distributed Matrix from a local matrix by scattering it
 * to ranks
 *
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_from_super_rank(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::shared_ptr<const matrix::Coo<ValueType, GlobalIndexType>> local_linop,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition)
{
    auto rank = comm.rank();

    // count number of nnzs and communicate with other ranks
    const ValueType* matrix_data_value_ptr = nullptr;
    const GlobalIndexType* matrix_col_idx_ptr = nullptr;
    const GlobalIndexType* matrix_row_ptr_ptr = nullptr;
    auto send_counts_nnz = std::vector<int>(comm.size(), 0);
    auto send_displ_nnz = std::vector<int>(comm.size() + 1, 0);
    auto recv_counts_nnz = std::vector<int>(comm.size(), 0);
    if (rank == 0) {
        auto num_elements = local_linop->get_num_stored_elements();
        auto mtx_data = matrix_data<ValueType, GlobalIndexType>();
        // TODO find a way not to use matrix_data
        local_linop->write(mtx_data);
        mtx_data.sort_row_major();
        auto dev_mtx_data =
            device_matrix_data<ValueType, GlobalIndexType>::create_from_host(
                exec, mtx_data);
        // TODO:
        // - add skip sorting flag
        // - copy to host
        matrix_data_value_ptr = dev_mtx_data.get_const_values();
        matrix_row_ptr_ptr = dev_mtx_data.get_const_row_idxs();
        matrix_col_idx_ptr = dev_mtx_data.get_const_col_idxs();

        for (auto i = 0; i < comm.size(); i++) {
            // TODO range bounds might be on device
            auto rank_start = partition->get_range_bounds()[i];
            auto rank_end = partition->get_range_bounds()[i + 1];
            send_counts_nnz[i] = rank_end - rank_start;
            send_displ_nnz[i + 1] = rank_end;
        }
    }

    int recv_counts = 0;
    comm.scatter(exec, send_counts_nnz.data(), 1, &recv_counts, 1, 0);
    recv_counts_nnz[0] = recv_counts;

    std::cout << __FILE__ << ":" << __LINE__ << " recv_rows " << recv_counts
              << "\n";

    gko::array<ValueType> repart_values(exec, recv_counts);
    comm.scatter_v(exec, matrix_data_value_ptr, send_counts_nnz.data(),
                   send_displ_nnz.data(), repart_values.get_data(), recv_counts,
                   0);

    gko::array<GlobalIndexType> repart_col_idxs(exec, recv_counts);
    comm.scatter_v(exec, matrix_col_idx_ptr, send_counts_nnz.data(),
                   send_displ_nnz.data(), repart_col_idxs.get_data(),
                   recv_counts, 0);

    gko::array<GlobalIndexType> repart_row_ptr(exec, recv_counts);
    comm.scatter_v(exec, matrix_row_ptr_ptr, send_counts_nnz.data(),
                   send_displ_nnz.data(), repart_row_ptr.get_data(),
                   recv_counts, 0);

    auto mtx =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(exec, comm);

    auto repart_mtx_data = device_matrix_data<ValueType, GlobalIndexType>(
        exec, gko::dim<2>{partition->get_size(), partition->get_size()}, repart_row_ptr,
        repart_col_idxs, repart_values);
    mtx->read_distributed(repart_mtx_data, partition);
    return mtx;
}

}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
