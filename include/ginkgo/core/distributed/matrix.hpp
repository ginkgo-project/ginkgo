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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;


}

namespace detail {


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


template <typename T, typename = void>
struct is_matrix_builder : std::false_type {};

template <typename T>
struct is_matrix_builder<
    T, xstd::void_t<decltype(std::declval<T>().template create<double, int>(
           std::declval<std::shared_ptr<const Executor>>()))>>
    : std::true_type {};


}  // namespace detail


/**
 * This function returns a type that delays a call to MatrixType::create.
 *
 * It can be used to set the used value and index type, as well as the executor
 * at a later stage.
 *
 * For example, the following code creates first a temporary object, which is
 * then used later to construct a operator of the previously defined base type:
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
 *       gko::distributed::Matrix more easily.
 *
 * @tparam MatrixType  A template type that accepts two types, the first one
 *                     will be set to the value type, the second one to the
 *                     index type.
 * @tparam Args  Types of the arguments passed to MatrixType::create.
 * @param create_args  arguments that will be forwarded to MatrixType::create
 * @return  A type with a function `create<value_type, index_type>(executor)`.
 */
template <template <typename, typename> class MatrixType, typename... Args>
auto with_matrix_type(Args&&... create_args)
{
    return detail::MatrixTypeBuilderFromValueAndIndex<MatrixType, Args...>{
        std::make_tuple(create_args...)};
}


namespace distributed {


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
 * One matrix, called `local_inner`, contains only entries from columns that are
 * also owned by the process, while the other one, called `local_ghost`,
 * contains entries from columns that are not owned by the process. The ghost
 * matrix is stored in a compressed format, where empty columns are discarded
 * and the remaining columns are renumbered. This splitting is depicted in the
 * following:
 * ```
 * Part-Id  Global                            Inner      Ghost
 * 0        | .. 1  ⁞ 2  .. ⁞ .. .. |         | .. 1  |  | 2  |
 * 0        | 3  4  ⁞ .. .. ⁞ .. .. |         | 3  4  |  | .. |
 *          |-----------------------|
 * 1        | .. 5  ⁞ 6  .. ⁞ 7  .. |  ---->  | 6  .. |  | 5  7  .. |
 * 1        | .. .. ⁞ .. 8  ⁞ ..  9 |  ---->  | 8  .. |  | .. .. 9  |
 *          |-----------------------|
 * 2        | .. .. ⁞ .. 10 ⁞ 11 12 |         | 11 12 |  | .. 10 |
 * 2        | 13 .. ⁞ .. .. ⁞ 14 .. |         | 14 .. |  | 13 .. |
 * ```
 * This uses the same ownership of the columns as for the rows.
 * Additionally, the ownership of the columns may be explicitly defined with an
 * second column partition. If that is not provided, the same row partition will
 * be used for the columns. Using a column partition also allows to create
 * non-square matrices, like the one below:
 * ```
 * Part-Id  Global                   Inner      Ghost
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
 * The Matrix LinOp supports the following operations:
 * ```cpp
 * distributed::Matrix *A;       // distributed matrix
 * distributed::Vector *b, *x;   // distributed multi-vectors
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
      public EnableCreateMethod<
          Matrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public ConvertibleTo<
          Matrix<next_precision<ValueType>, LocalIndexType, GlobalIndexType>>,
      public DistributedBase {
    friend class EnableCreateMethod<Matrix>;
    friend class EnablePolymorphicObject<Matrix, LinOp>;
    friend class Matrix<next_precision<ValueType>, LocalIndexType,
                        GlobalIndexType>;

public:
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using global_vector_type = gko::distributed::Vector<ValueType>;
    using local_vector_type = typename global_vector_type::local_vector_type;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;

    using EnableLinOp<Matrix>::convert_to;
    using EnableLinOp<Matrix>::move_to;

    void convert_to(Matrix<next_precision<value_type>, local_index_type,
                           global_index_type>* result) const override;

    void move_to(Matrix<next_precision<value_type>, local_index_type,
                        global_index_type>* result) override;

    /**
     * Reads a square matrix from the device_matrix_data structure and a global
     * partition.
     *
     * Both the number of rows and columns of the device_matrix_data is ignored.
     * The global size of the final matrix is inferred only from the size row
     * partition.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *        by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure.
     * @param partition  The global row and column partition.
     */
    void read_distributed(
        const device_matrix_data<value_type, global_index_type>& data,
        const Partition<local_index_type, global_index_type>* partition);

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
        const Partition<local_index_type, global_index_type>* partition);

    /**
     * Reads a matrix from the device_matrix_data structure, a global row
     * partition, and a global column partition.
     *
     * Both the number of rows and columns of the device_matrix_data is ignored.
     * The global size of the final matrix is inferred only from the size row
     * partition.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *        by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure.
     * @param row_partition  The global row partition.
     * @param col_partition  The global col partition.
     */
    void read_distributed(
        const device_matrix_data<value_type, global_index_type>& data,
        const Partition<local_index_type, global_index_type>* row_partition,
        const Partition<local_index_type, global_index_type>* col_partition);

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
        const Partition<local_index_type, global_index_type>* row_partition,
        const Partition<local_index_type, global_index_type>* col_partition);

    /**
     * Get read access to the local diagonal matrix
     * @return  Shared pointer to the local diagonal matrix
     */
    std::shared_ptr<const LinOp> get_const_local_diag() const
    {
        return diag_mtx_;
    }

    /**
     * Get read access to the local off-diagonal matrix
     * @return  Shared pointer to the local off-diagonal matrix
     */
    std::shared_ptr<const LinOp> get_const_local_offdiag() const
    {
        return offdiag_mtx_;
    }

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(const Matrix& other);

    Matrix& operator=(Matrix&& other) noexcept;

protected:
    /**
     * Creates an empty distributed matrix.
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     *              The default is the invalid MPI_COMM_NULL.
     */
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm = mpi::communicator(MPI_COMM_WORLD));

    /**
     * Creates an empty distributed matrix with specified type
     * for local matricies.
     *
     * @note This is mainly a convienience wrapper for
     *       Matrix(std::shared_ptr<const Executor>, mpi::communicator, const
     *       LinOp*)
     *
     * @tparam MatrixType  A type that has a `create<ValueType,
     *                     IndexType>(exec)` function to create an smart pointer
     *                     of a type derived from LinOp and
     *                     ReadableFromMatrixData
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param matrix_type  the local matrices will be constructed with the same
     *                     type as `create` returns. It should be the return
     *                     value of make_matrix_type.
     */
    template <typename MatrixType>
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, MatrixType matrix_type)
        : Matrix(
              exec, comm,
              static_cast<const LinOp*>(
                  matrix_type.template create<ValueType, LocalIndexType>(exec)
                      .get()))
    {}

    /**
     * Creates an empty distributed matrix with specified types for the local
     * inner matrix and the local ghost matrix.
     *
     * @note This is mainly a convienience wrapper for
     *       Matrix(std::shared_ptr<const Executor>, mpi::communicator,
     *       const LinOp*, const LinOp*)
     *
     * @tparam InnerMatrixType  A type that has a `create<ValueType,
     *                          IndexType>(exec)` function to create an smart
     *                          pointer of a type derived from LinOp and
     *                          ReadableFromMatrixData
     * @tparam GhostMatrixType  A (possible different) type with the same
     *                          constraints as InnerMatrixType
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param inner_matrix_type  the local inner matrix will be constructed with
     *                           the same type as `create` returns. It should be
     *                           the return value of make_matrix_type.
     * @param inner_matrix_type  the local ghost matrix will be constructed with
     *                           the same type as `create` returns. It should be
     *                           the return value of make_matrix_type.
     */
    template <typename InnerMatrixType, typename GhostMatrixType>
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, InnerMatrixType inner_matrix_type,
                    GhostMatrixType ghost_matrix_type)
        : Matrix(exec, comm,
                 static_cast<const LinOp*>(
                     inner_matrix_type
                         .template create<ValueType, LocalIndexType>(exec)
                         .get()),
                 static_cast<const LinOp*>(
                     ghost_matrix_type
                         .template create<ValueType, LocalIndexType>(exec)
                         .get()))
    {}

    /**
     * Creates an empty distributed matrix with specified type
     * for local matricies.
     *
     * @note It internally clones the passed in matrix_type. Therefore, this
     *       LinOp should be empty.
     *
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param matrix_type  the local matrices will be constructed with the same
     *                     runtime type.
     */
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, const LinOp* matrix_type);

    /**
     * Creates an empty distributed matrix with specified types for the local
     * inner matrix and the local ghost matrix.
     *
     * @note It internally clones the passed in inner_matrix_type and
     *       ghost_matrix_type. Therefore, these LinOps should be empty.
     *
     * @tparam InnerMatrixType  A type that has a `create<ValueType,
     *                          IndexType>(exec)` function to create an smart
     *                          pointer of a type derived from LinOp and
     *                          ReadableFromMatrixData
     * @tparam GhostMatrixType  A (possible different) type with the same
     *                          constraints as InnerMatrixType
     * @param exec  Executor associated with this matrix.
     * @param comm  Communicator associated with this matrix.
     * @param inner_matrix_type  the local inner matrix will be constructed with
     *                           the same runtime type.
     * @param inner_matrix_type  the local ghost matrix will be constructed with
     *                           the same runtime type.
     */
    explicit Matrix(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, const LinOp* inner_matrix_type,
                    const LinOp* ghost_matrix_type);

    /**
     * Starts a non-blocking communication of the values of b that are shared
     * with other processors.
     * @param local_b  The full local vector to be communicated. The subset of
     *                 shared values is automatically extracted.
     * @return  MPI request for the non-blocking communication.
     */
    mpi::request communicate(const local_vector_type* local_b) const;

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> recv_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    Array<local_index_type> gather_idxs_;
    Array<global_index_type> local_to_global_ghost_;
    ::gko::detail::DenseCache<value_type> one_scalar_;
    ::gko::detail::DenseCache<value_type> host_send_buffer_;
    ::gko::detail::DenseCache<value_type> host_recv_buffer_;
    ::gko::detail::DenseCache<value_type> send_buffer_;
    ::gko::detail::DenseCache<value_type> recv_buffer_;
    std::shared_ptr<LinOp> diag_mtx_;
    std::shared_ptr<LinOp> offdiag_mtx_;
};


}  // namespace distributed
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
