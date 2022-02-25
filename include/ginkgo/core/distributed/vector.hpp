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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/cache.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace distributed {

namespace detail {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void read_distributed_impl(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<LocalIndexType, GlobalIndexType>* partition,
    Vector<ValueType>* result);


}


/**
 * Vector is a format which explicitly stores (multiple) distributed column
 * vectors in a dense storage format.
 *
 * The (multi-)vector is distributed by row, which is described by a @see
 * Partition. The local vectors are stored using the @see Dense format. The
 * vector should be filled using the read_distributed method, e.g.
 * ```
 * auto part = Partition<...>::build_from_mapping(...);
 * auto vector = Vector<...>::create(exec, comm);
 * vector->read_distributed(matrix_data, part);
 * ```
 * Using this approach the size of the global vectors, as well as the size of
 * the local vectors, will be automatically inferred. It is possible to create a
 * vector with specified global and local sizes and fill the local vectors using
 * the accessor get_local.
 *
 * @note Operations between two vectors (axpy, dot product, etc.) are only valid
 * if both vectors where created using the same partition.
 *
 * @tparam ValueType  The precision of vector elements.
 *
 * @ingroup dist_vector
 * @ingroup distributed
 */
template <typename ValueType = double>
class Vector
    : public EnableLinOp<Vector<ValueType>>,
      public EnableCreateMethod<Vector<ValueType>>,
      public ConvertibleTo<Vector<next_precision<ValueType>>>,
      public EnableAbsoluteComputation<remove_complex<Vector<ValueType>>>,
      public DistributedBase {
    friend class EnableCreateMethod<Vector<ValueType>>;
    friend class EnablePolymorphicObject<Vector<ValueType>, LinOp>;
    friend class Vector<to_complex<ValueType>>;
    friend class Vector<next_precision<ValueType>>;

public:
    using EnableLinOp<Vector>::convert_to;
    using EnableLinOp<Vector>::move_to;

    using value_type = ValueType;
    using absolute_type = remove_complex<Vector>;
    using real_type = absolute_type;
    using complex_type = Vector<to_complex<value_type>>;
    using local_vector_type = gko::matrix::Dense<value_type>;

    /**
     * Reads a vector from the device_matrix_data structure and a global row
     * partition.
     *
     * The number of rows of the matrix data is ignored, only its number of
     * columns is relevant. The number of rows is inferred from the partition.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *        by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure
     * @param partition  The global row partition
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    void read_distributed(
        const device_matrix_data<ValueType, GlobalIndexType>& data,
        const Partition<LocalIndexType, GlobalIndexType>* partition)
    {
        auto global_cols = data.get_size()[1];
        this->resize(
            dim<2>(partition->get_size(), global_cols),
            dim<2>(partition->get_part_size(this->get_communicator().rank()),
                   global_cols));
        detail::read_distributed_impl(data, partition, this);
    }

    /**
     * Reads a vector from the matrix_data structure and a global row
     * partition.
     *
     * See @read_distributed
     *
     * @note For efficiency it is advised to use the device_matrix_data
     * overload.
     */
    template <typename LocalIndexType, typename GlobalIndexType>
    void read_distributed(
        const matrix_data<ValueType, GlobalIndexType>& data,
        const Partition<LocalIndexType, GlobalIndexType>* partition)
    {
        this->read_distributed(
            device_matrix_data<value_type, GlobalIndexType>::create_from_host(
                this->get_executor(), data),
            std::move(partition));
    }


    void convert_to(Vector<next_precision<ValueType>>* result) const override;

    void move_to(Vector<next_precision<ValueType>>* result) override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Creates a complex copy of the original vectors. If the original vectors
     * were real, the imaginary part of the result will be zero.
     */
    std::unique_ptr<complex_type> make_complex() const;

    /**
     * Writes a complex copy of the original vectors to given complex vectors.
     * If the original vectors were real, the imaginary part of the result will
     * be zero.
     */
    void make_complex(complex_type* result) const;

    /**
     * Creates new real vectors and extracts the real part of the original
     * vectors into that.
     */
    std::unique_ptr<real_type> get_real() const;

    /**
     * Extracts the real part of the original vectors into given real vectors.
     */
    void get_real(real_type* result) const;

    /**
     * Creates new real vectors and extracts the imaginary part of the
     * original vectors into that.
     */
    std::unique_ptr<real_type> get_imag() const;

    /**
     * Extracts the imaginary part of the original vectors into given real
     * vectors.
     */
    void get_imag(real_type* result) const;

    /**
     * Fill the distributed vectors with a given value.
     *
     * @param value  the value to be filled
     */
    void fill(ValueType value);

    /**
     * Scales the vectors with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 Dense matrx, the all vectors are scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column vector is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of vectors).
     */
    void scale(const LinOp* alpha);

    /**
     * Scales the vectors with the inverse of a scalar.
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors are scaled
     *               by 1 / alpha. If it is a Dense row vector of values,
     *               then i-th column vector is scaled with the inverse
     *               of the i-th element of alpha (the number of columns of
     *               alpha has to match the number of vectors).
     */
    void inv_scale(const LinOp* alpha);

    /**
     * Adds `b` scaled by `alpha` to the vectors (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors of b are
     * scaled by alpha. If it is a Dense row vector of values, then i-th column
     * vector of b is scaled with the i-th element of alpha (the number of
     * columns of alpha has to match the number of vectors).
     * @param b  a (multi-)vector of the same dimension as this
     */
    void add_scaled(const LinOp* alpha, const LinOp* b);

    /**
     * Subtracts `b` scaled by `alpha` from the vectors (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors of b are
     * scaled by alpha. If it is a Dense row vector of values, then i-th column
     * vector of b is scaled with the i-th element of alpha (the number of c
     * @param b  a (multi-)vector of the same dimension as this
     */
    void sub_scaled(const LinOp* alpha, const LinOp* b);

    /**
     * Computes the column-wise dot product of this (multi-)vector and `b` using
     * a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     */
    void compute_dot(const LinOp* b, LinOp* result) const;

    /**
     * Computes the column-wise dot product of this (multi-)vector and `conj(b)`
     * using a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     */
    void compute_conj_dot(const LinOp* b, LinOp* result) const;

    /**
     * Computes the Euclidian (L^2) norm of this (multi-)vector using a global
     * reduction.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     */
    void compute_norm2(LinOp* result) const;

    /**
     * Computes the column-wise (L^1) norm of this (multi-)vector.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     */
    void compute_norm1(LinOp* result) const;

    /**
     * Direct (read) access to the underlying local local_vector_type vectors.
     *
     * @return a constant pointer to the underlying local_vector_type vectors
     */
    const local_vector_type* get_const_local() const;

    /*
     * Direct (read/write) access to the underlying local_vector_type Dense
     * vectors.
     *
     * @return a constant pointer to the underlying local_vector_type vectors
     */
    local_vector_type* get_local();

    /**
     * Create a real view of the (potentially) complex original multi-vector.
     * If the original vector is real, nothing changes. If the original vector
     * is complex, the result is created by viewing the complex vector with as
     * real with a reinterpret_cast with twice the number of columns and
     * double the stride.
     */
    std::unique_ptr<real_type> create_real_view();

    /**
     * @copydoc create_real_view()
     */
    std::unique_ptr<const real_type> create_real_view() const;

protected:
    /**
     * Creates an empty distributed vector with a specified size
     *
     * @param exec  Executor associated with vector
     * @param comm  Communicator associated with vector, the default is
     *              MPI_COMM_WORLD
     * @param partition  Partition of global rows
     * @param global_size  Global size of the vector
     * @param local_size  Processor-local size of the vector
     * @param stride  Stride of the local vector.
     */
    Vector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
           dim<2> global_size, dim<2> local_size, size_type stride);

    /**
     * Creates an empty distributed vector with a specified size
     *
     * @param exec  Executor associated with vector
     * @param comm  Communicator associated with vector, the default is
     *              MPI_COMM_WORLD
     * @param partition  Partition of global rows
     * @param global_size  Global size of the vector
     * @param local_size  Processor-local size of the vector, uses local_size[1]
     *                    as the stride
     */
    explicit Vector(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm = mpi::communicator(MPI_COMM_WORLD),
                    dim<2> global_size = {}, dim<2> local_size = {});

    /**
     * Creates a distributed vector from local vectors with a specified size.
     *
     * @param exec  Executor associated with this vector
     * @param comm  Communicator associated with this vector
     * @param global_size  The global size of the vector
     * @param local_vector  The underlying local vector, the date will be moved
     *                      into this
     */
    Vector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
           dim<2> global_size, local_vector_type* local_vector);

    /**
     * Creates a distributed vector from local vectors. The global size will
     * be deduced from the local sizes, which will incur a collective
     * communication.
     *
     * @param exec  Executor associated with this vector
     * @param comm  Communicator associated with this vector
     * @param local_vector  The underlying local vector, the date will be moved
     *                      into this
     */
    Vector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
           local_vector_type* local_vector);

    void resize(dim<2> global_size, dim<2> local_size);

    void apply_impl(const LinOp*, LinOp*) const override;

    void apply_impl(const LinOp*, const LinOp*, const LinOp*,
                    LinOp*) const override;

private:
    local_vector_type local_;
    ::gko::detail::DenseCache<ValueType> host_reduction_buffer_;
    ::gko::detail::DenseCache<remove_complex<ValueType>> host_norm_buffer_;
};


}  // namespace distributed
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_
