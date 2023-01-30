/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
class Partition;


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
 * the accessor get_local_vector.
 *
 * @note Operations between two vectors (axpy, dot product, etc.) are only valid
 * if both vectors where created using the same partition.
 *
 * @tparam ValueType  The precision of vector elements.
 *
 * @ingroup dist_vector
 * @ingroup distributed
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = double>
class Vector
    : public EnableDistributedLinOp<Vector<ValueType>>,
      public EnableCreateMethod<Vector<ValueType>>,
      public ConvertibleTo<Vector<next_precision<ValueType>>>,
      public EnableAbsoluteComputation<remove_complex<Vector<ValueType>>>,
      public DistributedBase {
    friend class EnableCreateMethod<Vector>;
    friend class EnableDistributedPolymorphicObject<Vector, LinOp>;
    friend class Vector<to_complex<ValueType>>;
    friend class Vector<remove_complex<ValueType>>;
    friend class Vector<next_precision<ValueType>>;

public:
    using EnableDistributedLinOp<Vector>::convert_to;
    using EnableDistributedLinOp<Vector>::move_to;
    using ConvertibleTo<Vector<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<Vector<next_precision<ValueType>>>::move_to;

    using value_type = ValueType;
    using absolute_type = remove_complex<Vector>;
    using real_type = absolute_type;
    using complex_type = Vector<to_complex<value_type>>;
    using local_vector_type = gko::matrix::Dense<value_type>;

    /**
     * Creates a distributed Vector with the same size and stride as another
     * Vector.
     *
     * @param other  The other vector whose configuration needs to copied.
     */
    static std::unique_ptr<Vector> create_with_config_of(
        pointer_param<const Vector> other);


    /**
     * Creates an empty Vector with the same type as another Vector, but on a
     * different executor.
     *
     * @param other  The other multi-vector whose type we target.
     * @param exec  The executor of the new multi-vector.
     *
     * @note  The new multi-vector uses the same communicator as other.
     *
     * @returns an empty Vector with the type of other.
     */
    static std::unique_ptr<Vector> create_with_type_of(
        pointer_param<const Vector> other,
        std::shared_ptr<const Executor> exec);

    /**
     * Creates an Vector with the same type as another Vector, but on a
     * different executor and with a different size.
     *
     * @param other  The other multi-vector whose type we target.
     * @param exec  The executor of the new multi-vector.
     * @param global_size  The global size of the multi-vector.
     * @param local_size  The local size of the multi-vector.
     * @param stride  The stride of the new multi-vector.
     *
     * @returns a Vector of specified size with the type of other.
     */
    static std::unique_ptr<Vector> create_with_type_of(
        pointer_param<const Vector> other, std::shared_ptr<const Executor> exec,
        const dim<2>& global_size, const dim<2>& local_size, size_type stride);

    /**
     * Reads a vector from the device_matrix_data structure and a global row
     * partition.
     *
     * The number of rows of the matrix data is ignored, only its number of
     * columns is relevant. Both the number of local and global rows are
     * inferred from the row partition.
     *
     * @note The matrix data can contain entries for rows other than those owned
     *        by the process. Entries for those rows are discarded.
     *
     * @param data  The device_matrix_data structure
     * @param partition  The global row partition
     */
    void read_distributed(
        const device_matrix_data<ValueType, int64>& data,
        pointer_param<const Partition<int64, int64>> partition);

    void read_distributed(
        const device_matrix_data<ValueType, int64>& data,
        pointer_param<const Partition<int32, int64>> partition);

    void read_distributed(
        const device_matrix_data<ValueType, int32>& data,
        pointer_param<const Partition<int32, int32>> partition);

    /**
     * Reads a vector from the matrix_data structure and a global row
     * partition.
     *
     * See @read_distributed
     *
     * @note For efficiency it is advised to use the device_matrix_data
     * overload.
     */
    void read_distributed(
        const matrix_data<ValueType, int64>& data,
        pointer_param<const Partition<int64, int64>> partition);

    void read_distributed(
        const matrix_data<ValueType, int64>& data,
        pointer_param<const Partition<int32, int64>> partition);

    void read_distributed(
        const matrix_data<ValueType, int32>& data,
        pointer_param<const Partition<int32, int32>> partition);

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
    void make_complex(pointer_param<complex_type> result) const;

    /**
     * Creates new real vectors and extracts the real part of the original
     * vectors into that.
     */
    std::unique_ptr<real_type> get_real() const;

    /**
     * Extracts the real part of the original vectors into given real vectors.
     */
    void get_real(pointer_param<real_type> result) const;

    /**
     * Creates new real vectors and extracts the imaginary part of the
     * original vectors into that.
     */
    std::unique_ptr<real_type> get_imag() const;

    /**
     * Extracts the imaginary part of the original vectors into given real
     * vectors.
     */
    void get_imag(pointer_param<real_type> result) const;

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
    void scale(pointer_param<const LinOp> alpha);

    /**
     * Scales the vectors with the inverse of a scalar.
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors are scaled
     *               by 1 / alpha. If it is a Dense row vector of values,
     *               then i-th column vector is scaled with the inverse
     *               of the i-th element of alpha (the number of columns of
     *               alpha has to match the number of vectors).
     */
    void inv_scale(pointer_param<const LinOp> alpha);

    /**
     * Adds `b` scaled by `alpha` to the vectors (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors of b are
     * scaled by alpha. If it is a Dense row vector of values, then i-th column
     * vector of b is scaled with the i-th element of alpha (the number of
     * columns of alpha has to match the number of vectors).
     * @param b  a (multi-)vector of the same dimension as this
     */
    void add_scaled(pointer_param<const LinOp> alpha,
                    pointer_param<const LinOp> b);

    /**
     * Subtracts `b` scaled by `alpha` from the vectors (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the all vectors of b are
     * scaled by alpha. If it is a Dense row vector of values, then i-th column
     * vector of b is scaled with the i-th element of alpha (the number of c
     * @param b  a (multi-)vector of the same dimension as this
     */
    void sub_scaled(pointer_param<const LinOp> alpha,
                    pointer_param<const LinOp> b);

    /**
     * Computes the column-wise dot product of this (multi-)vector and `b` using
     * a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     */
    void compute_dot(pointer_param<const LinOp> b,
                     pointer_param<LinOp> result) const;

    /**
     * Computes the column-wise dot product of this (multi-)vector and `b` using
     * a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_dot(pointer_param<const LinOp> b, pointer_param<LinOp> result,
                     array<char>& tmp) const;

    /**
     * Computes the column-wise dot product of this (multi-)vector and `conj(b)`
     * using a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     */
    void compute_conj_dot(pointer_param<const LinOp> b,
                          pointer_param<LinOp> result) const;

    /**
     * Computes the column-wise dot product of this (multi-)vector and `conj(b)`
     * using a global reduction.
     *
     * @param b  a (multi-)vector of same dimension as this
     * @param result  a Dense row matrix, used to store the dot product
     *                (the number of column in result must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_conj_dot(pointer_param<const LinOp> b,
                          pointer_param<LinOp> result, array<char>& tmp) const;

    /**
     * Computes the Euclidian (L^2) norm of this (multi-)vector using a global
     * reduction.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     */
    void compute_norm2(pointer_param<LinOp> result) const;

    /**
     * Computes the Euclidian (L^2) norm of this (multi-)vector using a global
     * reduction.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_norm2(pointer_param<LinOp> result, array<char>& tmp) const;

    /**
     * Computes the column-wise (L^1) norm of this (multi-)vector.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     */
    void compute_norm1(pointer_param<LinOp> result) const;

    /**
     * Computes the column-wise (L^1) norm of this (multi-)vector using a global
     * reduction.
     *
     * @param result  a Dense row matrix, used to store the norm
     *                (the number of columns in result must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_norm1(pointer_param<LinOp> result, array<char>& tmp) const;

    /**
     * Returns a single element of the multi-vector.
     *
     * @param row  the local row of the requested element
     * @param col  the local column of the requested element
     *
     * @note  the method has to be called on the same Executor the multi-vector
     * is stored at (e.g. trying to call this method on a GPU multi-vector from
     *        the OMP results in a runtime error)
     */
    value_type& at_local(size_type row, size_type col) noexcept;

    /**
     * @copydoc Vector::at(size_type, size_type)
     */
    value_type at_local(size_type row, size_type col) const noexcept;

    /**
     * Returns a single element of the multi-vector.
     *
     * Useful for iterating across all elements of the multi-vector.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param idx  a linear index of the requested element
     *             (ignoring the stride)
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    ValueType& at_local(size_type idx) noexcept;

    /**
     * @copydoc Vector::at(size_type)
     */
    ValueType at_local(size_type idx) const noexcept;

    /**
     * Returns a pointer to the array of local values of the multi-vector.
     *
     * @return the pointer to the array of local values
     */
    value_type* get_local_values();

    /**
     * @copydoc get_local_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_local_values() const;

    /**
     * Direct (read) access to the underlying local local_vector_type vectors.
     *
     * @return a constant pointer to the underlying local_vector_type vectors
     */
    const local_vector_type* get_local_vector() const;

    /**
     * Create a real view of the (potentially) complex original multi-vector.
     * If the original vector is real, nothing changes. If the original vector
     * is complex, the result is created by viewing the complex vector with as
     * real with a reinterpret_cast with twice the number of columns and
     * double the stride.
     */
    std::unique_ptr<const real_type> create_real_view() const;

    /**
     * @copydoc create_real_view
     */
    std::unique_ptr<real_type> create_real_view();

    size_type get_stride() const noexcept { return local_.get_stride(); }

protected:
    /**
     * Creates an empty distributed vector with a specified size
     *
     * @param exec  Executor associated with vector
     * @param comm  Communicator associated with vector
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
     * @param comm  Communicator associated with vector
     * @param global_size  Global size of the vector
     * @param local_size  Processor-local size of the vector, uses local_size[1]
     *                    as the stride
     */
    explicit Vector(std::shared_ptr<const Executor> exec,
                    mpi::communicator comm, dim<2> global_size = {},
                    dim<2> local_size = {});

    /**
     * Creates a distributed vector from local vectors with a specified size.
     *
     * @note  The data form the local_vector will be moved into the new
     *        distributed vector. This means, access to local_vector
     *        will be invalid after this call.
     *
     * @param exec  Executor associated with this vector
     * @param comm  Communicator associated with this vector
     * @param global_size  The global size of the vector
     * @param local_vector  The underlying local vector, the data will be moved
     *                      into this
     */
    Vector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
           dim<2> global_size, pointer_param<local_vector_type> local_vector);

    /**
     * Creates a distributed vector from local vectors. The global size will
     * be deduced from the local sizes, which will incur a collective
     * communication.
     *
     * @note  The data form the local_vector will be moved into the new
     *        distributed vector. This means, access to local_vector
     *        will be invalid after this call.
     *
     * @param exec  Executor associated with this vector
     * @param comm  Communicator associated with this vector
     * @param local_vector  The underlying local vector, the data will be moved
     *                      into this
     */
    Vector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
           pointer_param<local_vector_type> local_vector);

    void resize(dim<2> global_size, dim<2> local_size);

    template <typename LocalIndexType, typename GlobalIndexType>
    void read_distributed_impl(
        const device_matrix_data<ValueType, GlobalIndexType>& data,
        const Partition<LocalIndexType, GlobalIndexType>* partition);

    void apply_impl(const LinOp*, LinOp*) const override;

    void apply_impl(const LinOp*, const LinOp*, const LinOp*,
                    LinOp*) const override;

    /**
     * Creates a distributed vector with the same size and stride as the callers
     * vector.
     *
     * @returns a Vector with the same size and stride as the caller.
     */
    virtual std::unique_ptr<Vector> create_with_same_config() const;

    /**
     * Creates a Vector with the same type as the callers multi-vector.
     *
     * @note The new vector will use the same communicator as the caller.
     *
     * @param exec  the executor of the new vector.
     * @param global_size  global_size of the vector.
     * @param local_size  the size of the local Dense vector.
     * @param stride  the stride of the local Dense vector.
     *
     * @returns a Vector with the same type as the caller.
     */
    virtual std::unique_ptr<Vector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const;

private:
    local_vector_type local_;
    ::gko::detail::DenseCache<ValueType> host_reduction_buffer_;
    ::gko::detail::DenseCache<remove_complex<ValueType>> host_norm_buffer_;
};


}  // namespace distributed
}  // namespace experimental


namespace detail {


template <typename TargetType>
struct conversion_target_helper;


/**
 * @internal
 *
 * Specialization of conversion_target_helper for distributed vectors.
 * This is necessary, since Vector needs to be created from both an executor and
 * a communicator.
 *
 * @see conversion_target_helper
 */
template <typename ValueType>
struct conversion_target_helper<experimental::distributed::Vector<ValueType>> {
    using target_type = experimental::distributed::Vector<ValueType>;
    using source_type =
        experimental::distributed::Vector<previous_precision<ValueType>>;

    static std::unique_ptr<target_type> create_empty(const source_type* source)
    {
        return target_type::create(source->get_executor(),
                                   source->get_communicator());
    }
};


}  // namespace detail
}  // namespace gko


#endif  // GINKGO_BUILD_MPI


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_
