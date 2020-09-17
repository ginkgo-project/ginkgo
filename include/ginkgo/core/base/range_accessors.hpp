/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_RANGE_ACCESSORS_HPP_
#define GKO_CORE_BASE_RANGE_ACCESSORS_HPP_


#include <array>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include <iostream>
#define GKO_DEBUG_OUTPUT false


// CUDA TOOLKIT < 11 does not support constexpr in combination with
// thrust::complex, which is why constexpr is only present in later versions
#if defined(__CUDA_ARCH__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 11)

#define GKO_ENABLE_REFERENCE_CONSTEXPR

#else

#define GKO_ENABLE_REFERENCE_CONSTEXPR constexpr

#endif


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace accessor {


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @warning The current implementation is incomplete, and only allows for
 *          2-dimensional ranges.
 *
 * @tparam ValueType  type of values this accessor returns
 * @tparam Dimensionality  number of dimensions of this accessor (has to be
 * 2)
 */
template <typename ValueType, size_type Dimensionality>
class row_major {
public:
    friend class range<row_major>;

    static_assert(Dimensionality == 2,
                  "This accessor is only implemented for matrices");

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type *;

    /**
     * Number of dimensions of the accessor.
     */
    static constexpr size_type dimensionality = 2;

protected:
    /**
     * Creates a row_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param num_row  number of rows of the accessor
     * @param num_cols  number of columns of the accessor
     * @param stride  distance (in elements) between starting positions of
     *                consecutive rows (i.e. `data + i * stride` points to
     * the `i`-th row)
     */
    GKO_ATTRIBUTES constexpr explicit row_major(data_type data,
                                                size_type num_rows,
                                                size_type num_cols,
                                                size_type stride)
        : data{data}, lengths{num_rows, num_cols}, stride{stride}
    {}

public:
    /**
     * Returns the data element at position (row, col)
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (row, col)
     */
    GKO_ATTRIBUTES constexpr value_type &operator()(size_type row,
                                                    size_type col) const
    {
        return GKO_ASSERT(row < lengths[0]), GKO_ASSERT(col < lengths[1]),
               data[row * stride + col];
    }

    /**
     * Returns the sub-range spanning the range (rows, cols)
     *
     * @param rows  row span
     * @param cols  column span
     *
     * @return sub-range spanning the range (rows, cols)
     */
    GKO_ATTRIBUTES constexpr range<row_major> operator()(const span &rows,
                                                         const span &cols) const
    {
        return GKO_ASSERT(rows.is_valid()), GKO_ASSERT(cols.is_valid()),
               GKO_ASSERT(rows <= span{lengths[0]}),
               GKO_ASSERT(cols <= span{lengths[1]}),
               range<row_major>(data + rows.begin * stride + cols.begin,
                                rows.end - rows.begin, cols.end - cols.begin,
                                stride);
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return dimension < 2 ? lengths[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     * specific executor. It will always be performed sequentially.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        for (size_type i = 0; i < lengths[0]; ++i) {
            for (size_type j = 0; j < lengths[1]; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * An array of dimension sizes.
     */
    const std::array<const size_type, dimensionality> lengths;

    /**
     * Distance between consecutive rows.
     */
    const size_type stride;
};


namespace detail {


// tests if the cast operator to `ValueType` is present
template <typename Ref, typename ValueType, typename = xstd::void_t<>>
struct has_cast_operator : std::false_type {
};

template <typename Ref, typename ValueType>
struct has_cast_operator<
    Ref, ValueType,
    xstd::void_t<decltype(std::declval<Ref>().Ref::operator ValueType())>>
    : std::true_type {
};

/**
 * @internal
 * converts `ref` to ValueType while preferring the cast operator overload
 * from class `Ref` before falling back to a simple
 * `static_cast<ValueType>`.
 *
 * This function is only needed for CUDA TOOLKIT < 11 because
 * thrust::complex has a constructor call: `template<T> complex(const T
 * &other) : real(other), imag()`, which is always preferred over the
 * overloaded `operator value_type()`.
 */
template <typename ValueType, typename Ref>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<
    has_cast_operator<Ref, ValueType>::value, ValueType>
to_value_type(const Ref &ref)
{
    return ref.Ref::operator ValueType();
}

template <typename ValueType, typename Ref>
GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<
    !has_cast_operator<Ref, ValueType>::value, ValueType>
to_value_type(const Ref &ref)
{
    // TODO  remove
    static_assert(std::is_same<ValueType, void>::value,
                  "This should not be used here!");
    return static_cast<ValueType>(ref);
}

/**
 * This is a mixin which defines the binary operators for *, /, +, - for the
 * Reference class, the unary operator -, and the assignment operators
 * *=, /=, +=, -=
 * All assignment operators expect an rvalue reference (Reference &&) for
 * the Reference class in order to prevent copying the Reference object.
 *
 * @tparam Reference  The reference class this mixin provides operator
 * overloads for. The reference class needs to overload the cast operator
 * to ValueType
 *
 * @tparam ArithmeticType  arithmetic type the Reference class is supposed
 * to represent.
 *
 * @warning  This struct should only be used by reference classes.
 */
template <typename Reference, typename ArithmeticType>
struct enable_reference {
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;


#define GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(_oper, _op)          \
    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR \
        arithmetic_type                                             \
        _oper(const Reference &ref1, const Reference &ref2)         \
    {                                                               \
        return to_value_type<arithmetic_type>(ref1)                 \
            _op to_value_type<arithmetic_type>(ref2);               \
    }                                                               \
    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR \
        arithmetic_type                                             \
        _oper(const Reference &ref, const arithmetic_type &a)       \
    {                                                               \
        return to_value_type<arithmetic_type>(ref) _op a;           \
    }                                                               \
    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR \
        arithmetic_type                                             \
        _oper(const arithmetic_type &a, const Reference &ref)       \
    {                                                               \
        return a _op to_value_type<arithmetic_type>(ref);           \
    }
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator*, *)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator/, /)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator+, +)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator-, -)
#undef GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD

#define GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(_oper, _op)             \
    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR        \
        arithmetic_type                                                    \
        _oper(Reference &&ref1, const Reference &ref2)                     \
    {                                                                      \
        return std::move(ref1) = to_value_type<arithmetic_type>(ref1)      \
                   _op to_value_type<arithmetic_type>(ref2);               \
    }                                                                      \
    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR        \
        arithmetic_type                                                    \
        _oper(Reference &&ref, const arithmetic_type &a)                   \
    {                                                                      \
        return std::move(ref) = to_value_type<arithmetic_type>(ref) _op a; \
    }

    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator*=, *)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator/=, /)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator+=, +)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator-=, -)
#undef GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD

    // TODO test if comparison operators need to be overloaded as well

    friend GKO_ATTRIBUTES GKO_INLINE GKO_ENABLE_REFERENCE_CONSTEXPR
        arithmetic_type
        operator-(const Reference &ref)
    {
        return -to_value_type<arithmetic_type>(ref);
    }
};


/**
 * This namespace contains reference classes used inside accessors.
 *
 * @warning  This class should not be used by anything else but accessors.
 */
namespace reference {


/**
 * Reference class for a different storage than arithmetic type. The
 * conversion between both formats is done with a simple static_cast.
 *
 * Copying this reference is disabled, but move construction is possible to
 * allow for an additional layer (like gko::range).
 * The assignment operator only works for an rvalue reference (&&) to
 * prevent accidental copying the reference and working on a reference.
 *
 * @tparam ArithmeticType  Type used for arithmetic operations, therefore,
 * the type which is used for input and output of this class.
 *
 * @tparam StorageType  Type actually used as a storage, which is converted
 * to ArithmeticType before usage
 */
template <typename ArithmeticType, typename StorageType>
class ReducedStorageReference
    : public detail::enable_reference<
          ReducedStorageReference<ArithmeticType, StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    // Allow move construction, so perfect forwarding is possible (required
    // for `range` support)
    ReducedStorageReference(ReducedStorageReference &&) = default;

    ReducedStorageReference() = delete;
    // Forbid copy construction and move assignment
    ReducedStorageReference(const ReducedStorageReference &) = delete;
    ReducedStorageReference &operator=(ReducedStorageReference &&) = delete;

    GKO_ATTRIBUTES constexpr ReducedStorageReference(
        storage_type *const GKO_RESTRICT ptr)
        : ptr_{ptr}
    {}
    ~ReducedStorageReference() = default;

    GKO_ATTRIBUTES GKO_INLINE constexpr operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr);
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr arithmetic_type operator=(
        arithmetic_type val) &&
    {
        storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return *r_ptr = static_cast<storage_type>(val);
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr arithmetic_type operator=(
        const ReducedStorageReference &ref) &&
    {
        return std::move(*this) = static_cast<arithmetic_type>(ref);
    }

private:
    storage_type *const GKO_RESTRICT ptr_;
};

// Specialization for const storage_type to prevent `operator=`
template <typename ArithmeticType, typename StorageType>
class ReducedStorageReference<ArithmeticType, const StorageType>
    : public detail::enable_reference<
          ReducedStorageReference<ArithmeticType, const StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = const StorageType;
    // Allow move construction, so perfect forwarding is possible
    ReducedStorageReference(ReducedStorageReference &&) = default;

    ReducedStorageReference() = delete;
    // Forbid copy construction and move assignment
    ReducedStorageReference(const ReducedStorageReference &) = delete;
    ReducedStorageReference &operator=(ReducedStorageReference &&) = delete;

    GKO_ATTRIBUTES constexpr ReducedStorageReference(
        storage_type *const GKO_RESTRICT ptr)
        : ptr_{ptr}
    {}
    ~ReducedStorageReference() = default;

    GKO_ATTRIBUTES GKO_INLINE constexpr operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr);
    }

private:
    storage_type *const GKO_RESTRICT ptr_;
};


/**
 * Reference class for a different storage than arithmetic type with the
 * addition of a scaling factor. The conversion between both formats is done
 * with a static_cast to the ArithmeticType, followed by a multiplication
 * of the scale (when reading; for writing, the new value is divided by the
 * scale before casting to the StorageType).
 *
 * Copying this reference is disabled, but move construction is possible to
 * allow for an additional layer (like gko::range).
 * The assignment operator only works for an rvalue reference (&&) to
 * prevent accidental copying the reference and working on a reference.
 *
 * @tparam ArithmeticType  Type used for arithmetic operations, therefore,
 * the type which is used for input and output of this class.
 *
 * @tparam StorageType  Type actually used as a storage, which is converted
 * to ArithmeticType before usage
 */
template <typename ArithmeticType, typename StorageType>
class ScaledReducedStorageReference
    : public detail::enable_reference<
          ScaledReducedStorageReference<ArithmeticType, StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;

    // Allow move construction, so perfect forwarding is possible
    ScaledReducedStorageReference(ScaledReducedStorageReference &&) = default;

    ScaledReducedStorageReference() = delete;
    // Forbid copy construction and move assignment
    ScaledReducedStorageReference(const ScaledReducedStorageReference &) =
        delete;
    ScaledReducedStorageReference &operator=(ScaledReducedStorageReference &&) =
        delete;

    GKO_ATTRIBUTES constexpr ScaledReducedStorageReference(
        storage_type *const GKO_RESTRICT ptr, arithmetic_type scale)
        : ptr_{ptr}, scale_{scale}
    {}
    ~ScaledReducedStorageReference() = default;

    GKO_ATTRIBUTES GKO_INLINE constexpr operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr) * scale_;
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr arithmetic_type operator=(
        arithmetic_type val) &&
    {
        storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return *r_ptr = static_cast<storage_type>(val / scale_), val;
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr arithmetic_type operator=(
        const ScaledReducedStorageReference &ref) &&
    {
        return std::move(*this) = static_cast<arithmetic_type>(ref);
    }

private:
    storage_type *const GKO_RESTRICT ptr_;
    const arithmetic_type scale_;
};

// Specialization for constant storage_type (no `operator=`)
template <typename ArithmeticType, typename StorageType>
class ScaledReducedStorageReference<ArithmeticType, const StorageType>
    : public detail::enable_reference<
          ScaledReducedStorageReference<ArithmeticType, const StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = const StorageType;

    // Allow move construction, so perfect forwarding is possible
    ScaledReducedStorageReference(ScaledReducedStorageReference &&) = default;

    ScaledReducedStorageReference() = delete;
    // Forbid copy construction and move assignment
    ScaledReducedStorageReference(const ScaledReducedStorageReference &) =
        delete;
    ScaledReducedStorageReference &operator=(ScaledReducedStorageReference &&) =
        delete;

    GKO_ATTRIBUTES constexpr ScaledReducedStorageReference(
        storage_type *const GKO_RESTRICT ptr, arithmetic_type scale)
        : ptr_{ptr}, scale_{scale}
    {}
    ~ScaledReducedStorageReference() = default;

    GKO_ATTRIBUTES GKO_INLINE constexpr operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr) * scale_;
    }


private:
    storage_type *const GKO_RESTRICT ptr_;
    const arithmetic_type scale_;
};


}  // namespace reference
}  // namespace detail


/**
 * The ReducedStorage3d class allows a storage format that is different from
 * the arithmetic format (which is returned from the brace operator).
 * As storage, the storage_type is used.
 * This accessor uses row-major access.
 *
 * @note  This class only manages the accesses and not the memory itself.
 *
 * TODO Rename, so it is in lower_case only (since no virtual functions!)
 */
template <typename ArithmeticType, typename StorageType>
class ReducedStorage3d {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    using const_accessor =
        ReducedStorage3d<arithmetic_type, const storage_type>;
    static constexpr size_type dimensionality{3};
    static constexpr bool is_const{std::is_const<storage_type>::value};

protected:
    using reference =
        detail::reference::ReducedStorageReference<arithmetic_type,
                                                   storage_type>;

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so ReducedStorage3d(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion in
     * this class, so it is not invalid or UB code.
     */
    GKO_ATTRIBUTES constexpr ReducedStorage3d(storage_type *storage,
                                              gko::dim<dimensionality> size,
                                              size_type stride0,
                                              size_type stride1)
        : storage_{storage}, size_{size}, stride_{stride0, stride1}
    {}

    GKO_ATTRIBUTES constexpr ReducedStorage3d(storage_type *storage,
                                              dim<dimensionality> size)
        : ReducedStorage3d{storage, size, size[1] * size[2], size[2]}
    {}
    GKO_ATTRIBUTES constexpr ReducedStorage3d()
        : ReducedStorage3d{nullptr, {0, 0, 0}}
    {}

    GKO_ATTRIBUTES GKO_INLINE constexpr const_accessor to_const() const
    {
        return {storage_, size_, stride_[0], stride_[1]};
    }

    // Functions required by the `range` interface:
    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> length(
        size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }
    /**
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other)
    {
        // TODO figure out how to do this properly in an const accessor...
        //      Additionally, this will always be inefficient!
        for (size_type i = 0; i < size_[0]; ++i) {
            for (size_type j = 0; j < size_[1]; ++j) {
                for (size_type k = 0; k < size_[2]; ++k) {
                    (*this)(i, j, k) = other(i, j, k);
                }
            }
        }
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr std::conditional_t<
        is_const, arithmetic_type, reference>
    operator()(size_type x, size_type y, size_type z) const
    {
        return reference{this->storage_ + compute_index(x, y, z)};
    }

    GKO_ATTRIBUTES constexpr range<ReducedStorage3d> operator()(
        const span &x_span, const span &y_span, const span &z_span) const
    {
        return GKO_ASSERT(x_span.is_valid()), GKO_ASSERT(y_span.is_valid()),
               GKO_ASSERT(z_span.is_valid()),
               GKO_ASSERT(x_span <= span{size_[0]}),
               GKO_ASSERT(y_span <= span{size_[1]}),
               GKO_ASSERT(z_span <= span{size_[2]}),
               range<ReducedStorage3d>(
                   storage_ +
                       compute_index(x_span.begin, y_span.begin, z_span.begin),
                   dim<dimensionality>{x_span.end - x_span.begin,
                                       y_span.end - y_span.begin,
                                       z_span.end - z_span.begin},
                   stride_[0], stride_[1]);
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> get_size() const
    {
        return size_;
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type get_stride0() const
    {
        return stride_[0];
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type get_stride1() const
    {
        return stride_[1];
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr storage_type *get_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr const storage_type *get_const_storage()
        const
    {
        return storage_;
    }

protected:
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type compute_index(
        size_type x, size_type y, size_type z) const
    {
        return GKO_ASSERT(x < size_[0]), GKO_ASSERT(y < size_[1]),
               GKO_ASSERT(z < size_[2]), x * stride_[0] + y * stride_[1] + z;
    }

    storage_type *storage_;
    dim<dimensionality> size_;
    size_type stride_[2];  // std::array leads to conflicts on devices
};


namespace detail {


// In case of a const type, do not provide a write function
template <typename Accessor, typename ScaleType,
          bool = std::is_const<ScaleType>::value>
struct enable_scale_write {
    static_assert(std::is_const<ScaleType>::value,
                  "This class must have a constant ScaleType!");
    using scale_type = ScaleType;
};

// In case of a non-const type, enable the write function
template <typename Accessor, typename ScaleType>
struct enable_scale_write<Accessor, ScaleType, false> {
    static_assert(!std::is_const<ScaleType>::value,
                  "This class must NOT have a constant ScaleType!");
    using scale_type = ScaleType;

    /**
     * Reads the scale value at the given indices.
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr scale_type set_scale(
        size_type x, size_type z, scale_type value) const
    {
        scale_type *GKO_RESTRICT rest_scale = self()->scale_;
        return rest_scale[self()->compute_scale_index(x, z)] = value;
    }

private:
    GKO_ATTRIBUTES GKO_INLINE constexpr const Accessor *self() const
    {
        return static_cast<const Accessor *>(this);
    }
};


}  // namespace detail


/**
 * @internal
 *
 * The ScaledReducedStorage3d class hides the underlying storage format and
 * provides a simple interface for accessing a one dimensional storage.
 * Additionally, this accessor posesses a scale array, which is used for
 * each read and write operation to do a proper conversion.
 *
 * This class only manages the accesses, however, and not the memory itself.
 *
 * The accessor uses row-major access.
 */
template <typename ArithmeticType, typename StorageType>
class ScaledReducedStorage3d
    : public detail::enable_scale_write<
          ScaledReducedStorage3d<ArithmeticType, StorageType>, ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr size_type dimensionality{3};
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using scale_type =
        std::conditional_t<is_const, const arithmetic_type, arithmetic_type>;

    // Allow access to both `scale_` and `compute_scale_index()`
    friend detail::enable_scale_write<ScaledReducedStorage3d, scale_type>;
    using const_accessor =
        ScaledReducedStorage3d<arithmetic_type, const storage_type>;

protected:
    using reference =
        detail::reference::ScaledReducedStorageReference<arithmetic_type,
                                                         StorageType>;

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    GKO_ATTRIBUTES constexpr ScaledReducedStorage3d(storage_type *storage,
                                                    dim<dimensionality> size,
                                                    size_type stride0,
                                                    size_type stride1,
                                                    scale_type *scale)
        : storage_{storage},
          size_{size},
          stride_{stride0, stride1},
          scale_{scale}
    {}
    GKO_ATTRIBUTES constexpr ScaledReducedStorage3d(storage_type *storage,
                                                    dim<dimensionality> size,
                                                    scale_type *scale)
        : ScaledReducedStorage3d{storage, size, size[1] * size[2], size[2],
                                 scale}
    {}
    GKO_ATTRIBUTES constexpr ScaledReducedStorage3d()
        : ScaledReducedStorage3d{nullptr, {0, 0, 0}, nullptr}
    {}

    GKO_ATTRIBUTES GKO_INLINE constexpr const_accessor to_const() const
    {
        return {storage_, size_, stride_[0], stride_[1], scale_};
    }

    /**
     * Reads the scale value at the given indices.
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr scale_type read_scale(size_type x,
                                                              size_type z) const
    {
        const arithmetic_type *GKO_RESTRICT rest_scale = scale_;
        return rest_scale[compute_scale_index(x, z)];
    }

    // Functions required by the `range` interface:
    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> length(
        size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }

    /**
     * @warning Do not use this function since it is not optimized for a
     * specific executor. It will always be performed sequentially.
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other)
    {
        // TODO figure out how to do this properly in an const accessor...
        //      Additionally, this will always be inefficient!
        for (size_type i = 0; i < this->size_[0]; ++i) {
            for (size_type j = 0; j < this->size_[1]; ++j) {
                for (size_type k = 0; k < this->size_[2]; ++k) {
                    (*this)(i, j, k) = other(i, j, k);
                }
            }
        }
        for (size_type i = 0; i < this->size_[0]; ++i) {
            for (size_type k = 0; k < this->size_[2]; ++k) {
                this->set_scale(i, k, other.read_scale(i, k));
            }
        }
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr std::conditional_t<
        is_const, arithmetic_type, reference>
    operator()(size_type x, size_type y, size_type z) const
    {
        return reference{this->storage_ + this->compute_index(x, y, z),
                         this->read_scale(x, z)};
    }

    GKO_ATTRIBUTES constexpr range<ScaledReducedStorage3d> operator()(
        const span &x_span, const span &y_span, const span &z_span) const
    {
        return GKO_ASSERT(x_span.is_valid()), GKO_ASSERT(y_span.is_valid()),
               GKO_ASSERT(z_span.is_valid()),
               GKO_ASSERT(x_span <= span{this->size_[0]}),
               GKO_ASSERT(y_span <= span{this->size_[1]}),
               GKO_ASSERT(z_span <= span{this->size_[2]}),
               range<ScaledReducedStorage3d>(
                   this->storage_ + this->compute_index(x_span.begin,
                                                        y_span.begin,
                                                        z_span.begin),
                   dim<dimensionality>{x_span.end - x_span.begin,
                                       y_span.end - y_span.begin,
                                       z_span.end - z_span.begin},
                   this->stride_[0], this->stride_[1], this->scale_);
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> get_size() const
    {
        return size_;
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type get_stride0() const
    {
        return stride_[0];
    }
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type get_stride1() const
    {
        return stride_[1];
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr storage_type *get_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr const storage_type *get_const_storage()
        const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr scale_type *get_scale() const
    {
        return this->scale_;
    }

    GKO_ATTRIBUTES GKO_INLINE constexpr const scale_type *get_const_scale()
        const
    {
        return this->scale_;
    }

protected:
    GKO_ATTRIBUTES constexpr GKO_INLINE size_type
    compute_index(size_type x, size_type y, size_type z) const
    {
        return GKO_ASSERT(x < size_[0]), GKO_ASSERT(y < size_[1]),
               GKO_ASSERT(z < size_[2]), x * stride_[0] + y * stride_[1] + z;
    }

    GKO_ATTRIBUTES constexpr GKO_INLINE size_type
    compute_scale_index(size_type x, size_type z) const
    {
        return GKO_ASSERT(x < size_[0]), GKO_ASSERT(z < size_[2]),
               x * stride_[1] + z;
    }

    storage_type *storage_;
    dim<dimensionality> size_;
    size_type stride_[2];  // std::array leads to conflicts on devices
    scale_type *scale_;
};


}  // namespace accessor
}  // namespace gko


#endif  // GKO_CORE_BASE_RANGE_ACCESSORS_HPP_
