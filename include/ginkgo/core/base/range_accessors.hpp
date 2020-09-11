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
#include <ginkgo/core/base/types.hpp>


#include <iostream>
#define GKO_DEBUG_OUTPUT false


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
 * @tparam Dimensionality  number of dimensions of this accessor (has to be 2)
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
     *                consecutive rows (i.e. `data + i * stride` points to the
     *                `i`-th row)
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


/**
 * This is a mixin which defines the binary operators for *, /, +, - for the
 * Reference class, the unary operator -, and the assignment operators
 * *=, /=, +=, -=
 * Additionally, it prevents the default generation of copy and move constructor
 * and copy and move assignment.
 *
 * @warning  This struct should only be used by reference classes.
 */
template <typename Reference, typename ArithmeticType>
struct enable_reference {
    using arithmetic_type = ArithmeticType;

    /*GKO_ATTRIBUTES*/ enable_reference() =
        default;  // Needs to exist, so creation is not probibited
    enable_reference(enable_reference &&) = delete;
    enable_reference(const enable_reference &) = delete;
    enable_reference &operator=(const enable_reference &) = delete;
    enable_reference &operator=(enable_reference &&) = delete;

#define GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(_oper, _op)            \
    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type _oper( \
        const Reference &ref1, const Reference &ref2)                 \
    {                                                                 \
        return static_cast<arithmetic_type>(ref1)                     \
            _op static_cast<arithmetic_type>(ref2);                   \
    }                                                                 \
    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type _oper( \
        const Reference &ref, const arithmetic_type &a)               \
    {                                                                 \
        return static_cast<arithmetic_type>(ref) _op a;               \
    }                                                                 \
    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type _oper( \
        const arithmetic_type &a, const Reference &ref)               \
    {                                                                 \
        return a _op static_cast<arithmetic_type>(ref);               \
    }
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator*, *)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator/, /)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator+, +)
    GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD(operator-, -)
#undef GKO_REFERENCE_BINARY_OPERATOR_OVERLOAD

#define GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(_oper, _op)           \
    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type _oper(    \
        Reference &&ref1, const Reference &ref2)                         \
    {                                                                    \
        return std::move(ref1) = static_cast<arithmetic_type>(ref1)      \
                   _op static_cast<arithmetic_type>(ref2);               \
    }                                                                    \
    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type _oper(    \
        Reference &&ref, const arithmetic_type &a)                       \
    {                                                                    \
        return std::move(ref) = static_cast<arithmetic_type>(ref) _op a; \
    }

    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator*=, *)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator/=, /)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator+=, +)
    GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD(operator-=, -)
#undef GKO_REFERENCE_ASSIGNMENT_OPERATOR_OVERLOAD

    // TODO test if comparison operators need to be overloaded as well

    friend GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type
    operator-(const Reference &ref)
    {
        return -static_cast<arithmetic_type>(ref);
    }
};


/**
 * This namespace contains reference classes used inside accessors.
 *
 * @warning  This class should not be used by anything else but accessors.
 */
namespace reference {


/**
 * Reference class for a different storage than arithmetic type. The conversion
 * between both formats is done with a simple static_cast.
 */
template <typename ArithmeticType, typename StorageType>
class ReducedStorageReference
    : public detail::enable_reference<
          ReducedStorageReference<ArithmeticType, StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    ReducedStorageReference() = delete;
    GKO_ATTRIBUTES ReducedStorageReference(storage_type *const GKO_RESTRICT ptr)
        : ptr_{ptr}
    {}
    GKO_ATTRIBUTES GKO_INLINE operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr);
    }
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type operator=(arithmetic_type val) &&
    {
        storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return *r_ptr = static_cast<storage_type>(val);
    }
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type
    operator=(const ReducedStorageReference &ref) &&
    {
        return std::move(*this) = static_cast<arithmetic_type>(ref);
    }

private:
    storage_type *const GKO_RESTRICT ptr_;
};

/**
 * Reference class for a different storage than arithmetic type. The conversion
 * between both formats is done with a simple static_cast followed by a
 * multiplication with a scale.
 */
template <typename ArithmeticType, typename StorageType>
class ScaledReducedStorageReference
    : public detail::enable_reference<
          ScaledReducedStorageReference<ArithmeticType, StorageType>,
          ArithmeticType> {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;

    ScaledReducedStorageReference() = delete;
    GKO_ATTRIBUTES ScaledReducedStorageReference(
        storage_type *const GKO_RESTRICT ptr, arithmetic_type scale)
        : ptr_{ptr}, scale_{scale}
    {}
    GKO_ATTRIBUTES GKO_INLINE operator arithmetic_type() const
    {
        const storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return static_cast<arithmetic_type>(*r_ptr) * scale_;
    }
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type operator=(arithmetic_type val) &&
    {
        storage_type *const GKO_RESTRICT r_ptr = ptr_;
        return *r_ptr = static_cast<storage_type>(val / scale_), val;
    }
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type
    operator=(const ScaledReducedStorageReference &ref) &&
    {
        return std::move(*this) = static_cast<arithmetic_type>(ref);
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
 */
template <typename ArithmeticType, typename StorageType>
class ConstReducedStorage3d {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    using const_accessor = ConstReducedStorage3d;
    static_assert(!std::is_const<storage_type>::value,
                  "StorageType must not be const!");
    static constexpr size_type dimensionality{3};

protected:
    using reference =
        detail::reference::ReducedStorageReference<ArithmeticType, StorageType>;

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so ReducedStorage3d(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion in this
     * class, so it is not invalid or UB code.
     */
    GKO_ATTRIBUTES ConstReducedStorage3d(const storage_type *storage,
                                         size_type stride0, size_type stride1)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1}
    {}

    GKO_ATTRIBUTES GKO_INLINE const_accessor to_const() const { return *this; }

    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type operator()(IndexType x,
                                                         IndexType y,
                                                         IndexType z) const
    {
        return reference{storage_ + compute_index(x, y, z)};
    }

    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type at(IndexType x, IndexType y,
                                                 IndexType z) const
    {
        return reference{storage_ + compute_index(x, y, z)};
    }

    GKO_ATTRIBUTES GKO_INLINE size_type get_stride0() const
    {
        return stride_[0];
    }
    GKO_ATTRIBUTES GKO_INLINE size_type get_stride1() const
    {
        return stride_[1];
    }

    GKO_ATTRIBUTES GKO_INLINE const storage_type *get_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE const storage_type *get_const_storage() const
    {
        return storage_;
    }

protected:
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type compute_index(
        IndexType x, IndexType y, IndexType z) const
    {
        return x * stride_[0] + y * stride_[1] + z;
    }

    storage_type *storage_;
    size_type stride_[2];  // std::array leads to conflicts on devices
};

template <typename ArithmeticType, typename StorageType>
class ReducedStorage3d
    : public ConstReducedStorage3d<ArithmeticType, StorageType> {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    using const_accessor = ConstReducedStorage3d<arithmetic_type, storage_type>;

protected:
    using reference = typename const_accessor::reference;

public:
    GKO_ATTRIBUTES ReducedStorage3d() : const_accessor(nullptr, 0, 0) {}

    GKO_ATTRIBUTES ReducedStorage3d(storage_type *storage, size_type stride0,
                                    size_type stride1)
        : const_accessor(storage, stride0, stride1)
    {}

    GKO_ATTRIBUTES GKO_INLINE const_accessor to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1]};
    }

    using const_accessor::get_storage;
    GKO_ATTRIBUTES GKO_INLINE storage_type *get_storage()
    {
        return this->storage_;
    }

    // Makes sure the operator() const function is visible
    using const_accessor::operator();
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE reference operator()(IndexType x, IndexType y,
                                                   IndexType z)
    {
        return {this->storage_ + const_accessor::compute_index(x, y, z)};
    }

    using const_accessor::at;  // Makes sure the at() const function is visible
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE reference at(IndexType x, IndexType y,
                                           IndexType z)
    {
        return {this->storage_ + const_accessor::compute_index(x, y, z)};
    }
};


/**
 * @internal
 *
 * The ScaledReducedStorage3d class hides the underlying storage_ format and
 * provides a simple interface for accessing a one dimensional storage.
 * Additionally, this accessor posesses a scale array, which is used for each
 * read and write operation to do a proper conversion.
 *
 * This class only manages the accesses, however, and not the memory itself.
 *
 * TODO move this documentation to the core since this is a general purpose
 * accessor
 *
 * The dimensions {x, y, z} explained for the krylov_bases:
 * - x: selects the krylov vector (usually has krylov_dim + 1 vectors)
 * - y: selects the (row-)element of said krylov vector
 * - z: selects which column-element of said krylov vector should be used
 *
 * The accessor uses row-major access.
 */
template <typename ArithmeticType, typename StorageType>
class ConstScaledReducedStorage3d {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    using const_accessor = ConstScaledReducedStorage3d;
    static_assert(!std::is_const<storage_type>::value,
                  "StorageType must not be const!");
    static constexpr size_type dimensionality{3};

protected:
    using reference =
        detail::reference::ScaledReducedStorageReference<ArithmeticType,
                                                         StorageType>;

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so ScaledReducedStorage3d(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion (from
     * ConstScaledReducedStorage3d), so it is not invalid or UB code.
     */
    GKO_ATTRIBUTES ConstScaledReducedStorage3d(const storage_type *storage,
                                               size_type stride0,
                                               size_type stride1,
                                               const arithmetic_type *scale)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1},
          scale_{const_cast<arithmetic_type *>(scale)}
    {}

    GKO_ATTRIBUTES GKO_INLINE const_accessor to_const() const { return *this; }

    /**
     * Reads the scale value at the given indices.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type read_scale(IndexType x,
                                                         IndexType z) const
    {
        const arithmetic_type *GKO_RESTRICT rest_scale = scale_;
        return rest_scale[x * stride_[1] + z];
    }

    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type operator()(IndexType x,
                                                         IndexType y,
                                                         IndexType z) const
    {
        return reference{storage_ + compute_index(x, y, z), read_scale(x, z)};
    }

    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type at(IndexType x, IndexType y,
                                                 IndexType z) const
    {
        return reference{storage_ + compute_index(x, y, z), read_scale(x, z)};
    }

    GKO_ATTRIBUTES GKO_INLINE size_type get_stride0() const
    {
        return stride_[0];
    }
    GKO_ATTRIBUTES GKO_INLINE size_type get_stride1() const
    {
        return stride_[1];
    }

    GKO_ATTRIBUTES GKO_INLINE const storage_type *get_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE const storage_type *get_const_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES GKO_INLINE const arithmetic_type *get_scale() const
    {
        return this->scale_;
    }

    GKO_ATTRIBUTES GKO_INLINE const arithmetic_type *get_const_scale() const
    {
        return this->scale_;
    }

protected:
    template <typename IndexType>
    GKO_ATTRIBUTES constexpr GKO_INLINE size_type
    compute_index(IndexType x, IndexType y, IndexType z) const
    {
        return x * stride_[0] + y * stride_[1] + z;
    }

    storage_type *storage_;
    size_type stride_[2];  // std::array leads to conflicts on devices
    arithmetic_type *scale_;
};


template <typename ArithmeticType, typename StorageType>
class ScaledReducedStorage3d
    : public ConstScaledReducedStorage3d<ArithmeticType, StorageType> {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    using const_accessor =
        ConstScaledReducedStorage3d<arithmetic_type, storage_type>;

protected:
    using reference = typename const_accessor::reference;

public:
    GKO_ATTRIBUTES ScaledReducedStorage3d()
        : const_accessor(nullptr, {}, {}, nullptr)
    {}

    GKO_ATTRIBUTES ScaledReducedStorage3d(storage_type *storage,
                                          size_type stride0, size_type stride1,
                                          arithmetic_type *scale)
        : const_accessor(storage, stride0, stride1, scale)
    {}

    GKO_ATTRIBUTES GKO_INLINE const_accessor to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1],
                this->scale_};
    }

    // Makes sure the operator() const function is visible
    using const_accessor::operator();
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE reference operator()(IndexType x, IndexType y,
                                                   IndexType z)
    {
        return {this->storage_ + const_accessor::compute_index(x, y, z),
                this->read_scale(x, z)};
    }


    using const_accessor::at;
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE reference at(IndexType x, IndexType y,
                                           IndexType z)
    {
        return {this->storage_ + const_accessor::compute_index(x, y, z),
                this->read_scale(x, z)};
    }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE void write(IndexType x, IndexType y, IndexType z,
                                         arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = this->storage_;
        rest_storage[const_accessor::compute_index(x, y, z)] =
            static_cast<storage_type>(value / this->read_scale(x, z));
#if GKO_DEBUG_OUTPUT && not defined(__CUDA_ARCH__)
        std::cout << "storage[" << x << ", " << y << ", " << z << "] = "
                  << rest_storage[const_accessor::compute_index(x, y, z)]
                  << "; value = " << value
                  << "; scale = " << this->read_scale(x, z) << '\n';
#endif
    }

    /**
     * Writes the given value at the given index for a scale.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES GKO_INLINE void set_scale(IndexType x, IndexType z,
                                             arithmetic_type val)
    {
        arithmetic_type *GKO_RESTRICT rest_scale = this->scale_;
        rest_scale[x * this->stride_[1] + z] = val;
    }

    using const_accessor::get_storage;
    GKO_ATTRIBUTES GKO_INLINE storage_type *get_storage()
    {
        return this->storage_;
    }

    using const_accessor::get_scale;
    GKO_ATTRIBUTES GKO_INLINE arithmetic_type *get_scale()
    {
        return this->scale_;
    }
};


}  // namespace accessor
}  // namespace gko


#endif  // GKO_CORE_BASE_RANGE_ACCESSORS_HPP_
