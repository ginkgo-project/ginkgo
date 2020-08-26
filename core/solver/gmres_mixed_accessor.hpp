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

#ifndef GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
#define GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_


#include <cinttypes>
#include <limits>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include <iostream>
#define GKO_DEBUG_OUTPUT false


namespace gko {
namespace kernels {  // TODO maybe put into another separate namespace


namespace detail {


// using place_holder_type = float;


template <typename StorageType, typename ArithmeticType,
          bool = std::is_integral<StorageType>::value>
/*          bool = (std::is_same<StorageType, place_holder_type>::value &&
                  !std::is_same<StorageType, ArithmeticType>::value) ||
                 std::is_integral<StorageType>::value>
*/
struct helper_have_scale {};

template <typename StorageType, typename ArithmeticType>
struct helper_have_scale<StorageType, ArithmeticType, false>
    : public std::false_type {};

template <typename StorageType, typename ArithmeticType>
struct helper_have_scale<StorageType, ArithmeticType, true>
    : public std::true_type {};

/**
 * This is a mixin which defines the binary operators for *, /, +, - for the
 * Reference class, the unary operator -, and the assignment operators
 * *=, /=, +=, -=
 * Additionally, it prevents the default generation of copy and move constructor
 * and copy and move assignment.
 */
template <typename Reference, typename ArithmeticType>
struct enable_reference {
    using arithmetic_type = ArithmeticType;

    GKO_ATTRIBUTES enable_reference() =
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
 * @internal
 *
 * The ReducedStorage3d class hides the underlying storage_ format and provides
 * a simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses and not the memory itself.
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
class ConstReducedStorage3d {
public:
    using arithmetic_type = ArithmeticType;
    using storage_type = StorageType;
    using const_accessor = ConstReducedStorage3d;
    static_assert(!std::is_const<storage_type>::value,
                  "StorageType must not be const!");

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
    size_type stride_[2];
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
    size_type stride_[2];
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


namespace detail {


template <typename Accessor>
struct is_3d_scaled_accessor : public std::false_type {};

template <typename... Args>
struct is_3d_scaled_accessor<ScaledReducedStorage3d<Args...>>
    : public std::true_type {};

template <typename StorageType, bool = std::is_integral<StorageType>::value>
struct helper_require_scale {};

template <typename StorageType>
struct helper_require_scale<StorageType, false> : public std::false_type {};

template <typename StorageType>
struct helper_require_scale<StorageType, true> : public std::true_type {};


}  // namespace detail


template <typename ValueType, typename StorageType,
          bool = detail::helper_require_scale<StorageType>::value>
class Accessor3dHelper {};


template <typename ValueType, typename StorageType>
class Accessor3dHelper<ValueType, StorageType, true> {
public:
    using Accessor = ScaledReducedStorage3d<ValueType, StorageType>;

    Accessor3dHelper() = default;

    Accessor3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{krylov_dim},
          bases_{exec, krylov_dim_[0] * krylov_dim_[1] * krylov_dim_[2]},
          scale_{exec, krylov_dim_[0] * krylov_dim_[2]}
    {
        // For testing, initialize scale to ones
        // Array<ValueType> h_scale{exec->get_master(), krylov_dim[0]};
        Array<ValueType> h_scale{exec->get_master(),
                                 krylov_dim[0] * krylov_dim[2]};
        for (size_type i = 0; i < h_scale.get_num_elems(); ++i) {
            h_scale.get_data()[i] = one<ValueType>();
        }
        scale_ = h_scale;
    }

    Accessor get_accessor()
    {
        const auto stride0 = krylov_dim_[1] * krylov_dim_[2];
        const auto stride1 = krylov_dim_[2];
        return {bases_.get_data(), stride0, stride1, scale_.get_data()};
    }

    gko::Array<StorageType> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<StorageType> bases_;
    Array<ValueType> scale_;
};


template <typename ValueType, typename StorageType>
class Accessor3dHelper<ValueType, StorageType, false> {
public:
    using Accessor = ReducedStorage3d<ValueType, StorageType>;

    Accessor3dHelper() = default;

    Accessor3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{krylov_dim},
          bases_{std::move(exec),
                 krylov_dim_[0] * krylov_dim_[1] * krylov_dim_[2]}
    {}

    Accessor get_accessor()
    {
        const auto stride0 = krylov_dim_[1] * krylov_dim_[2];
        const auto stride1 = krylov_dim_[2];
        return {bases_.get_data(), stride0, stride1};
    }

    gko::Array<StorageType> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<StorageType> bases_;
};

//----------------------------------------------

template <typename Accessor3d,
          bool = detail::is_3d_scaled_accessor<Accessor3d>::value>
struct helper_functions_accessor {};

// Accessors having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, true> {
    using arithmetic_type = typename Accessor3d::arithmetic_type;
    static_assert(detail::is_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  arithmetic_type value)
    {
        using storage_type = typename Accessor3d::storage_type;
        constexpr arithmetic_type correction =
            std::is_integral<storage_type>::value
                // Use 2 instead of 1 here to allow for a bit more room
                ? 2 / static_cast<arithmetic_type>(
                          std::numeric_limits<storage_type>::max())
                : 1;
        krylov_bases.set_scale(vector_idx, col_idx, value * correction);
    }
};

// Accessors not having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, false> {
    using arithmetic_type = typename Accessor3d::arithmetic_type;
    static_assert(!detail::is_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must not have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  arithmetic_type value)
    {}
};

// calling it with:
// helper_functions_accessor<Accessor3d>::write_scale(krylov_bases, col_idx,
// value);

//----------------------------------------------

}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
