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


}  // namespace detail


template <typename StorageType, typename ArithmeticType,
          bool = detail::helper_have_scale<StorageType, ArithmeticType>::value>
class ConstAccessor3d {};

/**
 * @internal
 *
 * The Accessor3d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 *
 * The dimensions {x, y, z} explained for the krylov_bases:
 * - x: selects the krylov vector (usually has krylov_dim + 1 vectors)
 * - y: selects the (row-)element of said krylov vector
 * - z: selects which column-element of said krylov vector should be used
 *
 * The accessor uses row-major access.
 */
template <typename StorageType, typename ArithmeticType>
class ConstAccessor3d<StorageType, ArithmeticType, false> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    using const_accessor = ConstAccessor3d;
    static_assert(
        !detail::helper_have_scale<StorageType, ArithmeticType>::value,
        "storage_type must not be an integral in this class.");
    static constexpr bool has_scale{false};

protected:
    class reference
        : public detail::enable_reference<reference, arithmetic_type> {
    public:
        reference() = delete;
        GKO_ATTRIBUTES reference(storage_type *const GKO_RESTRICT ptr)
            : ptr_{ptr}
        {}
        GKO_ATTRIBUTES constexpr GKO_INLINE operator arithmetic_type() const
        {
            return static_cast<arithmetic_type>(*ptr_);
        }
        GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type
        operator=(arithmetic_type val) &&
        {
            return *ptr_ = static_cast<storage_type>(val);
        }
        GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type
        operator=(const reference &ref) &&
        {
            return std::move(*this) = static_cast<arithmetic_type>(ref);
        }

    private:
        storage_type *const GKO_RESTRICT ptr_;
    };

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so Accessor3d(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    GKO_ATTRIBUTES ConstAccessor3d(const storage_type *storage,
                                   size_type stride0, size_type stride1)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1}
    {}

    GKO_ATTRIBUTES const_accessor to_const() const { return *this; }

    /**
     * Reads the value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read(IndexType x, IndexType y,
                                        IndexType z) const
    {
        // Make use of the restrict (and const) qualifier. If the restrict
        // qualifier would have been put on the class attribute, it would be
        // ignored by nvcc.
        const storage_type *GKO_RESTRICT rest_storage = storage_;
        return static_cast<arithmetic_type>(
            rest_storage[x * stride_[0] + y * stride_[1] + z]);
    }

    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type at(IndexType x, IndexType y,
                                      IndexType z) const
    {
        return reference{storage_ + x * stride_[0] + y * stride_[1] + z};
    }

    GKO_ATTRIBUTES size_type get_stride0() const { return stride_[0]; }
    GKO_ATTRIBUTES size_type get_stride1() const { return stride_[1]; }

    GKO_ATTRIBUTES const storage_type *get_storage() const { return storage_; }

    GKO_ATTRIBUTES const storage_type *get_const_storage() const
    {
        return storage_;
    }

protected:
    storage_type *storage_;
    size_type stride_[2];
};

/**
 * @internal
 *
 * The Accessor3d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 * Additionally, this accessor posesses a scale array, which is used for each
 * read and write operation to do a proper conversion.
 *
 * This class only manages the accesses, however, and not the memory itself.
 *
 * The dimensions {x, y, z} explained for the krylov_bases:
 * - x: selects the krylov vector (usually has krylov_dim + 1 vectors)
 * - y: selects the (row-)element of said krylov vector
 * - z: selects which column-element of said krylov vector should be used
 *
 * The accessor uses row-major access.
 */
template <typename StorageType, typename ArithmeticType>
class ConstAccessor3d<StorageType, ArithmeticType, true> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    using const_accessor = ConstAccessor3d;
    static_assert(detail::helper_have_scale<StorageType, ArithmeticType>::value,
                  "storage_type must not be an integral in this class.");

    static constexpr bool has_scale{true};

protected:
    class reference
        : public detail::enable_reference<reference, arithmetic_type> {
    public:
        reference() = delete;
        GKO_ATTRIBUTES reference(storage_type *const GKO_RESTRICT ptr,
                                 arithmetic_type scale)
            : ptr_{ptr}, scale_{scale}
        {}
        GKO_ATTRIBUTES constexpr GKO_INLINE operator arithmetic_type() const
        {
            return static_cast<arithmetic_type>(*ptr_) * scale_;
        }
        GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type
        operator=(arithmetic_type val) &&
        {
            return *ptr_ = static_cast<storage_type>(val / scale_), val;
        }
        GKO_ATTRIBUTES constexpr GKO_INLINE arithmetic_type
        operator=(const reference &ref) &&
        {
            return std::move(*this) = static_cast<arithmetic_type>(ref);
        }

    private:
        storage_type *const GKO_RESTRICT ptr_;
        const arithmetic_type scale_;
    };

public:
    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so Accessor3d(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    GKO_ATTRIBUTES ConstAccessor3d(const storage_type *storage,
                                   size_type stride0, size_type stride1,
                                   const arithmetic_type *scale)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1},
          scale_{const_cast<arithmetic_type *>(scale)}
    {}

    GKO_ATTRIBUTES const_accessor to_const() const { return *this; }

    /**
     * Reads the scale value at the given indices.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read_scale(IndexType x, IndexType z) const
    {
        const arithmetic_type *GKO_RESTRICT rest_scale = scale_;
        return rest_scale[x * stride_[1] + z];
    }

    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type at(IndexType x, IndexType y,
                                      IndexType z) const
    {
        return reference{storage_ + x * stride_[0] + y * stride_[1] + z,
                         read_scale(x, z)};
    }

    /**
     * Reads the value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read(IndexType x, IndexType y,
                                        IndexType z) const
    {
        // Make use of the restrict (and const) qualifier. If the restrict
        // qualifier would have been put on the class attribute, it would be
        // ignored by nvcc.
        const storage_type *GKO_RESTRICT rest_storage = storage_;
        return static_cast<arithmetic_type>(
                   rest_storage[x * stride_[0] + y * stride_[1] + z]) *
               read_scale(x, z);
    }

    GKO_ATTRIBUTES size_type get_stride0() const { return stride_[0]; }
    GKO_ATTRIBUTES size_type get_stride1() const { return stride_[1]; }

    GKO_ATTRIBUTES const storage_type *get_storage() const { return storage_; }

    GKO_ATTRIBUTES const storage_type *get_const_storage() const
    {
        return storage_;
    }

    GKO_ATTRIBUTES const arithmetic_type *get_scale() const
    {
        return this->scale_;
    }

    GKO_ATTRIBUTES const arithmetic_type *get_const_scale() const
    {
        return this->scale_;
    }

protected:
    storage_type *storage_;
    size_type stride_[2];
    arithmetic_type *scale_;
};

template <typename StorageType, typename ArithmeticType,
          bool = detail::helper_have_scale<StorageType, ArithmeticType>::value>
class Accessor3d : public ConstAccessor3d<StorageType, ArithmeticType> {};

template <typename StorageType, typename ArithmeticType>
class Accessor3d<StorageType, ArithmeticType, false>
    : public ConstAccessor3d<StorageType, ArithmeticType, false> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    using const_accessor = ConstAccessor3d<storage_type, arithmetic_type>;
    static_assert(
        !detail::helper_have_scale<StorageType, ArithmeticType>::value,
        "storage_type must not be an integral in this class.");
    static constexpr bool has_scale{false};

protected:
    using reference = typename const_accessor::reference;

public:
    GKO_ATTRIBUTES Accessor3d() : const_accessor(nullptr, 0, 0) {}

    GKO_ATTRIBUTES Accessor3d(storage_type *storage, size_type stride0,
                              size_type stride1)
        : const_accessor(storage, stride0, stride1)
    {}

    GKO_ATTRIBUTES const_accessor to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1]};
    }

    GKO_ATTRIBUTES operator const_accessor() const { return to_const(); }

    using const_accessor::get_storage;
    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType x, IndexType y, IndexType z,
                              arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = this->storage_;
        rest_storage[x * this->stride_[0] + y * this->stride_[1] + z] =
            static_cast<storage_type>(value);
    }

    using const_accessor::at;  // Makes sure the at() const function is visible
    template <typename IndexType>
    GKO_ATTRIBUTES reference at(IndexType x, IndexType y, IndexType z)
    {
        return {this->storage_ + x * this->stride_[0] + y * this->stride_[1] +
                z};
    }
};


template <typename StorageType, typename ArithmeticType>
class Accessor3d<StorageType, ArithmeticType, true>
    : public ConstAccessor3d<StorageType, ArithmeticType, true> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    using const_accessor = ConstAccessor3d<storage_type, arithmetic_type>;
    static_assert(detail::helper_have_scale<StorageType, ArithmeticType>::value,
                  "storage_type must not be an integral in this class.");

    static constexpr bool has_scale{true};

protected:
    using reference = typename const_accessor::reference;

public:
    GKO_ATTRIBUTES Accessor3d() : const_accessor(nullptr, {}, {}, nullptr) {}

    GKO_ATTRIBUTES Accessor3d(storage_type *storage, size_type stride0,
                              size_type stride1, arithmetic_type *scale)
        : const_accessor(storage, stride0, stride1, scale)
    {}

    GKO_ATTRIBUTES const_accessor to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1],
                this->scale_};
    }

    GKO_ATTRIBUTES operator const_accessor() const { return to_const(); }


    using const_accessor::at;
    template <typename IndexType>
    GKO_ATTRIBUTES reference at(IndexType x, IndexType y, IndexType z)
    {
        return {
            this->storage_ + x * this->stride_[0] + y * this->stride_[1] + z,
            this->read_scale(x, z)};
    }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType x, IndexType y, IndexType z,
                              arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = this->storage_;
        const auto stride0 = this->stride_[0];
        const auto stride1 = this->stride_[1];
        rest_storage[x * stride0 + y * stride1 + z] =
            static_cast<storage_type>(value / this->read_scale(x, z));
#if GKO_DEBUG_OUTPUT && not defined(__CUDA_ARCH__)
        std::cout << "storage[" << x << ", " << y << ", " << z
                  << "] = " << rest_storage[x * stride0 + y * stride1 + z]
                  << "; value = " << value
                  << "; scale = " << this->read_scale(x, z) << '\n';
#endif
    }

    /**
     * Writes the given value at the given index for a scale.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void set_scale(IndexType x, IndexType z, arithmetic_type val)
    {
        arithmetic_type *GKO_RESTRICT rest_scale = this->scale_;
        storage_type max_val = one<storage_type>();
        // printf("(A) max_val = %d\n", max_val);
        if (std::is_integral<storage_type>::value) {
            max_val = std::numeric_limits<storage_type>::max();
            //    printf("(B) max_val = %ld\n", max_val);
        }
        rest_scale[x * this->stride_[1] + z] =
            val / static_cast<arithmetic_type>(max_val);
#if GKO_DEBUG_OUTPUT && not defined(__CUDA_ARCH__)
        std::cout << "scale[" << x << ", " << z
                  << "] = " << rest_scale[x * this->stride_[1] + z]
                  << "; val = " << val
                  << "; max_val = " << static_cast<arithmetic_type>(max_val)
                  << '\n';
#endif
        //    rest_scale[idx] = one<arithmetic_type>();
        //    printf("val = %10.5e , rest_scale = %10.5e\n", val,
        //    rest_scale[idx]); std::cout << val << " - " << rest_scale[idx] <<
        //    std::endl;
    }

    using const_accessor::get_storage;
    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }

    using const_accessor::get_scale;
    GKO_ATTRIBUTES arithmetic_type *get_scale() { return this->scale_; }
};


template <typename ValueType, typename ValueTypeKrylovBases,
          bool = Accessor3d<ValueTypeKrylovBases, ValueType>::has_scale>
class Accessor3dHelper {};


template <typename ValueType, typename ValueTypeKrylovBases>
class Accessor3dHelper<ValueType, ValueTypeKrylovBases, true> {
public:
    using Accessor = Accessor3d<ValueTypeKrylovBases, ValueType>;
    static_assert(Accessor::has_scale,
                  "This accessor must have a scale in this class!");

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

    gko::Array<ValueTypeKrylovBases> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<ValueTypeKrylovBases> bases_;
    Array<ValueType> scale_;
};


template <typename ValueType, typename ValueTypeKrylovBases>
class Accessor3dHelper<ValueType, ValueTypeKrylovBases, false> {
public:
    using Accessor = Accessor3d<ValueTypeKrylovBases, ValueType>;
    static_assert(!Accessor::has_scale,
                  "This accessor must not have a scale in this class!");

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

    gko::Array<ValueTypeKrylovBases> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<ValueTypeKrylovBases> bases_;
};

//----------------------------------------------

template <typename Accessor3d, bool = Accessor3d::has_scale>
struct helper_functions_accessor {};

// Accessors having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, true> {
    using value_type = typename Accessor3d::arithmetic_type;
    static_assert(Accessor3d::has_scale, "Accessor must have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  value_type value)
    {
        krylov_bases.set_scale(vector_idx, col_idx, value);
    }
};


// Accessors not having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, false> {
    using value_type = typename Accessor3d::arithmetic_type;
    static_assert(!Accessor3d::has_scale,
                  "Accessor must not have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  value_type value)
    {}
};

// calling it with:
// helper_functions_accessor<Accessor3d>::write_scale(krylov_bases, col_idx,
// value);

//----------------------------------------------

}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
