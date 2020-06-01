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
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include <iostream>


namespace gko {
namespace kernels {  // TODO maybe put into another separate namespace


namespace detail {


using place_holder_type = float;


template <typename StorageType, typename ArithmeticType,
          bool = (std::is_same<StorageType, place_holder_type>::value &&
                  !std::is_same<StorageType, ArithmeticType>::value) ||
                 std::is_integral<StorageType>::value>
// bool = std::is_same<StorageType, place_holder_type>::value &&
//        !std::is_same<StorageType, ArithmeticType>::value>
struct helper_have_scale {};

template <typename StorageType, typename ArithmeticType>
struct helper_have_scale<StorageType, ArithmeticType, false>
    : public std::false_type {};

template <typename StorageType, typename ArithmeticType>
struct helper_have_scale<StorageType, ArithmeticType, true>
    : public std::true_type {};


}  // namespace detail


template <typename StorageType, typename ArithmeticType,
          bool = detail::helper_have_scale<StorageType, ArithmeticType>::value>
class Accessor3dConst {};
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
class Accessor3dConst<StorageType, ArithmeticType, false> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(
        !detail::helper_have_scale<StorageType, ArithmeticType>::value,
        "storage_type must not be an integral in this class.");

    static constexpr bool has_scale{false};

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so Accessor3dConst(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    Accessor3dConst(const storage_type *storage, size_type stride0,
                    size_type stride1)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1}
    {}

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
class Accessor3dConst<StorageType, ArithmeticType, true> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(detail::helper_have_scale<StorageType, ArithmeticType>::value,
                  "storage_type must not be an integral in this class.");

    static constexpr bool has_scale{true};

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     *
     * @internal
     * const_cast needed, so Accessor3dConst(ac.get_storage(), ...)
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    Accessor3dConst(const storage_type *storage, size_type stride0,
                    size_type stride1, const arithmetic_type *scale)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride0, stride1},
          scale_{const_cast<arithmetic_type *>(scale)}
    {}

    /**
     * Reads the scale value at the given indices.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read_scale(IndexType x, IndexType z) const
    {
        const arithmetic_type *GKO_RESTRICT rest_scale = scale_;
        return rest_scale[x * stride_[1] + z];
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
class Accessor3d : public Accessor3dConst<StorageType, ArithmeticType> {};


/**
 * @internal
 *
 * The Accessor3d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor3d<StorageType, ArithmeticType, false>
    : public Accessor3dConst<StorageType, ArithmeticType> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(
        !detail::helper_have_scale<StorageType, ArithmeticType>::value,
        "storage_type must not be an integral in this class.");

private:
    using Accessor3dC = Accessor3dConst<storage_type, arithmetic_type>;

public:
    /**
     * Creates an empty accessor pointing to a nullptr.
     */
    Accessor3d() : Accessor3dC(nullptr, {}, {}) {}

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor3d(storage_type *storage, size_type stride0, size_type stride1)
        : Accessor3dC(storage, stride0, stride1)
    {}

    Accessor3dC to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1]};
    }

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

    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }
};


/**
 * @internal
 *
 * The Accessor3d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor3d<StorageType, ArithmeticType, true>
    : public Accessor3dConst<StorageType, ArithmeticType> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(detail::helper_have_scale<StorageType, ArithmeticType>::value,
                  "storage_type must not be an integral in this class.");

private:
    using Accessor3dC = Accessor3dConst<storage_type, arithmetic_type>;

public:
    /**
     * Creates an empty accessor pointing to a nullptr.
     */
    Accessor3d() : Accessor3dC(nullptr, {}, {}, nullptr) {}

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor3d(storage_type *storage, size_type stride0, size_type stride1,
               arithmetic_type *scale)
        : Accessor3dC(storage, stride0, stride1, scale)
    {}

    Accessor3dC to_const() const
    {
        return {this->storage_, this->stride_[0], this->stride_[1],
                this->scale_};
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
        // if (std::is_integer<storage_type>::value) {
        if (std::is_integral<storage_type>::value) {
            max_val = std::numeric_limits<storage_type>::max();
            //    printf("(B) max_val = %ld\n", max_val);
        }
        rest_scale[x * this->stride_[1] + z] =
            val / static_cast<arithmetic_type>(max_val);
        //    rest_scale[idx] = one<arithmetic_type>();
        //    printf("val = %10.5e , rest_scale = %10.5e\n", val,
        //    rest_scale[idx]); std::cout << val << " - " << rest_scale[idx] <<
        //    std::endl;
    }

    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }

    GKO_ATTRIBUTES arithmetic_type *get_scale() { return this->scale_; }
};


template <typename ValueType, typename ValueTypeKrylovBases,
          bool = Accessor3d<ValueTypeKrylovBases, ValueType>::has_scale>
class Accessor3dHelper {};


template <typename ValueType, typename ValueTypeKrylovBases>
class Accessor3dHelper<ValueType, ValueTypeKrylovBases, true> {
    using Accessor = Accessor3d<ValueTypeKrylovBases, ValueType>;
    static_assert(Accessor::has_scale,
                  "This accessor must have a scale in this class!");

public:
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
    using Accessor = Accessor3d<ValueTypeKrylovBases, ValueType>;
    static_assert(!Accessor::has_scale,
                  "This accessor must not have a scale in this class!");

public:
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

template <typename ValueType, typename KrylovType,
          bool = Accessor3d<KrylovType, ValueType>::has_scale>
struct helper_functions_accessor {};

// Accessors having a scale
template <typename ValueType, typename KrylovType>
struct helper_functions_accessor<ValueType, KrylovType, true> {
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(
        Accessor3d<KrylovType, ValueType> krylov_bases, IndexType vector_idx,
        IndexType col_idx, ValueType value)
    {
        krylov_bases.set_scale(vector_idx, col_idx, value);
    }
};


// Accessors not having a scale
template <typename ValueType, typename KrylovType>
struct helper_functions_accessor<ValueType, KrylovType, false> {
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(
        Accessor3d<KrylovType, ValueType> krylov_bases, IndexType vector_idx,
        IndexType col_idx, ValueType value)
    {}
};

// calling it with:
// helper_functions_accessor<ValueType,
// ValueTypeKrylovBases>::write_scale(krylov_bases, col_idx, value);

//----------------------------------------------

}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
