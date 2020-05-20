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
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {  // TODO maybe put into another separate namespace


namespace detail {


using place_holder_type = float;


template <typename StorageType, typename ArithmeticType,
          bool = std::is_same<StorageType, place_holder_type>::value &&
                 !std::is_same<StorageType, ArithmeticType>::value>
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
class Accessor2dConst {};
/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor2dConst<StorageType, ArithmeticType, false> {
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
     * const_cast needed, so Accessor2dConst(ac.get_storage(), ac.get_stride())
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    Accessor2dConst(const storage_type *storage, size_type stride)
        : storage_{const_cast<storage_type *>(storage)}, stride_{stride}
    {}

    /**
     * Reads the value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read(IndexType row, IndexType col) const
    {
        // Make use of the restrict (and const) qualifier. If the restrict
        // qualifier would have been put on the class attribute, it would be
        // ignored by nvcc.
        const storage_type *GKO_RESTRICT rest_storage = storage_;
        return static_cast<arithmetic_type>(rest_storage[row * stride_ + col]);
    }

    GKO_ATTRIBUTES size_type get_stride() const { return stride_; }

    GKO_ATTRIBUTES const storage_type *get_storage() const { return storage_; }

    GKO_ATTRIBUTES const storage_type *get_const_storage() const
    {
        return storage_;
    }

protected:
    storage_type *storage_;
    size_type stride_;
};


/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 * Additionally, since it is (soon) an integer type, a scale array is also
 * needed to do a proper conversion.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor2dConst<StorageType, ArithmeticType, true> {
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
     * const_cast needed, so Accessor2dConst(ac.get_storage(), ac.get_stride())
     * works. Also, storage_ is never accessed in a non-const fashion, so it
     * is not invalid or UB code.
     */
    Accessor2dConst(const storage_type *storage, size_type stride,
                    const arithmetic_type *scale)
        : storage_{const_cast<storage_type *>(storage)},
          stride_{stride},
          scale_{const_cast<arithmetic_type *>(scale)}
    {}

    /**
     * Reads the value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read(IndexType row, IndexType col) const
    {
        // Make use of the restrict (and const) qualifier. If the restrict
        // qualifier would have been put on the class attribute, it would be
        // ignored by nvcc.
        const storage_type *GKO_RESTRICT rest_storage = storage_;
        const arithmetic_type *GKO_RESTRICT rest_scale = scale_;
        return static_cast<arithmetic_type>(rest_storage[row * stride_ + col]) *
               rest_scale[col];
    }

    GKO_ATTRIBUTES size_type get_stride() const { return stride_; }

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
    arithmetic_type *scale_;
    size_type stride_;
};


template <typename StorageType, typename ArithmeticType,
          bool = detail::helper_have_scale<StorageType, ArithmeticType>::value>
class Accessor2d : public Accessor2dConst<StorageType, ArithmeticType> {};


/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor2d<StorageType, ArithmeticType, false>
    : public Accessor2dConst<StorageType, ArithmeticType> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(
        !detail::helper_have_scale<StorageType, ArithmeticType>::value,
        "storage_type must not be an integral in this class.");

private:
    using Accessor2dC = Accessor2dConst<storage_type, arithmetic_type>;

public:
    /**
     * Creates an empty accessor pointing to a nullptr.
     */
    Accessor2d() : Accessor2dC(nullptr, {}) {}

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor2d(storage_type *storage, size_type stride)
        : Accessor2dC(storage, stride)
    {}
    /*
        operator Accessor2dC() const
        {
            return {this->storage_, this->stride_};
        }
    */

    Accessor2dC to_const() const { return {this->storage_, this->stride_}; }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType row, IndexType col,
                              arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = this->storage_;
        rest_storage[row * this->stride_ + col] =
            static_cast<storage_type>(value);
    }

    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }
};


/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType>
class Accessor2d<StorageType, ArithmeticType, true>
    : public Accessor2dConst<StorageType, ArithmeticType> {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;
    static_assert(detail::helper_have_scale<StorageType, ArithmeticType>::value,
                  "storage_type must not be an integral in this class.");

private:
    using Accessor2dC = Accessor2dConst<storage_type, arithmetic_type>;

public:
    /**
     * Creates an empty accessor pointing to a nullptr.
     */
    Accessor2d() : Accessor2dC(nullptr, {}, nullptr) {}

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor2d(storage_type *storage, size_type stride, arithmetic_type *scale)
        : Accessor2dC(storage, stride, scale)
    {}

    Accessor2dC to_const() const
    {
        return {this->storage_, this->stride_, this->scale_};
    }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType row, IndexType col,
                              arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = this->storage_;
        const arithmetic_type *GKO_RESTRICT rest_scale = this->scale_;
        rest_storage[row * this->stride_ + col] =
            static_cast<storage_type>(value / rest_scale[col]);
    }

    /**
     * Writes the given value at the given index for a scale.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void set_scale(IndexType idx, arithmetic_type val)
    {
        arithmetic_type *GKO_RESTRICT rest_scale = this->scale_;
        rest_scale[idx] = val;
    }

    GKO_ATTRIBUTES storage_type *get_storage() { return this->storage_; }

    GKO_ATTRIBUTES arithmetic_type *get_scale() { return this->scale_; }
};


template <typename ValueType, typename ValueTypeKrylovBases,
          bool = Accessor2d<ValueTypeKrylovBases, ValueType>::has_scale>
class Accessor2dHelper {};


template <typename ValueType, typename ValueTypeKrylovBases>
class Accessor2dHelper<ValueType, ValueTypeKrylovBases, true> {
    using Accessor = Accessor2d<ValueTypeKrylovBases, ValueType>;
    static_assert(Accessor::has_scale,
                  "This accessor must have a scale in this class!");

public:
    Accessor2dHelper() = default;

    Accessor2dHelper(dim<2> krylov_dim, std::shared_ptr<const Executor> exec)
        : krylov_dim_{krylov_dim},
          bases_stride_{krylov_dim_[1]},
          bases_{exec, krylov_dim_[0] * bases_stride_},
          scale_{exec, krylov_dim_[1]}
    {
        // For testing, initialize scale to ones
        Array<ValueType> h_scale{exec->get_master(), krylov_dim[0]};
        for (size_type i = 0; i < h_scale.get_num_elems(); ++i) {
            h_scale.get_data()[i] = one<ValueType>();
        }
        scale_ = h_scale;
    }

    Accessor get_accessor()
    {
        return {bases_.get_data(), bases_stride_, scale_.get_data()};
    }

    gko::Array<ValueTypeKrylovBases> &get_bases() { return bases_; }

private:
    dim<2> krylov_dim_;
    size_type bases_stride_;
    Array<ValueTypeKrylovBases> bases_;
    Array<ValueType> scale_;
};


template <typename ValueType, typename ValueTypeKrylovBases>
class Accessor2dHelper<ValueType, ValueTypeKrylovBases, false> {
    using Accessor = Accessor2d<ValueTypeKrylovBases, ValueType>;
    static_assert(!Accessor::has_scale,
                  "This accessor must not have a scale in this class!");

public:
    Accessor2dHelper() = default;

    Accessor2dHelper(dim<2> krylov_dim, std::shared_ptr<const Executor> exec)
        : krylov_dim_{krylov_dim},
          bases_stride_{krylov_dim_[1]},
          bases_{exec, krylov_dim_[0] * bases_stride_}
    {}

    Accessor get_accessor() { return {bases_.get_data(), bases_stride_}; }

    gko::Array<ValueTypeKrylovBases> &get_bases() { return bases_; }

private:
    dim<2> krylov_dim_;
    size_type bases_stride_;
    Array<ValueTypeKrylovBases> bases_;
};


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
