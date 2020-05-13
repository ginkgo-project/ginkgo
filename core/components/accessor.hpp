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

#ifndef GKO_CORE_COMMON_ACCESSOR_HPP_
#define GKO_CORE_COMMON_ACCESSOR_HPP_


#include <cinttypes>


#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * @internal
 *
 * The Accessor class hides the underlying storage_ format and provides a simple
 * interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType = double>
class Accessor {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;

    /**
     * Creates the accessor with an already allocated storage space.
     */
    Accessor(storage_type *storage) : storage_{storage} {}

    /**
     * Reads the value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES arithmetic_type read(IndexType idx) const
    {
        // Make use of the restrict (and const) qualifier. If the restrict
        // qualifier would have been put on the class attribute, it would be
        // ignored by nvcc.
        const storage_type *GKO_RESTRICT rest_storage = storage_;
        return static_cast<arithmetic_type>(rest_storage[idx]);
    }

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType idx, arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = storage_;
        rest_storage[idx] = static_cast<storage_type>(value);
    }

private:
    storage_type *storage_;
};


template <typename StorageType, typename ArithmeticType>
class Accessor2d;


/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType = double>
class Accessor2dConst {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor2dConst(const storage_type *storage, size_type stride)
        : storage_{storage}, stride_{stride}
    {}

    /**
     * Creates a const accessor from a non-const accessor.
     */
    Accessor2dConst(Accessor2d<StorageType, ArithmeticType> acc)
        : storage_{acc.get_const_storage()}, stride_{acc.get_stride()}
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

private:
    const storage_type *storage_;
    size_type stride_;
};


/**
 * @internal
 *
 * The Accessor2d class hides the underlying storage_ format and provides a
 * simple interface for accessing a one dimensional storage.
 *
 * This class only manages the accesses, however, and not the memory itself.
 */
template <typename StorageType, typename ArithmeticType = double>
class Accessor2d {
public:
    using storage_type = StorageType;
    using arithmetic_type = ArithmeticType;

    /**
     * Creates an empty accessor pointing to a nullptr.
     */
    Accessor2d() : storage_{nullptr}, stride_{} {}

    /**
     * Creates the accessor with an already allocated storage space with a
     * stride.
     */
    Accessor2d(storage_type *storage, size_type stride)
        : storage_{storage}, stride_{stride}
    {}

    operator Accessor2dConst<storage_type, arithmetic_type>() const
    {
        return {storage_, stride_};
    }

    Accessor2dConst<storage_type, arithmetic_type> to_const() const
    {
        return {storage_, stride_};
    }

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

    /**
     * Writes the given value at the given index.
     */
    template <typename IndexType>
    GKO_ATTRIBUTES void write(IndexType row, IndexType col,
                              arithmetic_type value)
    {
        storage_type *GKO_RESTRICT rest_storage = storage_;
        rest_storage[row * stride_ + col] = static_cast<storage_type>(value);
    }

    GKO_ATTRIBUTES size_type get_stride() const { return stride_; }

    GKO_ATTRIBUTES storage_type *get_storage() { return storage_; }

    GKO_ATTRIBUTES const storage_type *get_const_storage() const
    {
        return storage_;
    }

private:
    storage_type *storage_;
    size_type stride_;
};


}  // namespace gko


#endif  // GKO_CORE_COMMON_ACCESSOR_HPP_
