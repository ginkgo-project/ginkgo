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

#ifndef GKO_CORE_BASE_ARRAY_HPP_
#define GKO_CORE_BASE_ARRAY_HPP_


#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


namespace detail {


/**
 * @internal
 *
 * Converts `size` elements of type `SourceType` stored at `src` on `exec`
 * to `TargetType` stored at `dst`.
 */
template <typename SourceType, typename TargetType>
void convert_data(const std::shared_ptr<const Executor> &exec, size_type size,
                  const SourceType *src, TargetType *dst);


}  // namespace detail


/**
 * An Array is a container which encapsulates fixed-sized arrays, stored on the
 * Executor tied to the Array.
 * The array stores and transfers its data as __raw__ memory, which means that
 * the constructors of its elements are not called when constructing, copying or
 * moving the Array. Thus, the Array class is most suitable for storing POD
 * types.
 *
 * @tparam ValueType  the type of elements stored in the array
 *
 * @ingroup array
 */
template <typename ValueType>
class Array {
public:
    /**
     * The type of elements stored in the array.
     */
    using value_type = ValueType;

    /**
     * The default deleter type used by Array.
     */
    using default_deleter = executor_deleter<value_type[]>;

    /**
     * The deleter type used for views.
     */
    using view_deleter = null_deleter<value_type[]>;

    /**
     * Creates an empty Array not tied to any executor.
     *
     * An array without an assigned executor can only be empty. Attempts to
     * change its size (e.g. via the resize_and_reset method) will result in an
     * exception. If such an array is used as the right hand side of an
     * assignment or move assignment expression, the data of the target array
     * will be cleared, but its executor will not be modified.
     *
     * The executor can later be set by using the set_executor method. If an
     * Array with no assigned executor is assigned or moved to, it will inherit
     * the executor of the source Array.
     */
    Array() noexcept
        : num_elems_(0),
          data_(nullptr, default_deleter{nullptr}),
          exec_(nullptr)
    {}

    /**
     * Creates an empty Array tied to the specified Executor.
     *
     * @param exec  the Executor where the array data is allocated
     */
    Array(std::shared_ptr<const Executor> exec) noexcept
        : num_elems_(0),
          data_(nullptr, default_deleter{exec}),
          exec_(std::move(exec))
    {}

    /**
     * Creates an Array on the specified Executor.
     *
     * @param exec  the Executor where the array data will be allocated
     * @param num_elems  the amount of memory (expressed as the number of
     *                   `value_type` elements) allocated on the Executor
     */
    Array(std::shared_ptr<const Executor> exec, size_type num_elems)
        : num_elems_(num_elems),
          data_(nullptr, default_deleter{exec}),
          exec_(std::move(exec))
    {
        if (num_elems > 0) {
            data_.reset(exec_->alloc<value_type>(num_elems));
        }
    }

    /**
     * Creates an Array from existing memory.
     *
     * The memory will be managed by the array, and deallocated using the
     * specified deleter (e.g. use std::default_delete for data allocated with
     * new).
     *
     * @tparam DeleterType  type of the deleter
     *
     * @param exec  executor where `data` is located
     * @param num_elems  number of elements in `data`
     * @param data  chunk of memory used to create the array
     * @param deleter  the deleter used to free the memory
     *
     * @see Array::view() to create an array that does not deallocate memory
     * @see Array(std::shared_ptr<cont Executor>, size_type, value_type*) to
     *      deallocate the memory using Executor::free() method
     */
    template <typename DeleterType>
    Array(std::shared_ptr<const Executor> exec, size_type num_elems,
          value_type *data, DeleterType deleter)
        : num_elems_{num_elems}, data_(data, deleter), exec_{exec}
    {}

    /**
     * Creates an Array from existing memory.
     *
     * The memory will be managed by the array, and deallocated using the
     * Executor::free method.
     *
     * @param exec  executor where `data` is located
     * @param num_elems  number of elements in `data`
     * @param data  chunk of memory used to create the array
     */
    Array(std::shared_ptr<const Executor> exec, size_type num_elems,
          value_type *data)
        : Array(exec, num_elems, data, default_deleter{exec})
    {}

    /**
     * Creates an array on the specified Executor and initializes it with
     * values.
     *
     * @tparam RandomAccessIterator  type of the iterators
     *
     * @param exec  the Executor where the array data will be allocated
     * @param begin  start of range of values
     * @param end  end of range of values
     */
    template <typename RandomAccessIterator>
    Array(std::shared_ptr<const Executor> exec, RandomAccessIterator begin,
          RandomAccessIterator end)
        : Array(exec)
    {
        Array tmp(exec->get_master(), std::distance(begin, end));
        std::copy(begin, end, tmp.data_.get());
        *this = std::move(tmp);
    }

    /**
     * Creates an array on the specified Executor and initializes it with
     * values.
     *
     * @tparam T  type of values used to initialize the array (T has to be
     *            implicitly convertible to value_type)
     *
     * @param exec  the Executor where the array data will be allocated
     * @param init_list  list of values used to initialize the Array
     */
    template <typename T>
    Array(std::shared_ptr<const Executor> exec,
          std::initializer_list<T> init_list)
        : Array(exec, begin(init_list), end(init_list))
    {}

    /**
     * Creates a copy of another array on a different executor.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param exec  the executor where the new array will be created
     * @param other  the Array to copy from
     */
    Array(std::shared_ptr<const Executor> exec, const Array &other)
        : Array(exec)
    {
        *this = other;
    }

    /**
     * Creates a copy of another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to copy from
     */
    Array(const Array &other) : Array(other.get_executor(), other) {}

    /**
     * Moves another array to a different executor.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param exec  the executor where the new array will be moved
     * @param other  the Array to move
     */
    Array(std::shared_ptr<const Executor> exec, Array &&other) : Array(exec)
    {
        *this = std::move(other);
    }

    /**
     * Moves another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to move
     */
    Array(Array &&other) : Array(other.get_executor(), std::move(other)) {}

    /**
     * Creates an Array from existing memory.
     *
     * The Array does not take ownership of the memory, and will not deallocate
     * it once it goes out of scope. This array type cannot use the function
     * `resize_and_reset` since it does not own the data it should resize.
     *
     * @param exec  executor where `data` is located
     * @param num_elems  number of elements in `data`
     * @param data  chunk of memory used to create the array
     *
     * @return an Array constructed from `data`
     */
    static Array view(std::shared_ptr<const Executor> exec, size_type num_elems,
                      value_type *data)
    {
        return Array{exec, num_elems, data, view_deleter{}};
    }

    /**
     * Copies data from another array or view. In the case of an array target,
     * the array is resized to match the source's size. In the case of a view
     * target, if the dimensions are not compatible a gko::OutOfBoundsError is
     * thrown.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the Array to copy from
     *
     * @return this
     */
    Array &operator=(const Array &other)
    {
        if (&other == this) {
            return *this;
        }
        if (exec_ == nullptr) {
            exec_ = other.get_executor();
            data_ = data_manager{nullptr, other.data_.get_deleter()};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        if (this->is_owning()) {
            this->resize_and_reset(other.get_num_elems());
        } else {
            GKO_ENSURE_COMPATIBLE_BOUNDS(other.get_num_elems(),
                                         this->num_elems_);
        }
        exec_->copy_from(other.get_executor().get(), other.get_num_elems(),
                         other.get_const_data(), this->get_data());
        return *this;
    }

    /**
     * Moves data from another array or view. Only the pointer and deleter type
     * change, a copy only happens when targeting another executor's data. This
     * means that in the following situation:
     * ```cpp
     *   gko::Array<int> a; // an existing array or view
     *   gko::Array<int> b; // an existing array or view
     *   b = std::move(a);
     * ```
     * Depending on whether `a` and `b` are array or view, this happens:
     * + `a` and `b` are views, `b` becomes the only valid view of `a`;
     * + `a` and `b` are arrays, `b` becomes the only valid array of `a`;
     * + `a` is a view and `b` is an array, `b` frees its data and becomes the
     *    only valid view of `a` ();
     * + `a` is an array and `b` is a view, `b` becomes the only valid array
     *    of `a`.
     *
     * In all the previous cases, `a` becomes invalid (e.g., a `nullptr`).
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the Array to move data from
     *
     * @return this
     */
    Array &operator=(Array &&other)
    {
        if (&other == this) {
            return *this;
        }
        if (exec_ == nullptr) {
            exec_ = other.get_executor();
            data_ = data_manager{nullptr, other.data_.get_deleter()};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }
        if (exec_ == other.get_executor()) {
            // same device, only move the pointer
            using std::swap;
            swap(data_, other.data_);
            swap(num_elems_, other.num_elems_);
            other.clear();
        } else {
            // different device, copy the data
            *this = other;
        }
        return *this;
    }

    /**
     * Copies and converts data from another array with another data type.
     * In the case of an array target, the array is resized to match the
     * source's size. In the case of a view target, if the dimensions are not
     * compatible a gko::OutOfBoundsError is thrown.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the Array to copy from
     * @tparam OtherValueType  the value type of `other`
     *
     * @return this
     */
    template <typename OtherValueType>
    xstd::enable_if_t<!std::is_same<ValueType, OtherValueType>::value, Array>
        &operator=(const Array<OtherValueType> &other)
    {
        if (this->exec_ == nullptr) {
            this->exec_ = other.get_executor();
            this->data_ = data_manager{nullptr, default_deleter{this->exec_}};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        if (this->is_owning()) {
            this->resize_and_reset(other.get_num_elems());
        } else {
            GKO_ENSURE_COMPATIBLE_BOUNDS(other.get_num_elems(),
                                         this->num_elems_);
        }
        Array<OtherValueType> tmp{this->exec_};
        const OtherValueType *source = other.get_const_data();
        // if we are on different executors: copy, then convert
        if (this->exec_ != other.get_executor()) {
            tmp = other;
            source = tmp.get_const_data();
        }
        detail::convert_data(this->exec_, other.get_num_elems(), source,
                             this->get_data());
        return *this;
    }

    /**
     * Deallocates all data used by the Array.
     *
     * The array is left in a valid, but empty state, so the same array can be
     * used to allocate new memory. Calls to Array::get_data() will return
     * a `nullptr`.
     */
    void clear() noexcept
    {
        num_elems_ = 0;
        data_.reset(nullptr);
    }

    /**
     * Resizes the array so it is able to hold the specified number of elements.
     * For a view and other non-owning Array types, this throws an exception
     * since these types cannot be resized.
     *
     * All data stored in the array will be lost.
     *
     * If the Array is not assigned an executor, an exception will be thrown.
     *
     * @param num_elems  the amount of memory (expressed as the number of
     *                   `value_type` elements) allocated on the Executor
     */
    void resize_and_reset(size_type num_elems)
    {
        if (num_elems == num_elems_) {
            return;
        }
        if (exec_ == nullptr) {
            throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                    "gko::Executor (nullptr)");
        }
        if (!this->is_owning()) {
            throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                    "Non owning gko::Array cannot be resized.");
        }

        if (num_elems > 0 && this->is_owning()) {
            num_elems_ = num_elems;
            data_.reset(exec_->alloc<value_type>(num_elems));
        } else {
            this->clear();
        }
    }

    /**
     * Returns the number of elements in the Array.
     *
     * @return the number of elements in the Array
     */
    size_type get_num_elems() const noexcept { return num_elems_; }

    /**
     * Returns a pointer to the block of memory used to store the elements of
     * the Array.
     *
     * @return a pointer to the block of memory used to store the elements of
     * the Array
     */
    value_type *get_data() noexcept { return data_.get(); }

    /**
     * Returns a constant pointer to the block of memory used to store the
     * elements of the Array.
     *
     * @return a constant pointer to the block of memory used to store the
     * elements of the Array
     */
    const value_type *get_const_data() const noexcept { return data_.get(); }

    /**
     * Returns the Executor associated with the array.
     *
     * @return the Executor associated with the array
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Changes the Executor of the Array, moving the allocated data to the new
     * Executor.
     *
     * @param exec  the Executor where the data will be moved to
     */
    void set_executor(std::shared_ptr<const Executor> exec)
    {
        if (exec == exec_) {
            // moving to the same executor, no-op
            return;
        }
        Array tmp(std::move(exec));
        tmp = *this;
        exec_ = std::move(tmp.exec_);
        data_ = std::move(tmp.data_);
    }

    /**
     * Tells whether this Array owns its data or not.
     *
     * Views do not own their data and this has multiple implications. They
     * cannot be resized since the data is not owned by the Array which stores a
     * view. It is also unclear whether custom deleter types are owning types as
     * they could be a user-created view-type, therefore only proper Array which
     * use the `default_deleter` are considered owning types.
     *
     * @return whether this Array can be resized or not.
     */
    bool is_owning()
    {
        return data_.get_deleter().target_type() == typeid(default_deleter);
    }


private:
    // Allow other Array types to access private members
    template <typename OtherValueType>
    friend class Array;

    using data_manager =
        std::unique_ptr<value_type[], std::function<void(value_type[])>>;

    size_type num_elems_;
    data_manager data_;
    std::shared_ptr<const Executor> exec_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_ARRAY_HPP_
